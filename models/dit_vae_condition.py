"""
DiT with VAE-Style Conditioning

This module implements a Diffusion Transformer (DiT) with sophisticated VAE-style conditioning.
Instead of using simple linear projection for conditioning, this implementation uses:

1. **Learnable Latent Vectors**: A small set of learnable latent vectors (typically 1 for 
   single embedding conditioning) that are initialized and learned during training.

2. **Cross-Attention Mechanism**: The latents attend to the input condition embeddings through
   a multi-head cross-attention mechanism, allowing them to extract rich, contextual information.

3. **Transformer Encoding**: Multiple transformer blocks with residual connections process the
   latents through cross-attention and feed-forward networks, producing sophisticated 
   conditioning representations.

4. **Normalization**: Uses RMSNorm (Root Mean Square Normalization) for better training stability
   and convergence compared to LayerNorm.

This architecture is inspired by VAE encoders where learnable latents compress input information
into a compact, information-rich representation. The key advantage is that the learned latents
can capture complex patterns in the conditioning signal through the attention mechanism.

Architecture Flow:
  condition [B, cond_dim] 
    -> expand to [B, 1, cond_dim]
    -> initialize learnable latents [B, num_latents, latent_dim]
    -> for each layer:
         latents attend to condition (cross-attention)
         latents -> FFN
    -> output latents [B, num_latents, latent_dim]
    -> used as conditioning for diffusion

Configuration Parameters (can be set in config.model):
  - cond_encoder_layers: Number of transformer layers (default: 2)
  - cond_encoder_hidden: Hidden size for attention (default: max(512, hidden_size))
  - cond_encoder_heads: Number of attention heads (default: 8)
  - cond_encoder_ff_mult: Feed-forward multiplier (default: 4)
"""

import math
import typing

import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.nn.init as init
from copy import deepcopy

import dataloader

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.cuda.amp.autocast(enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#                    VAE-Style Conditioning Components                           #
#################################################################################
class RMSNorm(nn.Module):
  """Root Mean Square Layer Normalization"""
  def __init__(self, dim: int, eps: float = 1e-8):
    super().__init__()
    self.scale = dim ** -0.5
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
    return x / norm.clamp(min=self.eps) * self.gamma


class FeedForwardNetwork(nn.Module):
  """Feed-forward network with GELU activation"""
  def __init__(self, embedding_dim: int, mult: int = 4, dropout_p: float = 0.1):
    super().__init__()
    hidden_dim = int(embedding_dim * mult)
    self.ffd = nn.Sequential(
      nn.LayerNorm(embedding_dim),
      nn.Linear(embedding_dim, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout_p),
      nn.Linear(hidden_dim, embedding_dim)
    )

  def forward(self, x):
    return self.ffd(x)


class LatentCrossAttention(nn.Module):
  """Cross-attention where latents attend to condition embeddings"""
  def __init__(self, 
               latent_dim: int,
               cond_dim: int, 
               hidden_size: int,
               num_heads: int = 8,
               dropout_p: float = 0.1,
               qk_norm: bool = True):
    super().__init__()
    
    self.dim_head = hidden_size // num_heads
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.latent_dim = latent_dim
    self.cond_dim = cond_dim
    self.dropout_p = dropout_p
    self.qk_norm = qk_norm
    
    # Normalization
    self.norm_cond = RMSNorm(cond_dim, eps=1e-6)
    self.norm_latents = RMSNorm(latent_dim, eps=1e-6)
    
    # Q, K, V projections
    self.cond_to_KV = nn.Linear(cond_dim, hidden_size * 2, bias=False)
    self.latents_to_Q = nn.Linear(latent_dim, hidden_size, bias=False)
    self.latents_to_KV = nn.Linear(latent_dim, hidden_size * 2, bias=False)
    
    # QK normalization
    self.query_norm = RMSNorm(self.dim_head, eps=1e-6) if qk_norm else nn.Identity()
    self.key_norm = RMSNorm(self.dim_head, eps=1e-6) if qk_norm else nn.Identity()
    
    # Output projection
    self.projector = nn.Linear(hidden_size, latent_dim, bias=False)
    self.proj_dropout = nn.Dropout(dropout_p)

  def forward(self, 
              cond_embeddings,      # [B, seq_len, cond_dim]
              latents,              # [B, num_latents, latent_dim]
              cond_mask=None):      # [B, seq_len]
    batch_size = latents.shape[0]
    num_latents = latents.shape[1]
    
    # Normalize
    cond_embeddings = self.norm_cond(cond_embeddings)
    latents = self.norm_latents(latents)
    
    # Project to Q, K, V
    q = self.latents_to_Q(latents)  # [B, num_latents, hidden_size]
    
    kv_cond = self.cond_to_KV(cond_embeddings)  # [B, seq_len, hidden_size * 2]
    kv_latents = self.latents_to_KV(latents)    # [B, num_latents, hidden_size * 2]
    
    # Concatenate latents and condition KV (latents first for self-attention)
    kv = torch.cat([kv_latents, kv_cond], dim=1)  # [B, num_latents + seq_len, hidden_size * 2]
    k, v = kv.split(self.hidden_size, dim=2)
    
    # Reshape for multi-head attention
    q = q.view(batch_size, num_latents, self.num_heads, self.dim_head).transpose(1, 2)
    k = k.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
    v = v.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
    
    # Apply QK normalization
    if self.qk_norm:
      q = self.query_norm(q)
      k = self.key_norm(k)
    
    # Create attention mask if needed
    attn_mask = None
    if cond_mask is not None:
      # Latents can attend to themselves (all ones) and to non-masked condition tokens
      latent_mask = torch.ones((batch_size, num_latents), device=latents.device, dtype=cond_mask.dtype)
      full_mask = torch.cat([latent_mask, cond_mask], dim=1)  # [B, num_latents + seq_len]
      
      # Expand mask for attention: [B, 1, num_latents, num_latents + seq_len]
      attn_mask = full_mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, num_latents, -1)
      attn_mask = (1.0 - attn_mask) * torch.finfo(q.dtype).min
    
    # Scaled dot-product attention
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
      y = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask,
        dropout_p=self.dropout_p if self.training else 0.0,
        is_causal=False,
      )
    
    # Reshape back
    y = y.transpose(1, 2).contiguous().view(batch_size, num_latents, self.hidden_size)
    
    # Output projection
    output = self.proj_dropout(self.projector(y))
    return output


class ConditionEncoderBlock(nn.Module):
  """Transformer block for condition encoding with cross-attention"""
  def __init__(self, 
               latent_dim: int,
               cond_dim: int,
               hidden_size: int,
               num_heads: int = 8,
               ff_mult: int = 4,
               dropout: float = 0.1):
    super().__init__()
    
    self.cross_attn = LatentCrossAttention(
      latent_dim=latent_dim,
      cond_dim=cond_dim,
      hidden_size=hidden_size,
      num_heads=num_heads,
      dropout_p=dropout,
      qk_norm=True
    )
    
    self.ffn = FeedForwardNetwork(
      embedding_dim=latent_dim,
      mult=ff_mult,
      dropout_p=dropout
    )
    
  def forward(self, cond_embeddings, latents, cond_mask=None):
    # Cross-attention with residual
    latents = latents + self.cross_attn(cond_embeddings, latents, cond_mask)
    # Feed-forward with residual
    latents = latents + self.ffn(latents)
    return latents


class ConditionEncoder(nn.Module):
  """
  Sophisticated condition encoder that uses learnable latents with cross-attention.
  Similar to VAE encoder architecture for rich conditioning.
  """
  def __init__(self,
               cond_dim: int,              # Input condition dimension
               latent_dim: int,            # Output latent dimension (hidden_size)
               num_latents: int = 1,       # Number of latent vectors (1 for single embedding)
               num_layers: int = 2,        # Number of transformer layers
               hidden_size: int = 512,     # Hidden size for attention
               num_heads: int = 8,         # Number of attention heads
               ff_mult: int = 4,           # Feed-forward multiplier
               dropout: float = 0.1):
    super().__init__()
    
    self.num_latents = num_latents
    self.latent_dim = latent_dim
    self.cond_dim = cond_dim
    
    # Learnable latent vectors
    self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
    nn.init.normal_(self.latents, std=0.02)
    
    # Input projection (if cond_dim != latent_dim)
    self.cond_proj = nn.Identity() if cond_dim == latent_dim else nn.Linear(cond_dim, cond_dim)
    
    # Initial normalization
    self.latent_norm = RMSNorm(latent_dim, eps=1e-6)
    
    # Transformer layers
    self.layers = nn.ModuleList([
      ConditionEncoderBlock(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        hidden_size=hidden_size,
        num_heads=num_heads,
        ff_mult=ff_mult,
        dropout=dropout
      ) for _ in range(num_layers)
    ])
    
    # Output normalization
    self.output_norm = RMSNorm(latent_dim, eps=1e-6)
    
  def forward(self, condition, cond_mask=None):
    """
    Args:
      condition: [B, seq_len, cond_dim] or [B, cond_dim] condition embeddings
      cond_mask: [B, seq_len] optional mask for condition tokens
      
    Returns:
      latents: [B, num_latents, latent_dim] processed latent representations
    """
    # Handle 2D input by adding sequence dimension
    if condition.dim() == 2:
      condition = condition.unsqueeze(1)  # [B, 1, cond_dim]
      if cond_mask is not None:
        cond_mask = cond_mask.unsqueeze(1)
    
    batch_size = condition.shape[0]
    
    # Project condition if needed
    cond_embeddings = self.cond_proj(condition)
    
    # Initialize latents
    latents = self.latents.view(1, self.num_latents, self.latent_dim).repeat(batch_size, 1, 1)
    latents = self.latent_norm(latents)
    
    # Process through transformer layers
    for layer in self.layers:
      latents = layer(cond_embeddings, latents, cond_mask)
    
    # Final normalization
    latents = self.output_norm(latents)
    
    return latents


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    # Sigma conditioning (timestep)
    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()
    
  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, rotary_cos_sin, c, seqlens=None, curr_embed=None, residual=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    # Process sigma conditioning
    
    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c).reshape(x.shape[0], 1, -1).chunk(6, dim=2)

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv,
                    'b s (three h d) -> b s three h d',
                    three=3,
                    h=self.n_heads)
    with torch.cuda.amp.autocast(enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    qkv = rearrange(qkv, 'b s ... -> (b s) ...')
    if seqlens is None:
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=qkv.device)
    else:
      cu_seqlens = seqlens.cumsum(-1)
    x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
      qkv, cu_seqlens, seq_len, 0., causal=False)
    
    x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x



class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.dim = dim
    self.vocab_dim = vocab_dim
    # Main embeddings parameter, possibly loaded from checkpoint (vocab_dim-1)
    self.embeddings = nn.Parameter(torch.empty((vocab_dim - 1, dim)))
    torch.nn.init.kaiming_uniform_(self.embeddings, a=math.sqrt(5))
    # New embedding for the last token
    self.new_embedding = nn.Parameter(torch.empty((1, dim)))
    torch.nn.init.kaiming_uniform_(self.new_embedding, a=math.sqrt(5))

  def forward(self, x):
    # If all indices are < vocab_dim-1, just use embeddings
    if torch.all(x < self.vocab_dim - 1):
      return self.embeddings[x]
    # Otherwise, need to handle the new embedding
    # Create a full embedding matrix on the fly
    full_embedding = torch.cat([self.embeddings, self.new_embedding], dim=0)
    return full_embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    # As before: main projection for all but the last token
    self.linear = nn.Linear(hidden_size, out_channels - 1)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    # One more projection for the extra token
    self.linear_extra = nn.Linear(hidden_size, 1)
    self.linear_extra.weight.data.zero_()
    self.linear_extra.bias.data.zero_()

    # Sigma conditioning (timestep)
    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    # Process sigma conditioning
    shift, scale = self.adaLN_modulation(c).reshape(x.shape[0], 1, -1).chunk(2, dim=2)
    
    x_mod = modulate_fused(self.norm_final(x), shift, scale)
    x_main = self.linear(x_mod)  # (..., out_channels - 1)
    x_extra = self.linear_extra(x_mod)  # (..., 1)
    x_out = torch.cat([x_main, x_extra], dim=-1)
    return x_out


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int, cond_dim: int = None, *args, **kwargs):
    super().__init__()
    
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)
  
    self.config = config
    self.vocab_size = vocab_size

    self.vocab_embed = EmbeddingLayer(config.model.hidden_size,
                                      vocab_size)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    
    # VAE-style sophisticated conditioning encoder
    if cond_dim is not None:
      # Use ConditionEncoder with learnable latents and cross-attention
      # Configuration can be adjusted based on performance
      self.use_vae_conditioning = True
      self.cond_encoder = ConditionEncoder(
        cond_dim=cond_dim,
        latent_dim=config.model.hidden_size,
        num_latents=1,  # Single latent for conditioning (as in VAE training)
        num_layers=getattr(config.model, 'cond_encoder_layers', 2),  # Default 2 layers
        hidden_size=getattr(config.model, 'cond_encoder_hidden', max(512, config.model.hidden_size)),
        num_heads=getattr(config.model, 'cond_encoder_heads', 8),
        ff_mult=getattr(config.model, 'cond_encoder_ff_mult', 4),
        dropout=config.model.dropout
      )
    else:
      self.use_vae_conditioning = False
      self.cond_encoder = None
    
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTBlock(config.model.hidden_size,
                              config.model.n_heads,
                              config.model.cond_dim,
                              dropout=config.model.dropout))
    
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, indices, sigma, condition, curr_embed: typing.Optional[torch.Tensor] = None, *args, **kwargs):
    x = self.vocab_embed(indices)       # [B x T x d]
    c = F.silu(self.sigma_map(sigma))   # [B x d]

    if self.use_vae_conditioning: 
      if condition is None:
        # Create zero condition for unconditional generation
        cond = torch.zeros((x.shape[0], 1, self.config.model.hidden_size), device=x.device, dtype=x.dtype)
      else:
        # Process condition through VAE-style encoder
        # condition: [B, cond_dim] or [B, seq_len, cond_dim]
        latents = self.cond_encoder(condition)  # [B, num_latents, latent_dim]
        
        # Apply SiLU activation for consistency with original code
        cond = F.silu(latents)  # [B, num_latents, hidden_size]
        
        # If num_latents > 1, take the first one or mean pool
        # For VAE training with single latent, this is already [B, 1, hidden_size]
        if cond.shape[1] > 1:
          # Option 1: Take first latent
          cond = cond[:, 0:1, :]
          # Option 2: Mean pool all latents
          # cond = cond.mean(dim=1, keepdim=True)
        
      # cond.shape = [B x 1 x d]
      x = torch.cat([cond, x], dim=1)

    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      x = self.output_layer(x, c)

    # Remove conditioning token from output
    if self.use_vae_conditioning:
      return x[:, 1:, :]
    else:
      return x
