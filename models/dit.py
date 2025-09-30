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
  def __init__(self, dim, n_heads, cond_dim, cond_dim_embedding, mlp_ratio=4, dropout=0.1, use_residual_modulation: bool = False, use_weighted_sum: bool = True):
    super().__init__()
    self.n_heads = n_heads
    self.use_weighted_sum = bool(use_weighted_sum)

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
    
    # Condition embedding conditioning
    self.cond_embedding_modulation = nn.Linear(cond_dim_embedding, 6 * dim, bias=True)

    # Optional: residual FiLM from (condition - curr_embed) in the ORIGINAL cond space
    self.use_residual_modulation = bool(use_residual_modulation)
    if self.use_residual_modulation:
      # Map residual_error (B, cond_dim) -> 6 * dim deltas; zero-init to preserve old behavior
      self.residual_embedding_modulation = nn.Linear(cond_dim_embedding, 6 * dim, bias=True)
      init.xavier_uniform_(self.residual_embedding_modulation.weight)
      init.zeros_(self.residual_embedding_modulation.bias)
      
      self.curr_embed_modulation = nn.Linear(cond_dim_embedding, 6 * dim, bias=True)
      init.xavier_uniform_(self.curr_embed_modulation.weight)
      init.zeros_(self.curr_embed_modulation.bias)
      
      # coefficients in weighted sum
      self.res_w = nn.Parameter(torch.ones(6, 2)) if use_weighted_sum else torch.ones(6, 2) * 0.5
    else:
      self.residual_embedding_modulation = None
      self.res_w = None

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, rotary_cos_sin, c, cond, seqlens=None, curr_embed=None, residual=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    # print('\n\n\n\n\n\n')
    # print(f"x: {x.shape}")
    # print(f"c: {c.shape}")
    # print(f"cond: {cond.shape}")
    # print(f"seqlens: {seqlens.shape if seqlens is not None else None}")
    # print(f"curr_embed: {curr_embed.shape if curr_embed is not None else None}")
    # print(f"residual: {residual.shape if residual is not None else None}")
    # print('\n\n\n\n\n\n')
    
    # Process sigma conditioning
    (shift_msa_sigma, scale_msa_sigma, gate_msa_sigma, shift_mlp_sigma,
     scale_mlp_sigma, gate_mlp_sigma) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    
    # Process condition embedding conditioning
    (shift_msa_cond, scale_msa_cond, gate_msa_cond, shift_mlp_cond,
     scale_mlp_cond, gate_mlp_cond) = self.cond_embedding_modulation(cond)[:, None].chunk(6, dim=2)
    
    # print(f"shift_msa_sigma: {shift_msa_sigma.shape}")
    # print(f"scale_msa_sigma: {scale_msa_sigma.shape}")
    # print(f"gate_msa_sigma: {gate_msa_sigma.shape}")
    # print(f"shift_mlp_sigma: {shift_mlp_sigma.shape}")
    # print(f"scale_mlp_sigma: {scale_mlp_sigma.shape}")
    # print(f"gate_mlp_sigma: {gate_mlp_sigma.shape}")
    # print(f"shift_msa_cond: {shift_msa_cond.shape}")
    # print(f"scale_msa_cond: {scale_msa_cond.shape}")
    # print(f"gate_msa_cond: {gate_msa_cond.shape}")
    # print(f"shift_mlp_cond: {shift_mlp_cond.shape}")
    # print(f"scale_mlp_cond: {scale_mlp_cond.shape}")
    # print(f"gate_mlp_cond: {gate_mlp_cond.shape}")
    # print('\n\n\n\n\n\n')
    
    # Combine conditioning parameters (base)
    shift_msa = shift_msa_sigma + shift_msa_cond
    scale_msa = scale_msa_sigma + scale_msa_cond
    gate_msa = gate_msa_sigma + gate_msa_cond
    shift_mlp = shift_mlp_sigma + shift_mlp_cond
    scale_mlp = scale_mlp_sigma + scale_mlp_cond
    gate_mlp = gate_mlp_sigma + gate_mlp_cond

    # Optional residual FiLM: apply deltas derived from residual_error = condition - curr_embed
    if (self.residual_embedding_modulation is not None) and (residual is not None):
      # residual has shape (B, cond_dim)
      # print(f"residual: {residual.shape}")
      # print(f"residual_embedding_modulation: {self.residual_embedding_modulation.weight.shape}")
      (res_shift_msa, res_scale_msa, res_gate_msa,
       res_shift_mlp, res_scale_mlp, res_gate_mlp) = self.residual_embedding_modulation(residual)[:, None].chunk(6, dim=2)

      # curr_embed has shape (B, cond_dim)
      (curr_shift_msa, curr_scale_msa, curr_gate_msa,
       curr_shift_mlp, curr_scale_mlp, curr_gate_mlp) = self.curr_embed_modulation(curr_embed)[:, None].chunk(6, dim=2)
      
      # apply weighted sum
      shift_msa = shift_msa + self.res_w[0, 0] * res_shift_msa + self.res_w[0, 1] * curr_shift_msa  
      scale_msa = scale_msa + self.res_w[1, 0] * res_scale_msa + self.res_w[1, 1] * curr_scale_msa
      gate_msa  = gate_msa  + self.res_w[2, 0] * res_gate_msa  + self.res_w[2, 1] * curr_gate_msa
      shift_mlp = shift_mlp + self.res_w[3, 0] * res_shift_mlp + self.res_w[3, 1] * curr_shift_mlp
      scale_mlp = scale_mlp + self.res_w[4, 0] * res_scale_mlp + self.res_w[4, 1] * curr_scale_mlp
      gate_mlp  = gate_mlp  + self.res_w[5, 0] * res_gate_mlp  + self.res_w[5, 1] * curr_gate_mlp

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
  def __init__(self, hidden_size, out_channels, cond_dim, cond_dim_embedding):
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
    
    # Condition embedding conditioning
    self.cond_embedding_modulation = nn.Linear(cond_dim_embedding,
                                               2 * hidden_size,
                                               bias=True)

  def forward(self, x, c, cond):
    # Process sigma conditioning
    shift_sigma, scale_sigma = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    
    # Process condition embedding conditioning
    shift_cond, scale_cond = self.cond_embedding_modulation(cond)[:, None].chunk(2, dim=2)
    
    # Combine conditioning parameters
    shift = shift_sigma + shift_cond
    scale = scale_sigma + scale_cond
    
    x_mod = modulate_fused(self.norm_final(x), shift, scale)
    x_main = self.linear(x_mod)  # (..., out_channels - 1)
    x_extra = self.linear_extra(x_mod)  # (..., 1)
    x_out = torch.cat([x_main, x_extra], dim=-1)
    return x_out


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int, cond_dim: int = None, use_residual_modulation: bool = False, use_weighted_sum: bool = True):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    print('\n\n\n\n\n\n')
    print(use_residual_modulation, use_weighted_sum, 
          '\n\n\n\n\n\n')
  
    self.config = config
    self.vocab_size = vocab_size
    self.use_residual_modulation = bool(use_residual_modulation)
    self.use_weighted_sum = bool(use_weighted_sum) and self.use_residual_modulation

    self.vocab_embed = EmbeddingLayer(config.model.hidden_size,
                                      vocab_size)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    if cond_dim is not None:
      self.cond_embed = nn.Linear(cond_dim, config.model.cond_dim_embedding)
      print(f"\n\n\ncond_embed: {config.model.cond_dim_embedding}\n\n\n")
    else:
      self.cond_embed = None
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTBlock(config.model.hidden_size,
                              config.model.n_heads,
                              config.model.cond_dim,
                              config.model.cond_dim_embedding,
                              dropout=config.model.dropout,
                              use_residual_modulation=self.use_residual_modulation,
                              use_weighted_sum=self.use_weighted_sum))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim,
      config.model.cond_dim_embedding)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, indices, sigma, condition, curr_embed: typing.Optional[torch.Tensor] = None):
    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))

    # If unconditional (no cond_embed or no condition), use zeros
    if self.cond_embed is None or condition is None:
      cond = torch.zeros((x.shape[0], self.config.model.cond_dim_embedding), device=x.device, dtype=x.dtype)
      residual = None
    else:
      cond = F.silu(self.cond_embed(condition))
      # Apply condition dropout during training
      if self.training and self.config.text_embedder.cond_dropout > 0:
        batch_size = cond.shape[0]
        dropout_mask = torch.rand(batch_size, 1, device=cond.device) >= self.config.text_embedder.cond_dropout
        dropout_std = float(self.config.text_embedder.cond_dropout_std)
        if dropout_std > 0.0:
          noise = torch.randn_like(cond) * dropout_std
          cond = cond * dropout_mask.float() + noise * (~dropout_mask).float()
        else:
          cond = cond * dropout_mask.float()
      # Compute residual only if feature is enabled AND curr_embed is provided
      if self.use_residual_modulation and (curr_embed is not None):
        curr_embed = F.silu(self.cond_embed(curr_embed))
        
        # residual in ORIGINAL cond space (same shape as `condition`): (B, cond_dim)
        residual = cond - curr_embed
      else:
        residual = None
    
    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, cond, seqlens=None, curr_embed=curr_embed, residual=residual)
      x = self.output_layer(x, c, cond)

    return x
