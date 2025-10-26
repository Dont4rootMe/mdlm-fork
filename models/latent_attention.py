import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import DictConfig

from .blocks import RMSNorm


class LatentAttention(nn.Module):
    def __init__(self, cfg: DictConfig, latents_first: bool = True):
        super().__init__()

        # Use cfg values directly from yaml
        self.dim_head = cfg.attention.head_size
        self.num_attention_heads = cfg.attention.num_heads
        self.hidden_size = cfg.hidden.size
        self.embedding_dim = cfg.embedding.dim
        self.latent_dim = cfg.latent.dim
        self.dropout_p = cfg.attention.probs_dropout
        self.latents_first = latents_first
        self.layer_norm_eps = cfg.normalization.layer_eps
        self.qk_norm = cfg.attention.qk_norm

        # normalization
        self.norm_embeddings = RMSNorm(self.embedding_dim, eps=self.layer_norm_eps)
        self.norm_latents = RMSNorm(self.latent_dim, eps=self.layer_norm_eps)


        # initialize key, query, value matrix in one batch for optimization
        self.embeddings_to_KV = nn.Linear(self.embedding_dim, self.hidden_size * 2, bias = False)
        
        self.latents_to_Q = nn.Linear(self.latent_dim, self.hidden_size, bias = False)
        self.latents_to_KV = nn.Linear(self.latent_dim, self.hidden_size * 2, bias = False)

        self.query_norm = RMSNorm(self.dim_head, eps=self.layer_norm_eps) if self.qk_norm else nn.Identity()
        self.key_norm = RMSNorm(self.dim_head, eps=self.layer_norm_eps) if self.qk_norm else nn.Identity()
        
        # output projection
        self.projector_latents = nn.Linear(self.hidden_size, self.latent_dim, bias = False)
        
        # regularization using dropout
        self.proj_dropout = nn.Dropout(self.dropout_p)

    def forward(self, 
                hidden_state_of_embs, 
                hidden_state_of_latents, 
                mask_tokens, 
                mask_latents
        ):
        batch_size = hidden_state_of_latents.shape[0] # batch size, sequence length
        seq_len = hidden_state_of_latents.shape[1]
        
        # normalization
        hidden_state_of_embs = self.norm_embeddings(hidden_state_of_embs)
        hidden_state_of_latents = self.norm_latents(hidden_state_of_latents)


        # calculate query, key, values for all heads in batch and split batch into three parts for q, k, v
        # move head forward to be the batch dim
        q = self.latents_to_Q(hidden_state_of_latents)
        
        kv_embeddings = self.embeddings_to_KV(hidden_state_of_embs)
        kv_latents = self.latents_to_KV(hidden_state_of_latents)
        
        if self.latents_first:
            kv = torch.cat((kv_latents, kv_embeddings), dim=1)
            mask = torch.cat((mask_latents, mask_tokens), dim=1)
        else:
            kv = torch.cat((kv_embeddings, kv_latents), dim=1)
            mask = torch.cat((mask_tokens, mask_latents), dim=1)
        k, v = kv.split(self.hidden_size, dim=2)
            
        # Reshape [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, self.n_head, hidden_size // self.n_head]
        # Transpose [batch_size, seq_len, self.n_head, hidden_size // self.n_head] -> [batch_size, self.n_head, seq_len, hidden_size // self.n_head]
        # in order to calculate attention over different heads
        q = q.view(batch_size, q.shape[1], self.num_attention_heads, self.dim_head).transpose(1, 2) 
        k = k.view(batch_size, k.shape[1], self.num_attention_heads, self.dim_head).transpose(1, 2) 
        v = v.view(batch_size, v.shape[1], self.num_attention_heads, self.dim_head).transpose(1, 2) 

        if self.qk_norm:
            q = self.query_norm(q)
            k = self.key_norm(k)
        
        # Apply masking to attention scores, fill -inf to attention scores where mask is false
        mask = mask.view(mask.shape[0], 1, mask.shape[1]).repeat(1, seq_len, 1)
        mask = mask.view(mask.shape[0], 1, mask.shape[1], mask.shape[2])
        mask = (1.0 - mask) * torch.finfo(q.dtype).min

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            y = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
                scale=None,
            )
        
        y = y.transpose(1, 2).contiguous().view(
            batch_size, 
            seq_len, 
            self.hidden_size
        ) 

        # Apply output projection & dropout
        hidden_state_of_latents = self.proj_dropout(self.projector_latents(y))
        
        return hidden_state_of_latents