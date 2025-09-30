import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from transformers import AutoModel
from copy import deepcopy
from omegaconf import DictConfig

from .blocks import AbsolutePositionalEmbedding, RMSNorm, FeedForwardNetwork
from .latent_attention import LatentAttention


class EncoderTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        
        self.latent_attn = LatentAttention(
            cfg=cfg,
            latents_first=True
        )
        
        self.ffn_latent = FeedForwardNetwork(
            embedding_dim=cfg.latent.dim,
            mult=cfg.hidden.ff_mult,
            dropout_p=cfg.hidden.dropout
        )
        
    def forward(self, hidden_state_of_embs, hidden_state_of_latents, mask_tokens, mask_latents):
        hidden_state_of_latents_add = self.latent_attn(
            hidden_state_of_embs=hidden_state_of_embs, 
            hidden_state_of_latents=hidden_state_of_latents, 
            mask_tokens=mask_tokens, 
            mask_latents=mask_latents,
        )
        hidden_state_of_latents = hidden_state_of_latents + hidden_state_of_latents_add
        
        hidden_state_of_latents = hidden_state_of_latents + self.ffn_latent(hidden_state_of_latents)
        return hidden_state_of_latents


class Encoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        self.num_latents = cfg.latent.num_latents
        self.latent_dim = cfg.latent.dim
        self.embedding_dim = cfg.embedding.dim
        self.vocab_size = cfg.tokens.vocab_size
        self.num_hidden_layers = cfg.hidden.num_layers
        
        # token embeddings
        self.text_encoder = AutoModel.from_pretrained(
            cfg.model.text_encoder,
            add_pooling_layer=False,
        )
        if cfg.model.text_encoder_freeze_params:
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)

        # latents
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        nn.init.normal_(self.latents, std=cfg.embedding.initializer_range)
        self.latents.requires_grad_(True)
        
        # positional encodings
        self.positional_emb = AbsolutePositionalEmbedding(
            self.embedding_dim,
            cfg.embedding.max_position_embeddings
        )

        # layers
        self.latent_norm = RMSNorm(
            self.latent_dim,
            eps=cfg.normalization.layer_eps
        )
        
        self.latent_layers = nn.ModuleList([
            EncoderTransformerBlock(cfg=deepcopy(cfg)) 
            for _ in range(self.num_hidden_layers)
        ])
        
        self.output_norm = RMSNorm(
            self.latent_dim,
            eps=cfg.normalization.layer_eps
        )

    def forward(self, token_ids, mask_tokens=None, token_embeddings=None):
        if mask_tokens is None:
            mask_tokens = torch.ones_like(token_ids)
            
        mask_latents = torch.ones(
            (token_ids.shape[0], self.num_latents),
            dtype=token_ids.dtype,
            device=token_ids.device
        )

        if token_embeddings is None:
            if self.cfg.model.text_encoder_freeze_params:
                with torch.no_grad():
                    token_embeddings = self.text_encoder(
                        input_ids=token_ids,
                        attention_mask=mask_tokens
                    ).last_hidden_state
            else:
                token_embeddings = self.text_encoder(
                    input_ids=token_ids,
                    attention_mask=mask_tokens
                ).last_hidden_state

        hidden_state_of_embs = token_embeddings + self.positional_emb(token_embeddings)
        
        hidden_state_of_latents = self.latents.view(1, self.num_latents, self.latent_dim).repeat(
            token_embeddings.shape[0], 1, 1
        )
        hidden_state_of_latents = self.latent_norm(hidden_state_of_latents)
        
        for layer in self.latent_layers:
            hidden_state_of_latents = layer(
                hidden_state_of_embs=hidden_state_of_embs, 
                hidden_state_of_latents=hidden_state_of_latents, 
                mask_tokens=mask_tokens, 
                mask_latents=mask_latents,
            )
        
        hidden_state_of_latents = self.output_norm(hidden_state_of_latents)
        return hidden_state_of_latents