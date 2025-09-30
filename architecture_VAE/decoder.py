import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from omegaconf import DictConfig
from transformers import AutoModel

from .blocks import AbsolutePositionalEmbedding, FeedForwardNetwork
from .latent_attention import LatentAttention


class ScaleMask(nn.Module):
    def __init__(self,):
        super().__init__()
        self.scale_mlp = nn.Linear(1, 1, bias=True)
        
    def forward(self, x, mask):
        scale = self.scale_mlp(mask.unsqueeze(-1))
        x = x * (scale + 1)
        return x


class DecoderTransformerBlock(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = deepcopy(cfg)
        self.cfg.embedding.dim, self.cfg.latent.dim = cfg.latent.dim, cfg.embedding.dim

        self.latent_attn = LatentAttention(
            cfg=self.cfg,
            latents_first=False
        )
        self.attn_scale = ScaleMask()
        
        self.ffn_latent = FeedForwardNetwork(
            embedding_dim=self.cfg.latent.dim,
            mult=self.cfg.hidden.ff_mult,
            dropout_p=self.cfg.hidden.dropout
        )
        self.ffn_scale = ScaleMask()
        
    def forward(self, hidden_state_of_embs, hidden_state_of_latents, mask_tokens, mask_latents, mask_of_mask_tokens):
        hidden_state_of_embs_add = self.latent_attn(
            hidden_state_of_embs=hidden_state_of_latents, 
            hidden_state_of_latents=hidden_state_of_embs, 
            mask_tokens=mask_latents, 
            mask_latents=mask_tokens,
        )
        hidden_state_of_embs = hidden_state_of_embs + self.attn_scale(hidden_state_of_embs_add, mask_of_mask_tokens)
        
        hidden_state_of_embs = hidden_state_of_embs + self.ffn_scale(self.ffn_latent(hidden_state_of_embs), mask_of_mask_tokens)
        return hidden_state_of_embs


def get_embedding_matrix(cfg: DictConfig):
    if cfg.model.text_encoder == "bert-base-cased":
        embedding_matrix = AutoModel.from_pretrained(
            cfg.model.text_encoder,
            add_pooling_layer=False,
        ).embeddings.word_embeddings
        return embedding_matrix
    else:
        raise ValueError(f"Unsupported text encoder: {cfg.model.text_encoder}")


class Decoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        self.max_position_embeddings = cfg.embedding.max_position_embeddings
        self.embedding_dim = cfg.embedding.dim
        self.vocab_size = cfg.tokens.vocab_size
        self.num_hidden_layers = cfg.hidden.num_layers
        self.num_latents = cfg.latent.num_latents
        self.latent_dim = cfg.latent.dim
        self.mask_token_id = cfg.tokens.mask_token_id
        self.max_seq_len = cfg.embedding.max_position_embeddings
        
        # positional encodings for encoder latents
        self.positional_emb = AbsolutePositionalEmbedding(
            self.embedding_dim,
            self.max_position_embeddings
        )
        if self.embedding_dim != self.latent_dim:
            self.positional_latent = AbsolutePositionalEmbedding(
                self.latent_dim,
                self.num_latents
            )
        
        # Embedding matrix
        self.embedding = get_embedding_matrix(cfg)
        if cfg.model.text_encoder_freeze_params:
            self.embedding.requires_grad_(False)
        
        # layers
        self.embedding_ffn = FeedForwardNetwork(
            embedding_dim=self.embedding_dim,
            mult=cfg.hidden.ff_mult,
            dropout_p=cfg.hidden.dropout
        )
        self.layers = nn.ModuleList([
            DecoderTransformerBlock(cfg=deepcopy(cfg)) 
            for _ in range(self.num_hidden_layers)
        ])
        
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        self.scale_embedding = ScaleMask()

    def forward(self, encoder_latents, masked_input_ids=None, return_last_hidden_state=False):
        batch_size = encoder_latents.shape[0]
        
        # Create masks if not provided
        encoder_latents_mask = torch.ones(
            (encoder_latents.shape[0], encoder_latents.shape[1]),
            dtype=encoder_latents.dtype,
            device=encoder_latents.device
        )

        if masked_input_ids is None:
            batch_size = encoder_latents.shape[0]
            masked_input_ids = torch.ones((batch_size, self.max_seq_len), device=encoder_latents.device, dtype=torch.long) * self.mask_token_id

        tokens_mask = torch.ones(
            (batch_size, masked_input_ids.shape[1]),
            dtype=encoder_latents.dtype,
            device=encoder_latents.device
        )

        mask_of_mask_tokens = (masked_input_ids == self.mask_token_id).float()
        
        if self.cfg.model.text_encoder_freeze_params:
            with torch.no_grad():
                embedding = self.embedding(masked_input_ids)
        else:
            embedding = self.embedding(masked_input_ids)

        hidden_state_of_decoder = self.scale_embedding(self.embedding_ffn(embedding), mask_of_mask_tokens) + self.positional_emb(embedding)
        if self.embedding_dim != self.latent_dim:
            hidden_state_of_encoder_latents = encoder_latents + self.positional_latent(encoder_latents)
        else:
            hidden_state_of_encoder_latents = encoder_latents + self.positional_emb(encoder_latents)

        for layer in self.layers:
            hidden_state_of_decoder = layer(
                hidden_state_of_embs=hidden_state_of_decoder,
                hidden_state_of_latents=hidden_state_of_encoder_latents,
                mask_tokens=tokens_mask,
                mask_latents=encoder_latents_mask,
                mask_of_mask_tokens=mask_of_mask_tokens,
            )
            
        logits = self.lm_head(hidden_state_of_decoder)
        if return_last_hidden_state:
            return logits, hidden_state_of_decoder
        else:
            return logits
        