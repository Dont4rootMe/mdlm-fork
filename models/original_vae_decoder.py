import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from omegaconf import DictConfig
from transformers import AutoModel
import os

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


def get_embedding():
    # Embedding matrix - get BERT [MASK] token embedding
    from transformers import AutoModel, AutoTokenizer
    
    # Load BERT model and tokenizer
    bert_model_name = "bert-base-cased"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModel.from_pretrained(bert_model_name)
    
    # Get [MASK] token ID and embedding
    mask_token_id = bert_tokenizer.mask_token_id
    mask_embedding = bert_model.embeddings.word_embeddings.weight[mask_token_id].detach().clone()
    
    return mask_embedding


def load_compatible_weights(model, checkpoint_path):
    """
    Загружает веса из чекпоинта, сопоставляя названия слоев и проверяя размерности.
    
    Args:
        model: PyTorch модель для инициализации
        checkpoint_path: путь к чекпоинту
    
    Returns:
        dict: статистика загрузки (loaded, skipped, missing)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return {"loaded": 0, "skipped": 0, "missing": 0}
    
    # Загружаем state_dict из чекпоинта
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrained_state_dict = checkpoint.get('decoder', {})
    
    if not pretrained_state_dict:
        print("No 'decoder' key found in checkpoint")
        return {"loaded": 0, "skipped": 0, "missing": 0}
    
    # Получаем state_dict текущей модели
    model_state_dict = model.state_dict()
    
    # Статистика
    loaded_count = 0
    skipped_count = 0
    size_mismatch_count = 0
    
    print("\n" + "="*80)
    print("Loading pretrained weights for decoder...")
    print("="*80)
    
    # Проходим по всем параметрам модели
    for model_key, model_param in model_state_dict.items():
        model_shape = model_param.shape
        
        # Пытаемся найти соответствующий ключ в чекпоинте
        # Прямое совпадение
        if model_key in pretrained_state_dict:
            pretrained_param = pretrained_state_dict[model_key]
            pretrained_shape = pretrained_param.shape
            
            # Проверяем размерности
            if model_shape == pretrained_shape:
                model_state_dict[model_key].copy_(pretrained_param)
                loaded_count += 1
                print(f"✓ Loaded: {model_key:60s} {str(model_shape):30s}")
            else:
                size_mismatch_count += 1
                print(f"✗ Size mismatch: {model_key:60s} model{model_shape} vs ckpt{pretrained_shape}")
        else:
            skipped_count += 1
            print(f"⊘ Not found in checkpoint: {model_key:60s} {str(model_shape):30s}")
    
    # Загружаем обновленный state_dict
    model.load_state_dict(model_state_dict)
    
    print("="*80)
    print(f"Summary:")
    print(f"  Loaded: {loaded_count} parameters")
    print(f"  Skipped (not found): {skipped_count} parameters")
    print(f"  Skipped (size mismatch): {size_mismatch_count} parameters")
    print(f"  Total model parameters: {len(model_state_dict)}")
    print("="*80 + "\n")
    
    return {
        "loaded": loaded_count,
        "skipped": skipped_count,
        "size_mismatch": size_mismatch_count,
        "total": len(model_state_dict)
    }


class Decoder(nn.Module):
    def __init__(self, 
                 config: DictConfig, 
                 vocab_size: int, 
                 cond_dim: int = None, 
                 use_residual_modulation: bool = False, 
                 use_weighted_sum: bool = True,
                 mask_token_id = None
    ):
        super().__init__()
        print('\n\n\n\nusing Original VAE Decoder\n\n\n\n')
        
        self.cfg = config
        
        self.max_position_embeddings = config.vae_encoder.latent_encoder.embedding.max_position_embeddings
        self.embedding_dim = config.vae_encoder.latent_encoder.embedding.dim
        self.vocab_size = vocab_size
        self.num_hidden_layers = config.vae_encoder.latent_encoder.hidden.num_layers
        self.num_latents = config.vae_encoder.latent_encoder.latent.num_latents
        self.latent_dim = cond_dim
        
        self.mask_token_id = mask_token_id
        
        assert self.mask_token_id is not None
        # self.mask_token_id = config.vae_encoder.latent_encoder.tokens.mask_token_id
        # self.max_seq_len = config.vae_encoder.latent_encoder.embedding.max_position_embeddings
        
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
        
        mask_embedding = get_embedding()
        
        # Verify embedding dimension compatibility
        bert_embedding_dim = mask_embedding.shape[0]
        if bert_embedding_dim != self.embedding_dim:
            raise ValueError(
                f"BERT embedding dimension ({bert_embedding_dim}) does not match "
                f"expected embedding dimension ({self.embedding_dim}). "
                f"Please adjust config.vae_encoder.latent_encoder.embedding.dim to {bert_embedding_dim}"
            )
        
        # Create embedding parameter - single [MASK] embedding for all positions
        self.embedding = nn.Parameter(
            mask_embedding.unsqueeze(0).unsqueeze(0),  # Shape: [1, 1, embedding_dim]
            requires_grad=True
        )
        
        # layers
        self.embedding_ffn = FeedForwardNetwork(
            embedding_dim=self.embedding_dim,
            mult=config.vae_encoder.latent_encoder.hidden.ff_mult,
            dropout_p=config.vae_encoder.latent_encoder.hidden.dropout
        )
        self.layers = nn.ModuleList([
            DecoderTransformerBlock(cfg=deepcopy(config.vae_encoder.latent_encoder)) 
            for _ in range(self.num_hidden_layers)
        ])
        
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        self.scale_embedding = ScaleMask()
        
        # Загружаем предобученные веса, если чекпоинт существует
        if hasattr(config.vae_encoder.latent_encoder, 'checkpoint') and config.vae_encoder.latent_encoder.checkpoint:
            checkpoint_path = config.vae_encoder.latent_encoder.checkpoint
            load_compatible_weights(self, checkpoint_path)
        else:
            raise ValueError(f"Checkpoint not found: {config.vae_encoder.latent_encoder.checkpoint}")

    # def forward(self, encoder_latents, masked_input_ids=None, return_last_hidden_state=False):
    def forward(self, indices, sigma=None, condition=None, curr_embed=None, return_last_hidden_state=False):
        """Forward pass for VAE decoder.
        
        Args:
            indices: Input token indices (will be masked internally)
            sigma: Noise level (unused, kept for API compatibility)
            condition: Encoder latents - the only input actually used
            curr_embed: Current embeddings (unused, kept for API compatibility)
            return_last_hidden_state: Whether to return hidden states for MSE loss
        """
        
        # Note: sigma and curr_embed are ignored by this decoder
        # Only condition (encoder latents) is used
        encoder_latents = condition
        
        # Validate that condition is provided
        assert encoder_latents is not None, "Condition (encoder latents) must be provided for VAE decoder"
        
        if len(encoder_latents.shape) == 2:
            encoder_latents = encoder_latents.unsqueeze(1) # [B, 1, cond_dim]
        
        batch_size = encoder_latents.shape[0]
        
        # Create masks if not provided
        encoder_latents_mask = torch.ones(
            (encoder_latents.shape[0], encoder_latents.shape[1]),
            dtype=encoder_latents.dtype,
            device=encoder_latents.device
        )

        # mask out all the indices
        masked_input_ids = (indices * 0.0 + self.mask_token_id).long()

        mask_of_mask_tokens = (masked_input_ids == self.mask_token_id).float()

        tokens_mask = torch.ones(
            (batch_size, masked_input_ids.shape[1]),
            dtype=encoder_latents.dtype,
            device=encoder_latents.device
        )

        # Use the single [MASK] embedding for all positions
        seq_len = masked_input_ids.shape[1]
        # Expand [MASK] embedding to match sequence length: [1, 1, dim] -> [batch_size, seq_len, dim]
        embedding = self.embedding.expand(batch_size, seq_len, -1)
        
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
        