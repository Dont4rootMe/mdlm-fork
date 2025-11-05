from omegaconf import DictConfig
from transformers import AutoTokenizer
from architecture_VAE.encoder import Encoder
import torch
from pathlib import Path
from torch import nn

class VAEEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        print('\n\n\n\nusing VAEEncoder: ', cfg, '\n\n\n\n')
        
        self.device = None
        self.cfg = cfg
        self.model = Encoder(cfg.latent_encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.latent_encoder.model.text_encoder)
        
        
        if hasattr(self.tokenizer, 'cls_token') and self.tokenizer.cls_token is not None:
            self.BOS = self.tokenizer.cls_token_id
        else:
            self.BOS = self.tokenizer.encode(self.tokenizer.bos_token, add_special_tokens=False)[0] if self.tokenizer.bos_token else None
            
        if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token is not None:
            self.EOS = self.tokenizer.sep_token_id  
        else:
            self.EOS = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)[0] if self.tokenizer.eos_token else None
        
        
        ckpt_path = Path(cfg.latent_encoder.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Load encoder weights
        self.model.load_state_dict(checkpoint['encoder'])
        
        # Load normalization statistics for BERT embeddings
        if 'encodings_mean' in checkpoint and 'encodings_std' in checkpoint:
            self.encodings_mean = checkpoint['encodings_mean']
            self.encodings_std = checkpoint['encodings_std']
            print(f"✓ Loaded normalization statistics from checkpoint")
        else:
            print("⚠ Warning: No normalization statistics found in checkpoint!")
            print("  BERT embeddings will NOT be normalized (may cause poor performance)")
            self.encodings_mean = None
            self.encodings_std = None
        
    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)
        # Move normalization statistics to device
        if self.encodings_mean is not None:
            self.encodings_mean = self.encodings_mean.to(device)
        if self.encodings_std is not None:
            self.encodings_std = self.encodings_std.to(device)
        return self
    
    def normalize_encodings(self, encodings: torch.Tensor) -> torch.Tensor:
        """Normalize BERT encodings using mean and std statistics.
        
        Args:
            encodings: torch.Tensor of shape (batch_size, sequence_length, hidden_dim)
        
        Returns:
            torch.Tensor: Normalized encodings
        """
        if self.encodings_mean is None or self.encodings_std is None:
            # If statistics not available, return unchanged
            print("⚠ Warning: normalize_encodings called but statistics not available!")
            return encodings
        
        return (encodings - self.encodings_mean) / self.encodings_std
        
    def encode(self, texts: list[str]) -> torch.Tensor:
        """for back compitability only"""
        
        # Токенизация без автоматических специальных токенов
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        # Вручную добавляем BOS в начало и EOS в конец каждой последовательности
        # с учетом max_position_embeddings
        max_pos_emb = self.cfg.latent_encoder.embedding.max_position_embeddings
        input_ids_with_special = []
        attention_masks = []
        
        for input_ids in tokens['input_ids']:
            # Обрезаем текст, чтобы оставить место для BOS и EOS
            # Если текст слишком длинный, обрезаем его до (max_pos_emb - 2)
            max_text_length = max_pos_emb - 2  # -2 для BOS и EOS
            if len(input_ids) > max_text_length:
                input_ids = input_ids[:max_text_length]
            
            # Создаем последовательность: [BOS] + текст + [EOS]
            ids_with_special = torch.cat([
                torch.tensor([self.BOS]),
                input_ids,
                torch.tensor([self.EOS])
            ])
            input_ids_with_special.append(ids_with_special)
            attention_masks.append(torch.ones_like(ids_with_special))
        
        max_length = min(max(len(ids) for ids in input_ids_with_special), max_pos_emb)
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(input_ids_with_special, attention_masks):
            padding_length = max_length - len(ids)
            if padding_length > 0:
                # Добавляем PAD токены в конец
                ids = torch.cat([ids, torch.full((padding_length,), self.tokenizer.pad_token_id)])
                mask = torch.cat([mask, torch.zeros(padding_length)])
            padded_input_ids.append(ids)
            padded_attention_masks.append(mask)
        
        tokens = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks)
        }
        tokens = {k: v.to(self.model.text_encoder.device) for k, v in tokens.items()}

        with torch.no_grad():
            # Get BERT embeddings (same as in encoder_trainer.py)
            bert_hidden_state = self.model.text_encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]
            ).last_hidden_state
            
            # Normalize BERT embeddings (CRITICAL!)
            bert_hidden_state = self.normalize_encodings(bert_hidden_state)
            
            # Pass normalized embeddings to encoder
            embeddings = self.model(
                token_ids=tokens["input_ids"],
                mask_tokens=tokens["attention_mask"],
                token_embeddings=bert_hidden_state  # ← NORMALIZED!
            )
        
        return embeddings[:, 0, :] # take embedding of first latent (BOS position)
    
    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.encode(texts).detach()