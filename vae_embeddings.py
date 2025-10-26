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
        
        ckpt_path = Path(cfg.latent_encoder.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        self.model.load_state_dict(torch.load(ckpt_path)['encoder'])
        
    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)
        return self
        
    def encode(self, texts: list[str]) -> torch.Tensor:
        """for back compitability only"""
        
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.text_encoder.device)
        
        if tokens['input_ids'].shape[1] > self.cfg.latent_encoder.embedding.max_position_embeddings:
            tokens['input_ids'] = tokens['input_ids'][:, :self.cfg.latent_encoder.embedding.max_position_embeddings]
            tokens['attention_mask'] = tokens['attention_mask'][:, :self.cfg.latent_encoder.embedding.max_position_embeddings]

        with torch.no_grad():
            embeddings = self.model(
                tokens["input_ids"],
                mask_tokens=tokens["attention_mask"]
            )
        
        return embeddings[:, 0, :] # take embedding of BOS
    
    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.encode(texts).detach()