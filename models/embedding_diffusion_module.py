import os
from typing import Optional, List

import lightning as L
import torch
import torch.nn.functional as F

from models.embedding_diffusion import EmbeddingDiffusionBackbone
import embeddings as emb

class EmbeddingDiffusionModule(L.LightningModule):
  """Lightning module wrapping EmbeddingDiffusionBackbone.

  Responsibilities:
  - Manage optimizer and training loop over precomputed embeddings
  - Provide sampling API for inference
  - Store optional length stats used for sampling sequence lengths
  """
  def __init__(
      self,
      cond_dim: int,
      timesteps: int = 1000,
      hidden_dim: int = 512,
      num_layers: int = 3,
      lr: float = 1e-3,
      weight_decay: float = 0.0,
      t_sampling_exponent: float = 2.0,
      length_mean: Optional[float] = None,
      length_std: Optional[float] = None,
      max_seq_len: Optional[int] = None,
      tokenizer=None,
      text_embedder: Optional[emb.TextEmbedder] = None,
      fid_sample_size: int = 2048,
  ) -> None:
    super().__init__()
    # Avoid saving tokenizer and embedder in hparams
    self.save_hyperparameters(ignore=['tokenizer', 'text_embedder'])
    self.net = EmbeddingDiffusionBackbone(
      cond_dim=cond_dim,
      timesteps=timesteps,
      hidden_dim=hidden_dim,
      num_layers=num_layers,
      t_sampling_exponent=t_sampling_exponent,
      length_mean=length_mean,
      length_std=length_std,
      max_seq_len=max_seq_len,
    )
    self.lr = lr
    self.weight_decay = weight_decay

    # Buffer to hold a view of the training set embeddings provided externally
    self.embeddings_dataset: Optional[torch.Tensor] = None

    # Text pipeline
    self.tokenizer = tokenizer
    self.text_embedder = text_embedder
    self.pad_token_id = None if tokenizer is None else getattr(tokenizer, 'pad_token_id', 0)

    # Validation sampling buffer for FID-like metric
    self.fid_sample_size = int(fid_sample_size)
    self._val_emb_buffer: Optional[torch.Tensor] = None

  def set_embeddings_dataset(self, embeddings: torch.Tensor) -> None:
    self.embeddings_dataset = embeddings

  def training_step(self, batch, batch_idx: int):
    # Compute embeddings on-the-fly from tokens
    x0 = self._batch_to_embeddings(batch)
    loss = self.net.training_loss(x0)
    self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
    return loss

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

  @torch.no_grad()
  def sample(self, num_samples: int, device: Optional[torch.device] = None):
    return self.net.sample(num_samples=num_samples, device=device)

  @torch.no_grad()
  def _batch_to_embeddings(self, batch) -> torch.Tensor:
    if self.text_embedder is None or self.tokenizer is None:
      raise RuntimeError('Text embedder and tokenizer must be provided to the module for on-the-fly embedding computation.')
    input_ids = batch['input_ids'] if isinstance(batch, dict) else batch
    # Decode to texts
    texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    embs = self.text_embedder(texts)
    if not isinstance(embs, torch.Tensor):
      embs = torch.tensor(embs)
    return embs.to(self.device, dtype=torch.float32)

  def on_validation_epoch_start(self) -> None:
    self._val_emb_buffer = None

  def validation_step(self, batch, batch_idx: int):
    x0 = self._batch_to_embeddings(batch)
    loss = self.net.training_loss(x0)
    self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    # Accumulate embeddings for FID-like metric
    with torch.no_grad():
      # Detach to CPU to reduce GPU memory
      x_cpu = x0.detach().to('cpu')
      if self._val_emb_buffer is None:
        self._val_emb_buffer = x_cpu[:self.fid_sample_size]
      else:
        remaining = max(0, self.fid_sample_size - self._val_emb_buffer.shape[0])
        if remaining > 0:
          to_take = min(remaining, x_cpu.shape[0])
          self._val_emb_buffer = torch.cat([self._val_emb_buffer, x_cpu[:to_take]], dim=0)
    return loss

  def on_validation_epoch_end(self) -> None:
    # Compute Fréchet distance between real val embeddings and samples from the model
    if self._val_emb_buffer is None or self._val_emb_buffer.shape[0] < 2:
      return
    with torch.no_grad():
      real = self._val_emb_buffer.to(self.device, dtype=torch.float32)
      m1 = real.mean(dim=0)
      x = real - m1
      C1 = (x.t() @ x) / max(1, (x.shape[0] - 1))
      n = real.shape[0]
      samples = self.net.sample_embeddings(num_samples=n, device=self.device).to(torch.float32)
      m2 = samples.mean(dim=0)
      y = samples - m2
      C2 = (y.t() @ y) / max(1, (y.shape[0] - 1))
      # Fréchet distance
      e_vals = torch.linalg.eigvals(C1 @ C2)
      # Numerical stability: real part and clamp
      e_vals = torch.real(e_vals)
      e_vals = torch.clamp(e_vals, min=0.0)
      sqrt_trace = torch.sum(torch.sqrt(e_vals))
      diff = (m1 - m2)
      fd = diff.dot(diff) + torch.trace(C1 + C2) - 2.0 * sqrt_trace
      self.log('val/frechet', fd, prog_bar=True, on_epoch=True, sync_dist=True)

      # Normalized 0..1 distribution approximation score (higher is better)
      # Use real distribution dispersion to scale FD and squash via exp.
      denom = (torch.trace(C1) + m1.dot(m1)).clamp_min(1e-8)
      dist_score = torch.exp(-fd / denom)
      self.log('val/dist_score', dist_score, prog_bar=True, on_epoch=True, sync_dist=True)
      
      # t = os.listdir('/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/mdlm-fork')
      # num = len([m for m in t if m.startswith('diffusion_state_dict')])
      # torch.save(self.state_dict(), f'/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/mdlm-fork/diffusion_state_dict_{num}.pt')

  def on_fit_start(self) -> None:
    # Ensure text embedder is on the same device as the module
    try:
      if self.text_embedder is not None and hasattr(self.text_embedder, 'model'):
        self.text_embedder.model = self.text_embedder.model.to(self.device)
        self.text_embedder.device = self.device
    except Exception:
      pass

