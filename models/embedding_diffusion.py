import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
  return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class SinusoidalTimeEmbedding(nn.Module):
  def __init__(self, embedding_dim: int):
    super().__init__()
    self.embedding_dim = embedding_dim

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    # t is (batch,), assumed in [0, T-1]
    half_dim = self.embedding_dim // 2
    freqs = torch.exp(
      torch.linspace(
        math.log(1.0),
        math.log(10000.0),
        steps=half_dim,
        device=t.device,
        dtype=torch.float32,
      ) * (-1.0)
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < self.embedding_dim:
      emb = F.pad(emb, (0, self.embedding_dim - emb.shape[-1]))
    return emb


class EpsilonNet(nn.Module):
  """Simple MLP epsilon network for embedding diffusion."""
  def __init__(self, input_dim: int, time_dim: int, hidden_dim: int, num_layers: int):
    super().__init__()
    self.time_embed = SinusoidalTimeEmbedding(time_dim)
    layers = []
    last = input_dim + time_dim
    for _ in range(max(0, num_layers - 1)):
      layers.append(nn.Linear(last, hidden_dim))
      layers.append(nn.GELU())
      last = hidden_dim
    layers.append(nn.Linear(last, input_dim))
    self.net = nn.Sequential(*layers)

  def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_emb = self.time_embed(t)
    h = torch.cat([x_t, t_emb], dim=-1)
    return self.net(h)


class EmbeddingDiffusionBackbone(nn.Module):
  """Gaussian diffusion (DDPM) over text condition embeddings.

  Trains an epsilon network to denoise embeddings corrupted by a fixed beta schedule.
  Optionally stores simple Gaussian stats for sequence lengths.
  """
  def __init__(
      self,
      cond_dim: int,
      timesteps: int = 1000,
      hidden_dim: int = 512,
      num_layers: int = 3,
      beta_start: float = 1e-4,
      beta_end: float = 0.02,
      t_sampling_exponent: float = 2.0,
      length_mean: Optional[float] = None,
      length_std: Optional[float] = None,
      max_seq_len: Optional[int] = None,
  ) -> None:
    super().__init__()
    self.cond_dim = int(cond_dim)
    self.timesteps = int(timesteps)
    self.net = EpsilonNet(input_dim=self.cond_dim, time_dim=64, hidden_dim=hidden_dim, num_layers=num_layers)
    self.t_sampling_exponent = float(t_sampling_exponent)

    betas = _default_beta_schedule(self.timesteps, beta_start=beta_start, beta_end=beta_end)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    self.register_buffer('betas', betas)
    self.register_buffer('alphas', alphas)
    self.register_buffer('alphas_cumprod', alphas_cumprod)
    self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
    self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
    self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
    self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
    self.register_buffer('posterior_variance', posterior_variance)
    self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
    self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
    self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    # Optional length stats
    self.length_mean = None if length_mean is None else float(length_mean)
    self.length_std = None if length_std is None else float(length_std)
    self.max_seq_len = None if max_seq_len is None else int(max_seq_len)

  def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    if noise is None:
      noise = torch.randn_like(x0)
    sqrt_alpha_hat = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
    sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
    return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise

  def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps_theta = self.net(x_t, t)
    alpha_hat_t = self.alphas_cumprod[t].unsqueeze(-1)
    sqrt_recip_alpha = self.sqrt_recip_alphas[t].unsqueeze(-1)
    sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
    # Predict x0
    x0_pred = (x_t - sqrt_one_minus_alpha_hat * eps_theta) / torch.sqrt(alpha_hat_t)
    # Compute posterior mean
    mean = self.posterior_mean_coef1[t].unsqueeze(-1) * x0_pred + self.posterior_mean_coef2[t].unsqueeze(-1) * x_t
    var = self.posterior_variance[t].unsqueeze(-1)
    log_var = self.posterior_log_variance_clipped[t].unsqueeze(-1)
    return mean, var, log_var

  @torch.no_grad()
  def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    mean, _, log_var = self.p_mean_variance(x_t, t)
    if (t == 0).all():
      return mean
    noise = torch.randn_like(x_t)
    return mean + torch.exp(0.5 * log_var) * noise

  @torch.no_grad()
  def sample_embeddings(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None:
      device = next(self.parameters()).device
    x = torch.randn(num_samples, self.cond_dim, device=device)
    for i in reversed(range(self.timesteps)):
      t = torch.full((num_samples,), i, device=device, dtype=torch.long)
      x = self.p_sample(x, t)
    return x

  @torch.no_grad()
  def sample(self, num_samples: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    embs = self.sample_embeddings(num_samples=num_samples, device=device)
    lengths = None
    if self.length_mean is not None and self.length_std is not None:
      device = embs.device
      l = torch.normal(mean=torch.tensor(self.length_mean, dtype=torch.float32, device=device), std=torch.tensor(self.length_std, dtype=torch.float32, device=device), size=(num_samples,))
      lengths = torch.round(l).to(torch.int64)
      if self.max_seq_len is not None:
        lengths = torch.clamp(lengths, min=1, max=self.max_seq_len)
      else:
        lengths = torch.clamp(lengths, min=1)
    return embs, lengths

  def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
    batch_size = x0.shape[0]
    device = x0.device
    # Bias timestep sampling toward higher-noise steps when exponent > 1.0
    if self.t_sampling_exponent <= 0:
      self.t_sampling_exponent = 1.0
    u = torch.rand(batch_size, device=device)
    x = torch.pow(u, 1.0 / self.t_sampling_exponent)
    t = torch.clamp((x * (self.timesteps - 1)).floor().to(torch.long), 0, self.timesteps - 1)
    noise = torch.randn_like(x0)
    x_t = self.q_sample(x0, t, noise)
    eps_pred = self.net(x_t, t)
    return F.mse_loss(eps_pred, noise)

  def configure_optim(self, lr: float = 1e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

  def extra_state(self) -> dict:
    return {
      'cond_dim': self.cond_dim,
      'timesteps': self.timesteps,
      'length_mean': self.length_mean,
      'length_std': self.length_std,
      'max_seq_len': self.max_seq_len,
    }

  def set_length_stats(self, length_mean: float, length_std: float, max_seq_len: Optional[int]) -> None:
    self.length_mean = float(length_mean)
    self.length_std = float(length_std)
    self.max_seq_len = None if max_seq_len is None else int(max_seq_len)

  # No save/load here; use the Lightning module to manage checkpoints

