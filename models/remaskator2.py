import torch
import torch.nn as nn
from typing import Optional
import omegaconf
from models.dit import DIT, LayerNorm, modulate_fused


class SingleLogitFinalLayer(nn.Module):
  def __init__(self, hidden_size: int, cond_dim: int, cond_dim_embedding: int):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, 1)

    # Initialize final layer to zeros for stable start
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    # Sigma conditioning (timestep)
    self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

    # Condition embedding conditioning
    self.cond_embedding_modulation = nn.Linear(
      cond_dim_embedding,
      2 * hidden_size,
      bias=True)

  def forward(self, x: torch.Tensor, c: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # Process sigma conditioning
    shift_sigma, scale_sigma = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)

    # Process condition embedding conditioning
    shift_cond, scale_cond = self.cond_embedding_modulation(cond)[:, None].chunk(2, dim=2)

    # Combine conditioning parameters
    shift = shift_sigma + shift_cond
    scale = scale_sigma + scale_cond

    x_mod = modulate_fused(self.norm_final(x), shift, scale)
    x_out = self.linear(x_mod)
    return x_out


class Remaskator2Net(nn.Module):
  """Remaskator implemented by reusing DIT blocks; single-logit per token output.

  Inputs:
    x_tokens: LongTensor of shape (batch, seq_len)
    cond: Optional FloatTensor of shape (batch, cond_dim)

  Outputs:
    logits: FloatTensor of shape (batch, seq_len)
  """

  def __init__(
      self,
      vocab_size: int,
      config: omegaconf.DictConfig,
      cond_dim: int,
      use_residual_modulation: bool = False,
      use_weighted_sum: bool = True,
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.seq_len = config.model.length

    # Build base DIT
    self.dit = DIT(config, vocab_size=vocab_size, cond_dim=cond_dim, use_residual_modulation=use_residual_modulation, use_weighted_sum=use_weighted_sum)

    # Replace the final projection with a single-logit head
    self.output_layer = SingleLogitFinalLayer(
      hidden_size=config.model.hidden_size,
      cond_dim=config.model.cond_dim,
      cond_dim_embedding=config.model.cond_dim_embedding,
    )

  def change_final_layer(self):
    self.dit.output_layer = self.output_layer

  def forward(self, x_tokens: torch.LongTensor, cond: Optional[torch.FloatTensor] = None, curr_embed: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    batch_size = x_tokens.shape[0]
    sigma = torch.zeros(batch_size, device=x_tokens.device, dtype=torch.float32)

    # DIT returns (batch, seq_len, 1); squeeze to (batch, seq_len)
    logits = self.dit(x_tokens, sigma, cond, curr_embed)
    logits = logits.squeeze(-1)
    return logits

