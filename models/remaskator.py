import torch
import torch.nn as nn
from typing import Optional


class RemaskatorNet(nn.Module):
  """Predicts per-token remask logits given token sequence and optional conditioning.

  Inputs:
    x_tokens: LongTensor of shape (batch, seq_len)
    cond: Optional FloatTensor of shape (batch, cond_dim)

  Outputs:
    logits: FloatTensor of shape (batch, seq_len) — higher means "should remask"
  """

  def __init__(
      self,
      vocab_size: int,
      seq_len: int,
      hidden_size: int = 512,
      num_layers: int = 8,
      num_heads: int = 8,
      cond_dim: Optional[int] = None,
      dropout: float = 0.025,
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.seq_len = seq_len
    self.hidden_size = hidden_size
    self.cond_dim = cond_dim
    self.num_layers = num_layers

    self.token_embedding = nn.Embedding(vocab_size, hidden_size)
    self.pos_embedding = nn.Embedding(seq_len, hidden_size)
    self.dropout = nn.Dropout(dropout)

    # Conditioning projections - one for initial embedding and one for each layer
    if cond_dim is not None:
      self.cond_proj_init = nn.Linear(cond_dim, hidden_size)
      self.cond_proj_layers = nn.ModuleList([
        nn.Linear(cond_dim, hidden_size) for _ in range(num_layers)
      ])
    else:
      self.cond_proj_init = None
      self.cond_proj_layers = None

    # Individual transformer layers (not using TransformerEncoder)
    self.layers = nn.ModuleList([
      nn.TransformerEncoderLayer(
        d_model=hidden_size,
        nhead=num_heads,
        dim_feedforward=hidden_size * 4,
        dropout=dropout,
        activation='gelu',
        batch_first=True,
      ) for _ in range(num_layers)
    ])

    self.head = nn.Linear(hidden_size, 1)

    self._reset_parameters()

  def _reset_parameters(self):
    nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
    nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
    
    if self.cond_proj_init is not None:
      nn.init.xavier_uniform_(self.cond_proj_init.weight)
      nn.init.zeros_(self.cond_proj_init.bias)
      
    if self.cond_proj_layers is not None:
      for cond_proj in self.cond_proj_layers:
        nn.init.xavier_uniform_(cond_proj.weight)
        nn.init.zeros_(cond_proj.bias)
        
    nn.init.xavier_uniform_(self.head.weight)
    nn.init.zeros_(self.head.bias)

  def forward(self, x_tokens: torch.LongTensor, cond: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    batch_size, seq_len = x_tokens.shape
    assert seq_len <= self.seq_len, (seq_len, self.seq_len)

    # Initial embeddings
    token_emb = self.token_embedding(x_tokens)
    positions = torch.arange(seq_len, device=x_tokens.device)
    pos_emb = self.pos_embedding(positions)[None, :, :].expand(batch_size, -1, -1)
    x = token_emb + pos_emb

    # Add initial conditioning
    if self.cond_proj_init is not None and cond is not None:
      cond_emb = self.cond_proj_init(cond)  # (batch, hidden)
      x = x + cond_emb[:, None, :]

    x = self.dropout(x)

    # Pass through each transformer layer with conditioning after each
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if self.cond_proj_layers is not None and cond is not None:
        layer_cond_emb = self.cond_proj_layers[i](cond)  # (batch, hidden)
        x = x + layer_cond_emb[:, None, :]

    logits = self.head(x).squeeze(-1)
    return logits

