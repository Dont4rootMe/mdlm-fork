import torch
import torch.nn as nn
import math
import inspect
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Set, Callable
import torch.nn.functional as F

from .llama_blocks import LlamaBlock
from .blocks import RMSNorm

TransformerBlock = LlamaBlock


class TimeLayerProjection(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.x_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm(self.x_proj(x))
        emb = self.linear(self.silu(emb_t))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        x = x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_self_cond = config.diffusion.use_self_cond
        self.num_hidden_layers = config.architecture.unconditional_encoder.num_hidden_layers
        self.hidden_size = config.architecture.unconditional_encoder.hidden_size
        self.layer_norm_eps = config.architecture.unconditional_encoder.layer_norm_eps

        self.input_blocks = torch.nn.ModuleList(
            [TransformerBlock(config.architecture.unconditional_encoder, is_cross_attention=config.diffusion.is_conditional) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.output_blocks = torch.nn.ModuleList(
            [TransformerBlock(config.architecture.unconditional_encoder, is_cross_attention=config.diffusion.is_conditional) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.time_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.hidden_size, self.hidden_size)
                ) for _ in range(0, self.num_hidden_layers)
            ]
        )
        if self.use_self_cond:
            self.self_cond_layers = torch.nn.ModuleList(
                [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(0, self.num_hidden_layers)]
            )
        self.output_norm = RMSNorm(self.hidden_size, eps=self.layer_norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            emb_t=None,
            cond=None,
            cond_mask=None,
            x_0_self_cond=None,
    ):
        x_input_list = []

        for i, block in enumerate(self.input_blocks):
            x_input_list.append(x)

            x = x + self.time_layers[i](emb_t[:, None])

            if self.use_self_cond:
                x = x + self.self_cond_layers[i](x_0_self_cond)

            x = block(
                hidden_states=x,
                attention_mask=attention_mask,
                encoder_hidden_states=cond,
                encoder_attention_mask=cond_mask
            )

        for i, block in enumerate(self.output_blocks):
            ind = i + self.num_hidden_layers // 2

            x = x + x_input_list.pop() + self.time_layers[ind](emb_t[:, None])

            if self.use_self_cond:
                x = x + self.self_cond_layers[ind](x_0_self_cond)

            x = block(
                hidden_states=x,
                attention_mask=attention_mask,
                encoder_hidden_states=cond,
                encoder_attention_mask=cond_mask
            )
        x = self.output_norm(x)

        return x


class ConditionalEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_hidden_layers = config.conditional_encoder.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps
        arch_config = deepcopy(config.diffusion_encoder)
        arch_config.is_conditional = False
        self.blocks = torch.nn.ModuleList([TransformerBlock(arch_config) for _ in range(0, self.num_hidden_layers)])
        self.output_norm = RMSNorm(self.hidden_size, eps=self.layer_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
    ):
        for _, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
        hidden_states = self.output_norm(hidden_states)
        return hidden_states


def timestep_embedding(timesteps, dim, max_period=10):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ScoreEstimator(nn.Module):
    def __init__(self, config):
        super(ScoreEstimator, self).__init__()

        self.config = config
        self.hidden_size = config.architecture.unconditional_encoder.hidden_size
        self.time_max_period = config.architecture.time_embedding.max_period
        self.is_conditional = config.diffusion.is_conditional


        self.time_emb = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        )

        self.encoder = TransformerEncoder(config)

        if self.is_conditional:
            self.conditional_encoder = ConditionalEncoder(config)

        if self.is_conditional:
            self._max_position_embeddings = self.config.architecture.conditional_encoder.max_position_embeddings
            self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
            self.position_embeddings = torch.nn.Embedding(self._max_position_embeddings, self.hidden_size)

    def get_extended_attention_mask(self, attention_mask, dtype):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(
            self,
            x_t: torch.Tensor,
            time_t: torch.Tensor,
            cond=None,
            attention_mask=None,
            cond_mask=None,
            x_0_self_cond=None,
    ):
        hidden_t = self.time_emb(timestep_embedding(time_t, self.hidden_size, self.time_max_period))

        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(
                attention_mask=attention_mask,
                dtype=x_t.dtype
            )
            
        if cond_mask is not None:
            cond_mask = self.get_extended_attention_mask(
                attention_mask=cond_mask,
                dtype=x_t.dtype
            )
        
        if self.is_conditional:
            cond = self.conditional_encoder(
                hidden_states=cond,
                attention_mask=cond_mask
            )

            seq_length = cond.size(1)
            position_ids = self.position_ids[:, : seq_length]
            emb_pos = self.position_embeddings(position_ids)
            cond = cond + emb_pos

        output = self.encoder(
            x=x_t,
            attention_mask=attention_mask,
            emb_t=hidden_t,
            cond=cond,
            cond_mask=cond_mask,
            x_0_self_cond=x_0_self_cond,
        )
        return output