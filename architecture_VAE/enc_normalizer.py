import os

import torch
from torch import nn, FloatTensor


class EncNormalizer(nn.Module):
    def __init__(self, enc_path: str):
        super().__init__()

        if not os.path.exists(enc_path):
            raise ValueError("enc_path is None")

        state_dict = torch.load(enc_path, map_location='cuda')
        self.enc_mean = nn.Parameter(
            state_dict["enc_mean"][None, None, :],
            requires_grad=False
        )
        self.enc_std = nn.Parameter(
            state_dict["enc_std"][None, None, :],
            requires_grad=False
        )

    def forward(self, *args, **kwargs):
        return nn.Identity()(*args, **kwargs)

    def normalize(self, encoding: FloatTensor) -> FloatTensor:
        return (encoding - self.enc_mean) / self.enc_std

    def denormalize(self, pred_x_0: FloatTensor) -> FloatTensor:
        return pred_x_0 * self.enc_std + self.enc_mean