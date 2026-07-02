import torch
from torch import nn

from GRFA import GRFA
from SA import SA


class VPRNet(nn.Module):
    def __init__(
            self,
            pretrained_foundation=True,
            foundation_model_path=None,
            token_length=64,
            topk=0.5,
            num_query=64,
            num_heads=16,
            channel_dim=256,
            row_dim=16,
            dropout=0.1,
            num_register=2,
    ):
        super().__init__()
        self.backbone = SA(
            foundation_model_path=foundation_model_path,
            load_pretrained=pretrained_foundation,
            token_length=token_length,
            k=topk,
        )
        self.aggregator = GRFA(
            in_channels=self.backbone.out_channels,
            num_query=num_query,
            num_heads=num_heads,
            channel_dim=channel_dim,
            row_dim=row_dim,
            dropout=dropout,
            num_register=num_register,
        )
        self.features_dim = channel_dim * row_dim

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return torch.nn.functional.normalize(x, p=2, dim=-1)
