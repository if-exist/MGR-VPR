import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from backbone.vision_transformer import vit_large


class SparseAdapter(nn.Module):
    def __init__(
            self,
            embed_dims: int = 1024,
            token_length: int = 64,
            r: int = 4,
            k: float = 0.5,
            scale_init: float = 0.001,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.k = k
        self.learnable_tokens = nn.Parameter(torch.randn(token_length, embed_dims))
        self.scale = nn.Parameter(torch.tensor(scale_init))

        self.mlp_token = nn.Sequential(
            nn.Linear(embed_dims, r),
            nn.GELU(),
            nn.Linear(r, embed_dims),
        )
        self.mlp_delta = nn.Sequential(
            nn.Linear(embed_dims, r),
            nn.GELU(),
            nn.Linear(r, embed_dims),
        )

    def forward(self, feats: Tensor) -> Tensor:
        feats = feats.permute(1, 0, 2)
        cls_token, patch_tokens = torch.tensor_split(feats, [1], dim=0)
        patch_tokens = patch_tokens + self.forward_delta_feat(patch_tokens) * self.scale
        feats = torch.cat([cls_token, patch_tokens], dim=0)
        return feats.permute(1, 0, 2)

    def forward_delta_feat(self, feats: Tensor) -> Tensor:
        tokens = self.learnable_tokens
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        topk = max(1, int(tokens.size(0) * self.k))
        index = torch.topk(attn, k=topk, dim=-1, largest=True)[1]
        mask = torch.zeros_like(attn, device=attn.device, requires_grad=False)
        mask.scatter_(-1, index, 1.0)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float("-inf")))
        attn = F.softmax(attn * (self.embed_dims ** -0.5), dim=-1)

        delta = torch.einsum("nbm,mc->nbc", attn, tokens + self.mlp_token(tokens))
        return self.mlp_delta(delta + feats)


class SA(nn.Module):
    def __init__(
            self,
            foundation_model_path=None,
            load_pretrained=True,
            token_length: int = 64,
            r: int = 4,
            k: float = 0.5,
    ):
        super().__init__()
        self.dino = vit_large(patch_size=14, img_size=518, init_values=1, block_chunks=0)
        if load_pretrained:
            if foundation_model_path is None:
                raise ValueError("Please specify foundation_model_path.")
            state_dict = torch.load(foundation_model_path, map_location="cpu")
            model_dict = self.dino.state_dict()
            model_dict.update(state_dict.items())
            self.dino.load_state_dict(model_dict)

        self.out_channels = self.dino.embed_dim
        for param in self.dino.parameters():
            param.requires_grad = False

        self.adapters = nn.ModuleList([
            SparseAdapter(embed_dims=self.out_channels, token_length=token_length, r=r, k=k)
            for _ in range(len(self.dino.blocks))
        ])

    def forward(self, x):
        x = self.dino.prepare_tokens_with_masks(x)
        y = x.clone()
        for index, block in enumerate(self.dino.blocks):
            x = block(x)
            y = self.adapters[index](y + x)
        return self.dino.norm(y)[:, 1:]
