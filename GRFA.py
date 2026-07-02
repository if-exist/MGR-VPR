import torch
from torch import nn
import torch.nn.functional as F


class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, num_heads=16, dropout=0., skip=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.skip = skip

    def forward(self, x):
        residual = x
        x = self.dropout(self.attn(x, x, x)[0])
        if self.skip:
            x = residual + x
        x = self.norm(x)
        return x


class CrossAttnLayer(nn.Module):
    def __init__(self, d_model, num_heads=16, dropout=0., skip=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.skip = skip

    def forward(self, q, x):
        residual = q
        q = self.dropout(self.attn(q, x, x)[0])
        if self.skip:
            q = residual + q
        q = self.norm(q)
        return q


def rearrange(x, query_side_len):
    """
    Args:
        x: [B, L, D], L = (query_side_len * reduce_factor)^2
        query_side_len: int, query_num = query_side_len^2
    Returns:
        x: [B * query_side_len^2, reduce_factor^2, D]
    """
    B = x.size(0)
    height = width = int(x.size(1) ** 0.5)
    assert (height // query_side_len) * query_side_len == height
    reduce_factor = (height // query_side_len)
    x = x.view(B, query_side_len, reduce_factor, query_side_len, reduce_factor, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().flatten(0, 2).flatten(1, 2)  # (B*side_len*side_len, reduce_factor*reduce_factor, dim)
    return x


class RegionalSelfAttnLayer(nn.Module):
    def __init__(self, d_model, num_heads=16, dropout=None, skip=True, side_len=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.skip = skip
        self.side_len = side_len

    def forward(self, x):
        """
        Args:
            x: [B, N, D]
        Returns:
            out: [B, N, D]
        """
        residual = x
        B, N, _ = x.size()
        x = rearrange(x, self.side_len)
        x = self.norm(x)
        x = self.dropout(self.attn(x, x, x)[0])
        x = x.reshape(B, N, -1)
        # x = x.view(B, N, -1)
        if self.skip:
            x = residual + x
        return x


class GeMPool(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x, output_size):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, output_size)
        x = x.pow(1. / self.p)
        return x


class RegionPartition(nn.Module):
    def __init__(self, num_query=64, levels=[1]):
        super().__init__()
        self.query_side_len = int(num_query ** 0.5)
        self.levels = levels
        self.pool = GeMPool()

    def forward(self, x):
        """
        Args:
            x: (B, N, D)
        Returns:
            x: (B * num_regions, region_size**2, D)
        """
        B, N, D = x.shape
        height = width = int(N ** 0.5)
        assert (height // self.query_side_len) * self.query_side_len == height
        region_side_len = height // self.query_side_len

        x_regions = []
        for i, level in enumerate(self.levels):
            query_side_len = self.query_side_len // level
            reduce_factor = (height // query_side_len)
            xi = x.view(B, query_side_len, reduce_factor, query_side_len, reduce_factor, -1)
            if level == 1:
                xi = xi.permute(0, 1, 3, 2, 4, 5).contiguous().flatten(1, 2).flatten(2, 3) 
            else:
                xi = xi.permute(0, 1, 3, 5, 2, 4).contiguous().flatten(0, 2) 
                xi = self.pool(xi, region_side_len).flatten(2, 3)  
                xi = xi.view(B, query_side_len * query_side_len, D, -1) 
                xi = xi.permute(0, 1, 3, 2)
            x_regions.append(xi)  
        x = torch.cat(x_regions, 1)  
        return x.flatten(0, 1)   


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FFN(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU, dropout=0., mlp_ratio=4.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_dim, act_layer=act_layer, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.mlp(self.norm(x))
        return x


class GRFA(nn.Module):
    def __init__(self, in_channels=1024, num_query=64, num_heads=16, dropout=0.1,
                 skip=True, channel_dim=256, row_dim=16, levels=[1, 2, 4, 8], mlp_ratio=2.,
                 num_register=4):
        super().__init__()

        proj_channels = in_channels
        self.encoder = RegionalSelfAttnLayer(proj_channels, num_heads, dropout, side_len=int(num_query**0.5))
        self.ffn = FFN(proj_channels, mlp_ratio=mlp_ratio, dropout=dropout)

        self.num_query = 0
        for i, level in enumerate(levels):
            self.num_query += num_query // level**2

        self.num_register = num_register
        self.queries = nn.Parameter(torch.randn(1, self.num_query + num_register, proj_channels))
        nn.init.normal_(self.queries, std=1e-6)

        self.partition = RegionPartition(num_query, levels)

        self.self_attn1 = SelfAttnLayer(proj_channels, num_heads, dropout, skip)
        self.global_cross_attn = CrossAttnLayer(proj_channels, num_heads, dropout, skip)

        self.self_attn2 = SelfAttnLayer(proj_channels, num_heads, dropout, skip)
        self.regional_cross_attn = CrossAttnLayer(proj_channels, num_heads, dropout, skip)

        self.channel_proj = nn.Linear(proj_channels, channel_dim)
        self.row_proj = nn.Linear(self.num_query, row_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        x = self.ffn(x)

        q = self.queries.repeat(B, 1, 1)

        q = self.self_attn1(q)
        q = self.global_cross_attn(q, x)

        q = self.self_attn2(q)
        q = q[:, self.num_register:]

        q = q.view(B, self.num_query, 1, -1).flatten(0, 1)
        x = self.partition(x)
        q = self.regional_cross_attn(q, x)
        q = q.view(B, self.num_query, -1)

        out = self.channel_proj(q)
        out = self.row_proj(out.permute(0, 2, 1)).flatten(1)
        out = F.normalize(out, p=2, dim=-1)
        return out
