"""Lightweight trajectory encoder for FlexTraj-style traj tokens (feeds WanModel.traj_in_proj)."""

import torch
import torch.nn as nn


class LocalTrajEncoder(nn.Module):
    """Token 内 4 帧 → 1 个 token 级 4D 向量（与 FloodNet 一致，1D Conv + mean）。"""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(-2) != 4 or x.size(-1) != 4:
            raise ValueError(f"expected (B,T,4,4), got {tuple(x.shape)}")
        b, t, l, c = x.shape
        y = x.reshape(b * t, l, c).transpose(1, 2).contiguous()
        y = self.net(y)
        y = y.mean(dim=-1)
        return y.reshape(b, t, 4)


class TrajEncoder(nn.Module):
    """
    轻量轨迹编码器：将 token 对齐的平面+朝向特征映射到 traj_enc_dim，再经 WanModel 投到主干 dim。
    默认 in_dim=4 对应 (x, z, cos ψ, sin ψ)；若仅有 (x,y,z) 可由调用方补全为 4D。
    """

    def __init__(self, in_dim=4, hidden_dim=64, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x):
        """
        Args:
            x: (B, T, in_dim) 轨迹特征
        Returns:
            (B, T, out_dim)
        """
        return self.mlp(x)
