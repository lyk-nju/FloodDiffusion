"""Lightweight trajectory encoder for root trajectory conditioning (MotionStream-style concat)."""

import torch
import torch.nn as nn


class TrajEncoder(nn.Module):
    """
    轻量轨迹编码器：升采样 + 降采样，与 4D 潜空间协调，避免喧宾夺主。
    3 -> 64 (升采样) -> 2 (降采样)，输出维度小于潜空间 4，更克制。
    """

    def __init__(self, in_dim=3, hidden_dim=64, out_dim=2):
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
            x: (B, T, 3) 根轨迹
        Returns:
            (B, T, out_dim) 轨迹编码，out_dim=2
        """
        return self.mlp(x)
