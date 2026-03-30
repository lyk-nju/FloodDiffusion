"""轨迹 batch：路径朝向 [x,z,cos,sin] 与 DiffForcing → WanModel 的轨迹编码输入。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_PATH_HEADING_EPS = 1e-8


def path_heading_features_from_root_xyz(
    traj_xyz: np.ndarray, eps: float = _PATH_HEADING_EPS
) -> np.ndarray:
    """
    根轨迹 (T,3) 的 x,y,z → (T,4)：[x, z, cos ψ, sin ψ]。

    ψ 为 **xz 路径朝向**（位移差分单位化），与 `xyz_traj_to_features_4d` 逻辑一致；
    用于数据集 `traj_features`，与仅能提供路径的推理条件对齐。
    """
    traj_xyz = np.asarray(traj_xyz, dtype=np.float64)
    t_len = traj_xyz.shape[0]
    x = traj_xyz[:, 0:1]
    z = traj_xyz[:, 2:3]
    if t_len == 1:
        cos = np.ones((1, 1), dtype=np.float64)
        sin_p = np.zeros((1, 1), dtype=np.float64)
        return np.concatenate([x, z, cos, sin_p], axis=-1).astype(np.float32)

    dx = np.zeros((t_len, 1), dtype=np.float64)
    dz = np.zeros((t_len, 1), dtype=np.float64)
    dx[0:1] = x[1:2] - x[0:1]
    dz[0:1] = z[1:2] - z[0:1]
    dx[1:] = x[1:] - x[:-1]
    dz[1:] = z[1:] - z[:-1]

    sq = dx * dx + dz * dz
    short = sq < eps * eps
    norm = np.sqrt(np.maximum(sq, eps * eps))
    cos_yaw = np.where(short, 1.0, dx / norm)
    sin_yaw = np.where(short, 0.0, dz / norm)
    return np.concatenate([x, z, cos_yaw, sin_yaw], axis=-1).astype(np.float32)


def xyz_traj_to_features_4d(
    traj_xyz: torch.Tensor, eps: float = _PATH_HEADING_EPS
) -> torch.Tensor:
    """
    (B,T,3) 列 x,y,z → (B,T,4)：[x, z, cos ψ, sin ψ]，ψ 为 **xz 路径朝向**（差分单位化）。

    与 `path_heading_features_from_root_xyz` 及数据集 `traj_features` 语义一致。
    """
    x_coord = traj_xyz[..., 0:1]
    z_coord = traj_xyz[..., 2:3]
    _, t, _ = x_coord.shape
    if t == 1:
        cos_yaw = torch.ones_like(x_coord)
        sin_yaw = torch.zeros_like(z_coord)
        return torch.cat([x_coord, z_coord, cos_yaw, sin_yaw], dim=-1)

    dx = torch.zeros_like(x_coord)
    dz = torch.zeros_like(z_coord)
    dx[:, 0:1] = x_coord[:, 1:2] - x_coord[:, 0:1]
    dz[:, 0:1] = z_coord[:, 1:2] - z_coord[:, 0:1]
    dx[:, 1:] = x_coord[:, 1:] - x_coord[:, :-1]
    dz[:, 1:] = z_coord[:, 1:] - z_coord[:, :-1]

    sq = dx * dx + dz * dz
    short = sq < eps * eps
    norm = sq.sqrt().clamp(min=eps)
    cos_yaw = torch.where(short, torch.ones_like(dx), dx / norm)
    sin_yaw = torch.where(short, torch.zeros_like(dz), dz / norm)
    return torch.cat([x_coord, z_coord, cos_yaw, sin_yaw], dim=-1)


def aggregate_frame_to_token_4x(
    feats_frame: torch.Tensor,
    token_len: int,
    mode: str = "last",
) -> torch.Tensor:
    """
    Aggregate frame-level features to token-level (temporal factor 4).

    Args:
        feats_frame: (B, T, C)
        token_len: target token length (T_token)
        mode: "last" or "mean"

    Returns:
        feats_token: (B, T_token, C)
    """
    if token_len <= 0:
        return feats_frame[:, :0, :]
    if feats_frame.size(1) == 0:
        return feats_frame[:, :0, :]

    mode = (mode or "last").lower()
    t = feats_frame.size(1)

    if mode == "last":
        # Token k uses last frame in its 4-frame window: index = 4*(k+1)-1
        idx = (torch.arange(token_len, device=feats_frame.device) + 1) * 4 - 1
        idx = idx.clamp(min=0, max=t - 1)
        return feats_frame.index_select(dim=1, index=idx)

    if mode == "mean":
        # Mean pool each 4-frame window [4k, 4k+4). If window exceeds T, use whatever remains.
        out = []
        for k in range(token_len):
            a = 4 * k
            b = min(4 * (k + 1), t)
            if a >= t:
                # pad with zeros if token_len is longer than available frames
                out.append(torch.zeros_like(feats_frame[:, :1, :]))
            else:
                out.append(feats_frame[:, a:b, :].mean(dim=1, keepdim=True))
        return torch.cat(out, dim=1)

    raise ValueError(f"Unsupported aggregate mode: {mode!r}. Use 'last' or 'mean'.")


def build_traj_emb_from_batch(
    x: dict,
    seq_len: int,
    device: torch.device,
    traj_encoder: torch.nn.Module | None,
    use_traj_cond: bool,
    traj_drop_out: float,
    training_dropout: bool,
) -> torch.Tensor | None:
    """
    返回 TrajEncoder 输出 (B,T,traj_enc_dim)，供 ``WanModel.forward(..., traj_emb=...)``。
    参数名 ``traj_emb`` 为历史兼容，语义是 **encoder 输出、尚未** ``traj_in_proj``。
    无轨迹或 dropout 时返回 None。优先 `traj_features`；否则由 `traj` xyz 经路径朝向补四维。
    """
    if not use_traj_cond or traj_encoder is None:
        return None
    if training_dropout and np.random.rand() <= traj_drop_out:
        return None

    def _mask_to_token(mask_any: torch.Tensor | None) -> torch.Tensor | None:
        """
        Accept either token-level (B, T_token) or frame-level (B, T_frame) mask.
        Convert to token-level (B, seq_len) by 4x aggregation if needed.
        """
        if mask_any is None:
            return None
        m = mask_any.to(device=device, dtype=torch.float32)
        if m.dim() == 1:
            m = m.unsqueeze(0)
        if m.size(1) == seq_len:
            return m
        # Frame-level mask -> token-level: take max within each 4-frame window.
        # (Any observed frame in the window means the token is observed.)
        bsz, t = m.shape
        out = []
        for k in range(seq_len):
            a = 4 * k
            b = min(4 * (k + 1), t)
            if a >= t:
                out.append(torch.zeros(bsz, 1, device=device, dtype=m.dtype))
            else:
                out.append(m[:, a:b].amax(dim=1, keepdim=True))
        return torch.cat(out, dim=1)

    aggregate_mode = x.get("traj_aggregate_mode", None) or "last"

    feats_token = None
    mask_token = None

    # Preferred: frame-level traj_features (B, T_frame, 4) prepared by dataset / caller.
    if "traj_features" in x and x["traj_features"] is not None:
        feats_any = x["traj_features"].to(device)
        if feats_any.dim() == 2:
            feats_any = feats_any.unsqueeze(0)
        # If it's already token-level, keep it; otherwise aggregate 4x to token-level.
        if feats_any.size(1) == seq_len:
            feats_token = feats_any
        else:
            feats_token = aggregate_frame_to_token_4x(
                feats_any, token_len=seq_len, mode=aggregate_mode
            )
        mask_token = _mask_to_token(x.get("traj_mask_token", None))

    # Fallback: raw traj xyz (B, T_frame, 3) -> features_4d (frame) -> aggregate to token.
    elif "traj" in x and x["traj"] is not None:
        traj_any = x["traj"].to(device)
        if traj_any.dim() == 2:
            traj_any = traj_any.unsqueeze(0)
        feats_frame = xyz_traj_to_features_4d(traj_any)
        feats_token = aggregate_frame_to_token_4x(
            feats_frame, token_len=seq_len, mode=aggregate_mode
        )
        mask_token = _mask_to_token(x.get("traj_mask_token", None))

    if feats_token is None:
        return None

    if mask_token is not None:
        feats_token = feats_token * mask_token.unsqueeze(-1).to(dtype=feats_token.dtype)

    return traj_encoder(feats_token)
