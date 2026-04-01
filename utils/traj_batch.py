"""轨迹 batch：路径朝向 [x,z,cos,sin] 与 DiffForcing → WanModel 的轨迹 Encoding。"""

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
    用于数据集 `traj_features`（帧级），与仅能提供路径的推理条件对齐。
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


def build_traj_emb_from_batch(
    x: dict,
    seq_len: int,
    device: torch.device,
    traj_encoder: torch.nn.Module | None,
    use_traj_cond: bool,
    traj_drop_out: float,
    training_dropout: bool,
    local_traj_encoder: torch.nn.Module | None = None,
) -> torch.Tensor | None:
    """
    返回 TrajEncoder 输出 (B, seq_len, traj_enc_dim)。

    与 FloodNet 对齐：
    - 默认数据为 **帧级** ``traj_features`` (B, T_frame, 4)，经 ``local_traj_encoder`` 压成 token 级再 MLP。
    - 若 ``traj_features`` 第二维已等于 ``seq_len``，视为 **token 级**（如流式 buffer），跳过 local。
    - ``token_mask`` 或 ``traj_mask``（帧级）做 gating；``token`` 级 mask 在 local 之后乘。
    """
    if not use_traj_cond or traj_encoder is None:
        return None
    if training_dropout and np.random.rand() <= traj_drop_out:
        return None

    if "traj_features" in x and x["traj_features"] is not None:
        feats_frame = x["traj_features"].to(device)
    elif "traj" in x and x["traj"] is not None:
        feats_frame = xyz_traj_to_features_4d(x["traj"].to(device))
    else:
        return None

    mask_frame = None
    if "traj_mask" in x and x["traj_mask"] is not None:
        mask_frame = x["traj_mask"].to(device=device, dtype=torch.float32)
    elif "token_mask" in x and x["token_mask"] is not None:
        tm = x["token_mask"].to(device=device, dtype=torch.float32)
        mask_frame = tm.repeat_interleave(4, dim=1)
    if mask_frame is not None:
        tf = feats_frame.shape[1]
        if mask_frame.shape[1] < tf:
            pad = mask_frame.new_zeros(mask_frame.shape[0], tf - mask_frame.shape[1])
            mask_frame = torch.cat([mask_frame, pad], dim=1)
        mask_frame = mask_frame[:, :tf]
        feats_frame = feats_frame * mask_frame.unsqueeze(-1).to(dtype=feats_frame.dtype)

    if feats_frame.shape[1] == seq_len:
        feats_tok = feats_frame
    else:
        if local_traj_encoder is None:
            # 旧 ckpt 无缺省时：线性插值到 seq_len（不推荐，仅兜底）
            if feats_frame.shape[1] != seq_len:
                feats_frame = F.interpolate(
                    feats_frame.permute(0, 2, 1),
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
            mask_tok = None
            if "token_mask" in x and x["token_mask"] is not None:
                mask_tok = x["token_mask"].to(device=device, dtype=torch.float32)
            if mask_tok is not None:
                if mask_tok.shape[1] < seq_len:
                    pad = mask_tok.new_zeros(mask_tok.shape[0], seq_len - mask_tok.shape[1])
                    mask_tok = torch.cat([mask_tok, pad], dim=1)
                mask_tok = mask_tok[:, :seq_len]
                feats_frame = feats_frame * mask_tok.unsqueeze(-1).to(dtype=feats_frame.dtype)
            return traj_encoder(feats_frame)

        need = seq_len * 4
        tf = feats_frame.shape[1]
        if tf < need:
            pad = feats_frame.new_zeros(
                feats_frame.shape[0], need - tf, feats_frame.shape[2]
            )
            feats_frame = torch.cat([feats_frame, pad], dim=1)
        feats_frame = feats_frame[:, :need, :]
        feats_4 = feats_frame.reshape(feats_frame.shape[0], seq_len, 4, 4)
        feats_tok = local_traj_encoder(feats_4)

    if "token_mask" in x and x["token_mask"] is not None:
        tm = x["token_mask"].to(device=device, dtype=torch.float32)
        if tm.shape[1] < seq_len:
            pad = tm.new_zeros(tm.shape[0], seq_len - tm.shape[1])
            tm = torch.cat([tm, pad], dim=1)
        tm = tm[:, :seq_len]
        feats_tok = feats_tok * tm.unsqueeze(-1).to(dtype=feats_tok.dtype)

    return traj_encoder(feats_tok)
