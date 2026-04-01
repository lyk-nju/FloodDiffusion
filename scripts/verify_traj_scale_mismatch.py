"""
验证嫌疑一：GT traj 是否从「标准化后的 new_joint_vecs」直接提取，导致与 VAE decode 的真实尺度不一致。

脚本会对同一条样本计算三种 root 轨迹（xz 重点）并打印统计：
1) traj_from_feature_norm: 直接从 feature(T,263) 提取（可能是标准化尺度）
2) traj_from_feature_denorm: 先用 Mean/Std 反标准化 feature 再提取（真实尺度）
3) traj_from_vae_decode_token: 从 token(T_token, z_dim) 经 VAE decode 得到 motion(T,263) 再提取（真实尺度）

运行（在 Floodcontrol 目录或项目根均可）：
    python3 Floodcontrol/scripts/verify_traj_scale_mismatch.py --config Floodcontrol/configs/ldf.yaml
或：
    python3 scripts/verify_traj_scale_mismatch.py --config configs/ldf.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
from torch_ema import ExponentialMovingAverage


def _ensure_repo_root_in_path():
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _stats_xyz(name: str, xyz: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float64)
    xz = xyz[:, [0, 2]]
    print(f"\n[{name}] shape={xyz.shape}")
    print(f"  xyz min={xyz.min(axis=0)} max={xyz.max(axis=0)}")
    print(f"  xz  min={xz.min(axis=0)} max={xz.max(axis=0)}")
    # rough per-step displacement magnitude on xz
    if len(xz) >= 2:
        dxz = xz[1:] - xz[:-1]
        step = np.linalg.norm(dxz, axis=-1)
        print(
            f"  xz step |Δ| mean={step.mean():.6f} p50={np.median(step):.6f} p95={np.quantile(step, 0.95):.6f}"
        )


def _rmse_xz(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    L = min(len(a), len(b))
    if L <= 0:
        return float("nan")
    da = a[:L, [0, 2]]
    db = b[:L, [0, 2]]
    mse = np.mean(np.sum((da - db) ** 2, axis=-1))
    return float(np.sqrt(mse))

def _rmse_all(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    L = min(len(a), len(b))
    if L <= 0:
        return float("nan")
    da = a[:L]
    db = b[:L]
    mse = np.mean((da - db) ** 2)
    return float(np.sqrt(mse))


def _stats_feat(name: str, feat: np.ndarray) -> None:
    feat = np.asarray(feat, dtype=np.float64)
    print(f"\n[{name}] shape={feat.shape}")
    print(
        f"  mean={feat.mean():.6f} std={feat.std():.6f} min={feat.min():.6f} max={feat.max():.6f}"
    )
    d = min(feat.shape[-1], 8)
    mins = feat[:, :d].min(axis=0)
    maxs = feat[:, :d].max(axis=0)
    print(f"  first{d} dims min={mins} max={maxs}")


def _infer_downsample_factor(token_len: int, feat_len: int) -> float:
    if token_len <= 0:
        return float("nan")
    return float(feat_len) / float(token_len)


def _load_mean_std(cfg) -> Tuple[np.ndarray, np.ndarray]:
    # Default locations used by VAE configs.
    # For LDF config we infer from dirs.raw_data.
    raw = cfg.dirs.raw_data
    mean_path = os.path.join(raw, "HumanML3D", "Mean.npy")
    std_path = os.path.join(raw, "HumanML3D", "Std.npy")
    mean = np.load(mean_path)
    std = np.load(std_path)
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ldf.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0, help="sample index inside the dataset (after filtering)")
    parser.add_argument("--use_ema", action="store_true", help="use EMA weights for VAE if present in ckpt")
    parser.add_argument(
        "--print_decoded_stats",
        action="store_true",
        help="print feature/decoded stats (helps diagnose token-VAE mismatch)",
    )
    args = parser.parse_args()

    _ensure_repo_root_in_path()

    from omegaconf import OmegaConf

    from datasets.humanml3d import HumanML3DDataset
    from utils.motion_process import extract_root_trajectory_263, extract_root_trajectory_263_torch
    from utils.initialize import load_config, instantiate

    cfg_wrap = load_config(args.config)
    cfg = cfg_wrap.config

    # Dataset: get one sample (not collated).
    ds = HumanML3DDataset(cfg, split=args.split)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty; check paths.yaml and raw_data.")
    sample = ds[args.index % len(ds)]

    feature = sample["feature"]  # (T,263) numpy
    token = sample.get("token", None)  # (T_token,z) numpy
    if token is None:
        raise RuntimeError("Sample has no token; check token_path in config.")

    mean, std = _load_mean_std(cfg)
    feature_denorm = feature * std[None, :] + mean[None, :]

    traj_feat_norm = extract_root_trajectory_263(feature)
    traj_feat_denorm = extract_root_trajectory_263(feature_denorm)

    # Load VAE and decode token -> motion 263D
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = instantiate(
        target=cfg.test_vae.target,
        cfg=None,
        hfstyle=False,
        **cfg.test_vae.params,
    )
    ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    vae.load_state_dict(ckpt["state_dict"], strict=True)
    if args.use_ema and "ema_state" in ckpt:
        ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        ema.load_state_dict(ckpt["ema_state"])
        ema.copy_to(vae.parameters())
    vae = vae.to(device).eval()

    with torch.no_grad():
        token_t = torch.from_numpy(token).float().to(device)[None, :, :]
        decoded = vae.decode(token_t)[0].float()  # (T_dec,263) on device
        traj_dec = extract_root_trajectory_263_torch(decoded[None, :, :])[0].detach().cpu().numpy()
        decoded_np = decoded.detach().cpu().numpy()

    # Print stats
    print("=== verify_traj_scale_mismatch ===")
    print(f"split={args.split} index={args.index} name={sample.get('name')}")
    print(f"feature shape={feature.shape} token shape={np.asarray(token).shape}")
    print(
        f"infer downsample factor (feature_len/token_len): "
        f"{_infer_downsample_factor(int(np.asarray(token).shape[0]), int(feature.shape[0])):.3f}"
    )

    if args.print_decoded_stats:
        _stats_feat("feature (as loaded from new_joint_vecs)", feature)
        _stats_feat("feature_denorm (feature*Std+Mean)", feature_denorm)
        _stats_feat("decoded (vae.decode(token))", decoded_np)

        print("\nRMSE over 263D feature space (lower means token-VAE matches dataset feature scale):")
        print(f"  feature vs decoded: { _rmse_all(feature, decoded_np):.6f}")
        print(f"  feature_denorm vs decoded: { _rmse_all(feature_denorm, decoded_np):.6f}")

    _stats_xyz("traj_from_feature_norm (可能是标准化尺度)", traj_feat_norm)
    _stats_xyz("traj_from_feature_denorm (Mean/Std 反标准化后)", traj_feat_denorm)
    _stats_xyz("traj_from_vae_decode_token (decode 后)", traj_dec)

    print("\nRMSE_xz comparisons (lower means consistent scale/coords):")
    print(f"  norm vs denorm: { _rmse_xz(traj_feat_norm, traj_feat_denorm):.6f}")
    print(f"  denorm vs decode: { _rmse_xz(traj_feat_denorm, traj_dec):.6f}")
    print(f"  norm vs decode: { _rmse_xz(traj_feat_norm, traj_dec):.6f}")

    print(
        "\nInterpretation:\n"
        "- 如果 denorm vs decode 很小，而 norm vs decode 很大：说明 GT traj 应该用 denorm。\n"
        "- 如果 feature vs decoded 的 RMSE 很大：说明 token 与当前加载的 VAE 不匹配（或 VAE 的 mean/std 配置不同）。\n"
        "- 如果三者都很大：可能还有坐标系/对齐（crop、downsample factor）问题。\n"
    )


if __name__ == "__main__":
    main()

