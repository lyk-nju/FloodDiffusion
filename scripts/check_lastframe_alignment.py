"""
纯脚本验证：last-frame 对齐是否一致（无需训练）。

检查内容（HumanML3D 数据管线输出）：
1) token 级 traj_features[k] 的 xz 是否等于 frame 级 traj[4*(k+1)-1] 的 xz（允许极小数值误差）
2) traj_mask（frame 级）是否只在 last-frame 索引处为 1，且与 traj_mask_token 保持一致

运行：
    conda run -n flooddiffusion python scripts/check_lastframe_alignment.py --config configs/ldf.yaml --split train
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


def _ensure_repo_root_in_path():
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ldf.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    _ensure_repo_root_in_path()

    from utils.initialize import load_config
    from datasets.humanml3d import HumanML3DDataset, collate_fn

    cfg_wrap = load_config(args.config)
    cfg = cfg_wrap.config

    ds = HumanML3DDataset(cfg, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    b = next(iter(dl))

    required = ["traj", "traj_length", "token_length", "traj_features", "traj_mask", "traj_mask_token"]
    for k in required:
        if k not in b:
            raise KeyError(f"batch missing {k!r}. Keys={sorted(b.keys())}")

    traj = b["traj"].float()  # (B, T, 3)
    traj_len = b["traj_length"].long()  # (B,)
    tok_len = b["token_length"].long()  # (B,)
    tf = b["traj_features"].float()  # (B, T_token, 4)
    m_tok = b["traj_mask_token"].float()  # (B, T_token)
    m_fr = b["traj_mask"].float()  # (B, T)

    B = int(traj.shape[0])
    atol = float(args.atol)
    max_xz_err = 0.0
    bad_xz = 0
    bad_mask = 0

    for i in range(B):
        Lf = int(traj_len[i].item())
        Lt = int(tok_len[i].item())
        if Lf <= 0 or Lt <= 0:
            continue

        # 1) xz last-frame alignment: traj_features[k,0:2] should match traj[last_idx, xz]
        # last_idx = 4*(k+1)-1
        idx = torch.arange(Lt) * 4 + 3
        idx = torch.clamp(idx, 0, Lf - 1)
        xz_from_traj = traj[i, idx, :][:, [0, 2]]  # (Lt,2)
        xz_from_tf = tf[i, :Lt, :][:, [0, 1]]      # (Lt,2) where tf=[x,z,cos,sin]
        err = (xz_from_traj - xz_from_tf).abs().max().item()
        max_xz_err = max(max_xz_err, float(err))
        if err > atol:
            bad_xz += 1

        # 2) mask mapping: frame mask should be 1 only at last-frame indices where token mask=1
        mapped = torch.zeros((Lf,), dtype=torch.float32)
        keep = m_tok[i, :Lt] > 0.5
        mapped[idx[keep]] = 1.0
        mf = m_fr[i, :Lf]
        if not torch.allclose(mf, mapped, rtol=0, atol=0):
            bad_mask += 1

    print("=== check_lastframe_alignment ===")
    print(f"split={args.split} batch_size={args.batch_size} atol={atol}")
    print(f"max |xz(traj_last) - xz(traj_features)| = {max_xz_err:.6e}")
    print(f"samples with xz misalignment (>atol): {bad_xz}/{B}")
    print(f"samples with mask mapping mismatch: {bad_mask}/{B}")

    if bad_xz > 0 or bad_mask > 0:
        raise SystemExit("Alignment check failed (see counts above).")


if __name__ == "__main__":
    main()

