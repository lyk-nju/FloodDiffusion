"""
验证帧级 traj_features 与根轨迹 xz 一致，以及 token_mask 与 traj_mask 的 4× 关系（FloodNet / Floodcontrol 对齐管线）。

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


def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ldf.yaml")
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val", "test"]
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    _ensure_repo_root_in_path()

    from datasets.humanml3d import HumanML3DDataset, collate_fn
    from utils.initialize import load_config

    cfg_wrap = load_config(args.config)
    cfg = cfg_wrap.config
    ds = HumanML3DDataset(cfg, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    b = next(iter(dl))

    required = ["traj", "traj_length", "token_length", "traj_features", "traj_mask"]
    for k in required:
        if k not in b:
            raise KeyError(f"batch missing {k!r}. Keys={sorted(b.keys())}")
    if "token_mask" not in b:
        raise KeyError(f"batch missing 'token_mask'. Keys={sorted(b.keys())}")

    traj = b["traj"].float()
    traj_len = b["traj_length"].long()
    tok_len = b["token_length"].long()
    tf = b["traj_features"].float()
    m_tok = b["token_mask"].float()
    m_fr = b["traj_mask"].float()

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

        xz_traj = traj[i, :Lf, [0, 2]]
        xz_tf = tf[i, :Lf, [0, 1]]
        err = (xz_traj - xz_tf).abs().max().item()
        max_xz_err = max(max_xz_err, float(err))
        if err > atol:
            bad_xz += 1

        expanded = np.repeat(_to_np(m_tok[i, :Lt]).astype(np.float32), 4)
        compare_len = min(expanded.shape[0], Lf)
        mf = _to_np(m_fr[i, :Lf]).astype(np.float32)
        if not np.allclose(mf[:compare_len], expanded[:compare_len], rtol=0, atol=0):
            bad_mask += 1

    print("=== check_traj_alignment (frame-level + 4× mask) ===")
    print(f"split={args.split} batch_size={args.batch_size} atol={atol}")
    print(f"max |xz(traj) - xz(traj_features[:T])| = {max_xz_err:.6e}")
    print(f"samples with xz misalignment (>atol): {bad_xz}/{B}")
    print(f"samples with traj_mask vs repeat(token_mask,4) mismatch: {bad_mask}/{B}")

    if bad_xz > 0 or bad_mask > 0:
        raise SystemExit("Alignment check failed (see counts above).")


if __name__ == "__main__":
    main()
