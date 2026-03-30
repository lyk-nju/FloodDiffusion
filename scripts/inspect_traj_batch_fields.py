"""
检查训练 batch 中轨迹相关字段是否正确、是否有非零信息、以及是否与 token_length 对齐。

运行：
    conda run -n flooddiffusion python scripts/inspect_traj_batch_fields.py --config configs/ldf.yaml --split train
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


def _np(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ldf.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    _ensure_repo_root_in_path()

    from utils.initialize import load_config, instantiate

    cfg_wrap = load_config(args.config)
    cfg = cfg_wrap.config

    from datasets.humanml3d import HumanML3DDataset, collate_fn

    ds = HumanML3DDataset(cfg, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dl))

    print("=== inspect_traj_batch_fields ===")
    print(f"split={args.split} batch_size={args.batch_size} n_samples_in_ds={len(ds)}")
    print("keys:", sorted(batch.keys()))

    def show_len(key):
        if key not in batch:
            return
        v = batch[key]
        if torch.is_tensor(v):
            print(f"- {key}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"- {key}: type={type(v)}")

    for k in ["feature", "token", "traj", "traj_features", "traj_mask_token", "traj_features_mask", "traj_mask"]:
        show_len(k)

    tok_len = batch.get("token_length")
    feat_len = batch.get("feature_length")
    if tok_len is not None:
        print(f"\n[token_length] min={int(tok_len.min())} max={int(tok_len.max())} mean={float(tok_len.float().mean()):.2f}")
    if feat_len is not None:
        print(f"[feature_length] min={int(feat_len.min())} max={int(feat_len.max())} mean={float(feat_len.float().mean()):.2f}")

    # Check masks density and alignment sample-wise
    B = int(tok_len.shape[0]) if tok_len is not None else 0
    for i in range(min(B, 4)):
        print(f"\n--- sample {i} ---")
        tl = int(tok_len[i].item()) if tok_len is not None else None
        fl = int(feat_len[i].item()) if feat_len is not None else None
        print(f"token_length={tl} feature_length={fl}")

        tft = batch.get("traj_features")
        if tft is not None:
            tfti = _np(tft[i, :tl])
            nz = float(np.abs(tfti).sum())
            print(f"traj_features: len={tfti.shape[0]} dim={tfti.shape[1]} abs_sum={nz:.6f}")

        m_tok = batch.get("traj_mask_token")
        if m_tok is None:
            m_tok = batch.get("traj_features_mask")
        if m_tok is not None and tl is not None:
            mt = _np(m_tok[i, :tl]).astype(np.float32)
            keep = float(mt.sum())
            print(f"mask_token: keep={keep:.0f}/{tl} density={keep/max(tl,1):.3f}")

        m_fr = batch.get("traj_mask")
        traj_len = batch.get("traj_length")
        if m_fr is not None and traj_len is not None:
            L = int(traj_len[i].item())
            mf = _np(m_fr[i, :L]).astype(np.float32)
            print(f"mask_frame(traj_mask): keep={mf.sum():.0f}/{L} (should ~= keep_token for last-frame mapping)")


if __name__ == "__main__":
    main()

