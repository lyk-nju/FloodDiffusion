"""
阶段 0 校验脚本：验证 dataloader 输出的 traj、traj_length、traj_mask 是否正确。

运行方式:
    python -m datasets.verify_traj_batch
    或
    cd FloodDiffusion && python datasets/verify_traj_batch.py
"""
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.humanml3d import HumanML3DDataset, collate_fn


def main():
    # 加载配置（与 train_ldf 一致）
    cfg = OmegaConf.create({})
    paths_path = "configs/paths.yaml" if os.path.exists("configs/paths.yaml") else "configs/paths_default.yaml"
    cfg = OmegaConf.merge(cfg, OmegaConf.load(paths_path))
    cfg = OmegaConf.merge(cfg, OmegaConf.load("configs/ldf.yaml"))

    print("Loading HumanML3D dataset (val split)...")
    try:
        dataset = HumanML3DDataset(cfg, split="val")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Ensure raw_data/HumanML3D exists and paths are correct in configs/paths.yaml")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Check data paths.")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )

    batch = next(iter(dataloader))
    if batch is None:
        print("Batch is None.")
        return

    print("\n=== Batch keys ===")
    print(list(batch.keys()))

    print("\n=== Shape check ===")
    B = batch["feature"].shape[0]
    T_max_feat = batch["feature"].shape[1]
    T_max_traj = batch["traj"].shape[1] if "traj" in batch else "N/A"

    print(f"feature:        {batch['feature'].shape}  (B, T_max, 263)")
    print(f"feature_length: {batch['feature_length'].shape} = {batch['feature_length'].tolist()}")
    if "traj" in batch:
        print(f"traj:           {batch['traj'].shape}  (B, T_max, 3)")
        print(f"traj_length:    {batch['traj_length'].shape} = {batch['traj_length'].tolist()}")
        print(f"traj_mask:      {batch['traj_mask'].shape}  (B, T_max)")
    else:
        print("traj: NOT FOUND - check that feature_path is set and feature is loaded")

    # 对齐检查
    if "traj" in batch:
        print("\n=== Alignment check ===")
        for i in range(min(2, B)):
            feat_len = batch["feature_length"][i].item()
            traj_len = batch["traj_length"][i].item()
            assert feat_len == traj_len, f"Sample {i}: feature_length={feat_len} != traj_length={traj_len}"
        print("feature_length == traj_length for all samples: OK")

        # traj_mask 非零比例
        mask_sum = batch["traj_mask"].sum(dim=1)
        print(f"\ntraj_mask sum per sample (should equal traj_length): {mask_sum.tolist()}")

    print("\n=== Verify passed ===")


if __name__ == "__main__":
    main()
