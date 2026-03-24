"""
HumanML3D 测试夹具（非 unittest 文件，供 Task1/2/3 测试模块导入）。

**内容**
- 从 ``configs/paths*.yaml`` + ``configs/ldf.yaml`` 合并出与训练一致的 ``OmegaConf``。
- ``try_get_val_batch``：构造 ``HumanML3DDataset(split="val")`` + ``collate_fn``，取 **一个** batch。

**逻辑**
- 路径相对 **项目根**（本文件上两级目录），避免依赖当前工作目录误配。
- ``max_token_length`` 非空时多次 shuffle 采样 dataloader，直到 ``token_length.max()`` 不超过阈值，
  便于 Wan 等小模型测试控显存；为 ``None`` 时直接 ``next(iter(dl))``，与旧 ``verify_traj_batch`` 一致。
- 任一步异常或空数据集返回 ``None``，调用方 ``raise SkipTest``，不把「无数据」当成测试失败。
"""

from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def project_root() -> str:
    return _ROOT


def merged_ldf_cfg():
    """与 ``verify_traj_batch`` / train 一致：paths + ldf.yaml。"""
    cfg_dir = os.path.join(project_root(), "configs")
    paths_path = os.path.join(cfg_dir, "paths.yaml")
    if not os.path.isfile(paths_path):
        paths_path = os.path.join(cfg_dir, "paths_default.yaml")
    ldf_path = os.path.join(cfg_dir, "ldf.yaml")
    if not os.path.isfile(ldf_path):
        return None
    cfg = OmegaConf.create({})
    cfg = OmegaConf.merge(cfg, OmegaConf.load(paths_path))
    cfg = OmegaConf.merge(cfg, OmegaConf.load(ldf_path))
    return cfg


def try_get_val_batch(
    *,
    batch_size: int = 4,
    max_token_length: int | None = 64,
    max_dataloader_tries: int = 32,
) -> dict | None:
    """
    从 val split 取一个 batch。若 ``max_token_length`` 非空，在 dataloader 中尽量选
    ``token_length.max() <= max_token_length`` 的 batch（多 shuffle 尝试）。
    """
    cfg = merged_ldf_cfg()
    if cfg is None:
        return None
    try:
        from datasets.humanml3d import HumanML3DDataset, collate_fn
    except ImportError:
        return None

    try:
        dataset = HumanML3DDataset(cfg, split="val")
    except Exception:
        return None

    if len(dataset) == 0:
        return None

    if max_token_length is None:
        dl = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        return next(iter(dl))

    for _ in range(max_dataloader_tries):
        dl = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        batch = next(iter(dl))
        if "token_length" not in batch:
            return batch
        if int(batch["token_length"].max().item()) <= max_token_length:
            return batch
    return None
