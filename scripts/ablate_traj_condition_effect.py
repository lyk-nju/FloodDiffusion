"""
验证轨迹条件是否真正影响模型输出（前向差异不应为 0）。

对同一个 batch，做两次前向：
1) 正常（use_traj_cond=True 且传入 traj_features + traj_mask_token）
2) 去掉轨迹条件（删 traj_features / traj_mask_token / traj 等字段）

比较输出 loss / control_aux 中 pred_x0_latent_list 的差异。

运行：
    conda run -n flooddiffusion python scripts/ablate_traj_condition_effect.py --config configs/ldf.yaml --split train
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


def _clone_batch_shallow(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v
    return out


def _del_keys(d: dict, keys: list[str]):
    for k in keys:
        if k in d:
            del d[k]


def _tensor_norm(x: torch.Tensor) -> float:
    return float(x.detach().float().norm().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ldf.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    _ensure_repo_root_in_path()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from utils.initialize import load_config, instantiate
    from datasets.humanml3d import HumanML3DDataset, collate_fn

    cfg_wrap = load_config(args.config)
    cfg = cfg_wrap.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset -> batch
    ds = HumanML3DDataset(cfg, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dl))

    # model forward expects token as "feature"
    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params)
    ckpt = torch.load(cfg.test_ckpt if "test_ckpt" in cfg else cfg.resume_ckpt, map_location="cpu", weights_only=False)
    use_traj_cond = bool(cfg.model.params.get("use_traj_cond", False))
    strict_load = not use_traj_cond
    model.load_state_dict(ckpt["state_dict"], strict=strict_load)
    if "ema_state" in ckpt:
        # keep it simple: ignore EMA here; we're only checking sensitivity
        pass
    model = model.to(device).eval()

    # build model_batch (mimic train_ldf.py)
    model_batch = _clone_batch_shallow(batch)
    model_batch["feature"] = batch["token"]
    model_batch["feature_length"] = batch["token_length"]
    if "token_text_end" in batch:
        model_batch["feature_text_end"] = batch["token_text_end"]

    for k, v in list(model_batch.items()):
        if torch.is_tensor(v):
            model_batch[k] = v.to(device)

    model_batch_no = _clone_batch_shallow(model_batch)
    _del_keys(
        model_batch_no,
        [
            "traj_features",
            "traj_features_length",
            "traj_mask_token",
            "traj_features_mask",
            "traj",
            "traj_length",
            "traj_mask",
        ],
    )

    with torch.no_grad():
        out_yes = model(model_batch)
        out_no = model(model_batch_no)

    print("=== ablate_traj_condition_effect ===")
    print(f"split={args.split} batch_size={args.batch_size} device={device} use_traj_cond={use_traj_cond}")
    print("loss keys yes:", [k for k in out_yes.keys()])
    print("loss keys no :", [k for k in out_no.keys()])
    print(f"total_loss yes={float(out_yes['total']):.6f} no={float(out_no['total']):.6f}")

    # Compare control_aux if present
    aux_yes = out_yes.get("control_aux")
    aux_no = out_no.get("control_aux")
    if aux_yes and "pred_x0_latent_list" in aux_yes:
        py = aux_yes["pred_x0_latent_list"]
        pn = aux_no["pred_x0_latent_list"] if (aux_no and "pred_x0_latent_list" in aux_no) else None
        if pn is None:
            print("control_aux present for yes, absent for no (expected if no traj fields).")
        else:
            # compute norm difference per sample
            diffs = []
            for a, b in zip(py, pn):
                diffs.append(float((a - b).detach().float().norm().item()))
            print("pred_x0_latent_list Δnorm per-sample:", diffs)
            print("mean Δnorm:", float(np.mean(diffs)))
    else:
        print("No control_aux in output (prediction_type might be 'noise' or traj_cond disabled).")


if __name__ == "__main__":
    main()

