"""
检查轨迹分支是否有梯度（避免“训练看起来在跑但 traj 分支没被更新”）。

会对一个 batch 做一次 forward + backward（仅用 out["total"]），然后打印：
- traj_encoder 参数 grad norm
- WanModel.traj_in_proj / traj_type_embed grad norm（如果存在）
- 各 LoRA(Q/K/V) traj 参数 grad norm（如果存在）

运行：
    conda run -n flooddiffusion python scripts/check_traj_grad_flow.py --config configs/ldf.yaml --split train
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


def _grad_norm(p: torch.nn.Parameter) -> float:
    if p.grad is None:
        return 0.0
    return float(p.grad.detach().float().norm().item())


def _group_grad_norm(named_params, substrings: tuple[str, ...]) -> float:
    s = 0.0
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if any(ss in n for ss in substrings):
            g = _grad_norm(p)
            s += g * g
    return float(np.sqrt(s))

def _debug_first_grads(named_params, prefix: str, max_items: int = 6):
    printed = 0
    for n, p in named_params:
        if not n.startswith(prefix):
            continue
        if not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            print(f"  {n}: grad=None")
        else:
            gf = g.detach().float()
            print(
                f"  {n}: grad_norm={float(gf.norm().item()):.6e} "
                f"grad_absmax={float(gf.abs().max().item()):.6e}"
            )
        printed += 1
        if printed >= max_items:
            break


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

    ds = HumanML3DDataset(cfg, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dl))

    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params)
    ckpt_path = cfg.test_ckpt if "test_ckpt" in cfg else cfg.resume_ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    use_traj_cond = bool(cfg.model.params.get("use_traj_cond", False))
    strict_load = not use_traj_cond
    model.load_state_dict(ckpt["state_dict"], strict=strict_load)
    model = model.to(device).train()

    # mimic train_ldf model_batch
    model_batch = dict(batch)
    model_batch["feature"] = batch["token"]
    model_batch["feature_length"] = batch["token_length"]
    if "token_text_end" in batch:
        model_batch["feature_text_end"] = batch["token_text_end"]
    for k, v in list(model_batch.items()):
        if torch.is_tensor(v):
            model_batch[k] = v.to(device)

    # simple optimizer over trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    opt.zero_grad(set_to_none=True)

    out = model(model_batch)
    loss = out["total"]
    loss.backward()

    named = list(model.named_parameters())
    g_traj_encoder = _group_grad_norm(named, ("traj_encoder.",))
    g_traj_in_proj = _group_grad_norm(named, ("traj_in_proj", "traj_type_embed"))
    g_lora = _group_grad_norm(named, ("lora_q_", "lora_k_", "lora_v_"))

    print("=== check_traj_grad_flow ===")
    print(f"split={args.split} batch_size={args.batch_size} device={device} use_traj_cond={use_traj_cond}")
    print(f"loss.total={float(loss.detach().item()):.6f}")
    print(f"grad_norm traj_encoder.* = {g_traj_encoder:.6e}")
    print(f"grad_norm traj_in_proj/type_embed = {g_traj_in_proj:.6e}")
    print(f"grad_norm traj LoRA (q/k/v) = {g_lora:.6e}")
    print("\nSample grads:")
    _debug_first_grads(named, "traj_encoder.")
    _debug_first_grads(named, "model.traj_in_proj")

    # sanity: do one step
    opt.step()

    # Second pass: after traj_in_proj is updated from zero-init, traj_encoder should receive gradient.
    opt.zero_grad(set_to_none=True)
    out2 = model(model_batch)
    loss2 = out2["total"]
    loss2.backward()
    named2 = list(model.named_parameters())
    g2_traj_encoder = _group_grad_norm(named2, ("traj_encoder.",))
    g2_traj_in_proj = _group_grad_norm(named2, ("traj_in_proj", "traj_type_embed"))
    g2_lora = _group_grad_norm(named2, ("lora_q_", "lora_k_", "lora_v_"))
    print("\n--- after 1 optimizer step, second backward ---")
    print(f"loss.total={float(loss2.detach().item()):.6f}")
    print(f"grad_norm traj_encoder.* = {g2_traj_encoder:.6e}")
    print(f"grad_norm traj_in_proj/type_embed = {g2_traj_in_proj:.6e}")
    print(f"grad_norm traj LoRA (q/k/v) = {g2_lora:.6e}")
    _debug_first_grads(named2, "traj_encoder.")


if __name__ == "__main__":
    main()

