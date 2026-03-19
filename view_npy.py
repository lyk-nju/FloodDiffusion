#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
View and summarize a .npy file.

Examples:
  /home/yuankai/.conda/envs/flooddiffusion/bin/python3.10 view_npy.py "path/to/xxx.npy"
  /home/yuankai/.conda/envs/flooddiffusion/bin/python3.10 view_npy.py "path/to/xxx.npy" --plot --outdir /tmp/npy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np


def _is_numeric_array(arr: np.ndarray) -> bool:
    return arr.dtype != object and np.issubdtype(arr.dtype, np.number)


def summarize(arr: Any, max_print: int = 20) -> None:
    print("=== NPY Summary ===")
    if not isinstance(arr, np.ndarray):
        print("Type:", type(arr))
        print("Value:", arr)
        return

    print("dtype:", arr.dtype)
    print("shape:", arr.shape)

    if arr.dtype == object:
        flat = arr.ravel()
        print("dtype is object; object len:", len(flat))
        for i in range(min(len(flat), max_print)):
            v = flat[i]
            if isinstance(v, np.ndarray):
                print(f"[{i}] ndarray shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"[{i}] type={type(v)} val={v}")
        return

    # Numeric array: show min/max + a small preview
    flat = arr.ravel()
    if flat.size == 0:
        print("Empty array.")
        return

    if _is_numeric_array(arr):
        mn = float(flat.min())
        mx = float(flat.max())
        print(f"min={mn} max={mx}")

        # Quick "mask-like" hint (values near 0/1).
        uniq_sample = np.unique(flat[: min(flat.size, 1000)])
        if uniq_sample.size <= 6:
            print("unique(sample<=1000):", uniq_sample)

    preview_n = min(flat.size, max_print)
    print(f"first {preview_n} values (ravel):")
    print(flat[:preview_n])


def plot_if_possible(arr: Any, outdir: str | None, prefix: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available; skip plotting.")
        return

    if not isinstance(arr, np.ndarray):
        print("Not a numpy array; skip plotting.")
        return
    if arr.dtype == object:
        print("object dtype; skip plotting.")
        return

    # For plotting we want reasonable ranges; convert to float32 for safety.
    a = arr

    save_path: str | None = None
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        save_path = os.path.join(outdir, f"{prefix}_plot.png")

    if a.ndim == 0:
        print("0D array; skip plotting.")
        return

    if a.ndim == 1:
        plt.figure(figsize=(8, 3))
        plt.plot(np.arange(a.shape[0]), a.astype(np.float32))
        plt.title(f"1D npy: {a.shape[0]} values")
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print("Saved plot:", save_path)
        else:
            plt.show()
        plt.close()
        return

    if a.ndim == 2:
        plt.figure(figsize=(6, 5))
        plt.imshow(a.astype(np.float32), aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.title(f"2D npy: {a.shape}")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print("Saved plot:", save_path)
        else:
            plt.show()
        plt.close()
        return

    # ndim >= 3: try to reduce to a 2D slice
    # Strategy: if last dim looks like channel, pick first channel; else pick index 0 on leading dims.
    if a.ndim >= 3 and a.shape[-1] in (1, 3, 4):
        img = a[..., 0] if a.shape[-1] == 1 else a[..., :3]
        plt.figure(figsize=(6, 5))
        if img.ndim == 2:
            plt.imshow(img.astype(np.float32), aspect="auto", cmap="viridis")
            plt.colorbar()
        else:
            # best effort: show first 3 channels as pseudo-RGB
            plt.imshow(img.astype(np.float32))
        plt.title(f"ndim={a.ndim} treated as image-like")
    else:
        # Pick first indices for leading dims and plot the last two dims.
        idx = (0,) * (a.ndim - 2)
        img2d = a[idx + (slice(None), slice(None))].astype(np.float32)
        plt.figure(figsize=(6, 5))
        plt.imshow(img2d, aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.title("ndim reduced to slice")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved plot:", save_path)
    else:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="View and summarize .npy file")
    parser.add_argument("npy_path", type=str, help="Path to .npy file")
    parser.add_argument("--plot", action="store_true", help="Try to plot (1D/2D/2D-slice)")
    parser.add_argument("--outdir", type=str, default=None, help="Save plots to directory")
    parser.add_argument("--max-print", type=int, default=20, help="Max values to print")
    args = parser.parse_args()

    npy_path = Path(args.npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(str(npy_path))

    # allow_pickle=True is needed because some npy may contain objects.
    arr = np.load(str(npy_path), allow_pickle=True)

    summarize(arr, max_print=args.max_print)

    if args.plot:
        plot_if_possible(arr, outdir=args.outdir, prefix=npy_path.stem)


if __name__ == "__main__":
    main()

