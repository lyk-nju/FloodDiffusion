"""
Task1：dataloader 轨迹字段与训练管线对齐。

**测试内容**
- batch 是否含 ``feature`` / ``feature_length``。
- ``traj`` 存在时：``feature_length`` 与 ``traj_length`` 逐样本一致。
- ``token`` 存在时：``token_length`` 与 ``feature_length`` 的约 4×（VAE）关系合理。
- ``traj_features`` 存在时：帧级长度与 ``traj_length`` / ``feature_length`` 一致（与 FloodNet 一致）。
- ``traj_mask`` 时间维不少于 ``traj_length``。
- **打印**：``token_mask``（token 轴稀疏 mask）与 ``traj_mask``（由 token mask 4× 扩到帧轴再 pad 到 ``traj_length``）的摘要，并校验二者 4× 关系。

**逻辑**
- ``setUpClass`` 调用 ``try_get_val_batch(..., max_token_length=None)`` 拉 **真实 val batch**（与旧脚本一致）。
- 各用例只读 ``self.batch`` 做断言；缺字段的用例 ``skipTest``（例如未开 ``feature_path`` 则无 ``traj``）。
- 无数据集或配置失败时 ``setUpClass`` 抛 ``SkipTest``，整类跳过。

**运行**（项目根）::

    python -m unittest tests.test_task1_traj_batch -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from tests.humanml3d_fixtures import try_get_val_batch
except ImportError:
    from humanml3d_fixtures import try_get_val_batch


class TestHumanML3DTrajBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = try_get_val_batch(batch_size=4, max_token_length=None)
        if cls.batch is None:
            raise unittest.SkipTest("HumanML3D val batch unavailable (paths/data)")

    def test_batch_keys_include_motion(self):
        b = self.batch
        self.assertIn("feature", b)
        self.assertIn("feature_length", b)

    def test_traj_alignment(self):
        b = self.batch
        if "traj" not in b:
            self.skipTest("no traj in batch (feature_path?)")
        B = b["feature"].shape[0]
        for i in range(min(2, B)):
            feat_len = b["feature_length"][i].item()
            traj_len = b["traj_length"][i].item()
            self.assertEqual(
                feat_len,
                traj_len,
                f"sample {i}: feature_length={feat_len} != traj_length={traj_len}",
            )

    def test_token_alignment(self):
        b = self.batch
        if "traj" not in b or "token" not in b:
            self.skipTest("need traj and token")
        B = b["feature"].shape[0]
        for i in range(min(4, B)):
            feat_len = b["feature_length"][i].item()
            tok_len = b["token_length"][i].item()
            expected = feat_len // 4
            self.assertTrue(
                1 <= tok_len <= expected + 1,
                f"sample {i}: token_length={tok_len} vs feature_length={feat_len}",
            )

    def test_traj_features_frame_length(self):
        b = self.batch
        if "traj_features" not in b or "traj_length" not in b:
            self.skipTest("no traj_features")
        B = b["feature"].shape[0]
        for i in range(min(4, B)):
            traj_len = b["traj_length"][i].item()
            tf = b["traj_features"][i]
            n = int(tf.shape[0])
            self.assertEqual(
                n,
                traj_len,
                f"sample {i}: traj_features T={n} != traj_length={traj_len}",
            )

    def test_traj_mask_length(self):
        b = self.batch
        if "traj" not in b:
            self.skipTest("no traj")
        B = b["feature"].shape[0]
        for i in range(min(4, B)):
            traj_len = b["traj_length"][i].item()
            self.assertGreaterEqual(
                b["traj_mask"][i].shape[0],
                traj_len,
                f"sample {i}: traj_mask shorter than traj_length",
            )

    def test_print_token_and_traj_masks(self):
        """打印 token 级 ``token_mask`` 与帧级 ``traj_mask``，并校验 4× 扩展。"""
        b = self.batch
        if "token_mask" not in b or "traj_mask" not in b:
            self.skipTest("need token_mask and traj_mask")
        if "token_length" not in b or "traj_length" not in b:
            self.skipTest("need token_length and traj_length")

        B = b["token_mask"].shape[0]
        n_show = min(B, 4)

        def _short_seq(arr: np.ndarray, width: int = 64) -> str:
            flat = arr.flatten()
            if flat.size <= width:
                return np.array2string(flat, separator="", max_line_width=120)
            head = flat[: width // 2]
            tail = flat[-width // 2 :]
            return (
                np.array2string(head, separator="", max_line_width=120)
                + "..."
                + np.array2string(tail, separator="", max_line_width=120)
            )

        for i in range(n_show):
            tok_len = int(b["token_length"][i].item())
            traj_len = int(b["traj_length"][i].item())
            tok_m = b["token_mask"][i, :tok_len]
            frm_m = b["traj_mask"][i, :traj_len]
            if isinstance(tok_m, torch.Tensor):
                tok_m = tok_m.detach().float().cpu().numpy()
            else:
                tok_m = np.asarray(tok_m, dtype=np.float32)
            if isinstance(frm_m, torch.Tensor):
                frm_m = frm_m.detach().float().cpu().numpy()
            else:
                frm_m = np.asarray(frm_m, dtype=np.float32)

            expanded = np.repeat(tok_m, 4).astype(np.float32)
            compare_len = min(expanded.shape[0], frm_m.shape[0])
            np.testing.assert_allclose(
                frm_m[:compare_len],
                expanded[:compare_len],
                rtol=0,
                atol=0,
                err_msg=f"sample {i}: traj_mask 前段应与 repeat(token_mask,4) 一致",
            )

            keep_tok = float(tok_m.sum())
            keep_frm = float(frm_m[:compare_len].sum())
            print(
                f"\n=== Task1 mask preview sample {i} ===\n"
                f"  token_length={tok_len}  traj_length={traj_len}\n"
                f"  token_mask (token 轴稀疏 mask): "
                f"sum={keep_tok:.0f}  density={keep_tok / max(tok_len, 1):.3f}\n"
                f"  seq[{_short_seq(tok_m, 72)}]\n"
                f"  traj_mask (帧轴, 由上一行 4× 扩展再 pad/截断到 traj_length): "
                f"有效比对长度 compare_len={compare_len}  sum(该段)={keep_frm:.0f}\n"
                f"  seq[{_short_seq(frm_m[: min(compare_len, 96)], 72)}]\n"
                f"  4× 扩展校验: frm_m[:compare_len] == repeat(tok_m,4)[:compare_len]  OK"
            )


if __name__ == "__main__":
    unittest.main()
