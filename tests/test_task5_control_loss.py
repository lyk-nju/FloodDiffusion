"""
Task5：显式控制损失（xz-only）单元测试。

测试内容：
- `y`（竖直）变化不应影响 loss（仅监督 xz）。
- `traj_mask` 为 0 的帧不计入误差。
- 归一化应按有效 mask 帧数进行平均。

运行：
    python -m unittest tests.test_task5_control_loss -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _DummyVAE:
    @staticmethod
    def decode(latent_b1):
        # 模拟 VAE decode：保留前三维作为“root xyz”motion
        return latent_b1[..., :3]


class TestTask5ControlLoss(unittest.TestCase):
    def test_y_dimension_does_not_affect_loss(self):
        import train_ldf as tl

        old_extract = tl.extract_root_trajectory_263_torch
        tl.extract_root_trajectory_263_torch = lambda x: x
        try:
            pred = torch.tensor([[1.0, 10.0, 2.0], [2.0, 20.0, 4.0]], dtype=torch.float32)
            pred4 = torch.cat([pred, torch.zeros(pred.size(0), 1)], dim=-1)  # (T,4)
            pred_list = [pred4]
            traj_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
            traj_length = torch.tensor([2], dtype=torch.long)

            gt1 = torch.tensor([[[1.5, 0.0, 2.5], [2.0, 0.0, 4.0]]], dtype=torch.float32)
            gt2 = gt1.clone()
            gt2[..., 1] = 999.0  # 只改 y

            l1 = tl.CustomLightningModule._compute_control_loss_xz(
                pred_list, gt1, traj_mask, traj_length, _DummyVAE(), torch.device("cpu")
            )
            l2 = tl.CustomLightningModule._compute_control_loss_xz(
                pred_list, gt2, traj_mask, traj_length, _DummyVAE(), torch.device("cpu")
            )
            torch.testing.assert_close(l1, l2, rtol=0, atol=0)
        finally:
            tl.extract_root_trajectory_263_torch = old_extract

    def test_mask_excludes_unobserved_frames(self):
        import train_ldf as tl

        old_extract = tl.extract_root_trajectory_263_torch
        tl.extract_root_trajectory_263_torch = lambda x: x
        try:
            pred = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 100.0]], dtype=torch.float32)
            pred4 = torch.cat([pred, torch.zeros(pred.size(0), 1)], dim=-1)
            pred_list = [pred4]
            gt = torch.zeros(1, 2, 3, dtype=torch.float32)
            traj_length = torch.tensor([2], dtype=torch.long)

            mask_keep_first = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
            loss = tl.CustomLightningModule._compute_control_loss_xz(
                pred_list, gt, mask_keep_first, traj_length, _DummyVAE(), torch.device("cpu")
            )
            # 第一帧 xz 误差为 0，第二帧虽巨大但 mask=0，应不计入
            self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        finally:
            tl.extract_root_trajectory_263_torch = old_extract

    def test_normalization_by_valid_frames(self):
        import train_ldf as tl

        old_extract = tl.extract_root_trajectory_263_torch
        tl.extract_root_trajectory_263_torch = lambda x: x
        try:
            pred = torch.tensor([[1.0, 0.0, 1.0], [3.0, 0.0, 4.0]], dtype=torch.float32)
            pred4 = torch.cat([pred, torch.zeros(pred.size(0), 1)], dim=-1)
            pred_list = [pred4]
            gt = torch.zeros(1, 2, 3, dtype=torch.float32)
            traj_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
            traj_length = torch.tensor([2], dtype=torch.long)

            # 帧0误差: 1^2+1^2=2; 帧1误差: 3^2+4^2=25; 平均=13.5
            loss = tl.CustomLightningModule._compute_control_loss_xz(
                pred_list, gt, traj_mask, traj_length, _DummyVAE(), torch.device("cpu")
            )
            self.assertAlmostEqual(float(loss.item()), 13.5, places=6)
        finally:
            tl.extract_root_trajectory_263_torch = old_extract


if __name__ == "__main__":
    unittest.main()
