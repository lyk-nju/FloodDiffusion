"""
Task3（数据侧）：推理语义的轨迹编码，与 ``generate`` / ``stream_generate_step`` 对齐，不拉 T5 与整图扩散。

**测试内容**
- ``test_inference_traj_emb_from_traj_features``：对真实 batch 的 ``traj_features``（+ mask）调用
  ``build_traj_emb_from_batch(..., training_dropout=False)``，断言输出形状 ``(B, seq_len, dim)`` 且有限；
  与 ``DiffForcingWanModel._build_traj_emb`` 在推理时的 dropout 行为一致。
- ``test_stream_style_xyz_slice_encoding``：若有 ``traj``，取前 ``seq_len`` 帧（不足则 **左侧零 pad**），
  ``xyz_traj_to_features_4d`` → ``TrajEncoder``，对齐 ``stream_generate_step`` 里对 traj_buffer 切片再编码的路径。

**逻辑**
- 只测 **轨迹张量 → traj_emb** 这一段子图；多步去噪、CFG、噪声调度不在此文件覆盖。
- ``seq_len`` 取 batch 内 ``token_length.max()``，与 token 时间轴一致。
- 设备：有 CUDA 用 GPU，否则 CPU（TrajEncoder 很小）。

**依赖**：HumanML3D val 可读；缺 ``traj_features`` 或 ``traj`` 时对应用例 skip。

**运行**（项目根）::

    python -m unittest tests.test_task3_inference_batch -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from tests.humanml3d_fixtures import try_get_val_batch
except ImportError:
    from humanml3d_fixtures import try_get_val_batch


class TestTask3TrajEmbFromDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = try_get_val_batch(batch_size=2, max_token_length=64)
        if cls.batch is None:
            raise unittest.SkipTest("HumanML3D val batch unavailable")
        if "traj_features" not in cls.batch:
            raise unittest.SkipTest("batch missing traj_features")

    def test_inference_traj_emb_from_traj_features(self):
        from models.tools.traj_encoder import TrajEncoder
        from utils.traj_batch import build_traj_emb_from_batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b = self.batch
        B = b["traj_features"].shape[0]
        seq_len = int(b["token_length"].max().item())
        traj_enc = TrajEncoder(in_dim=4, hidden_dim=64, out_dim=32).to(device)

        x_in = {
            "traj_features": b["traj_features"].to(device),
            "traj_features_mask": b.get("traj_features_mask"),
        }
        traj_emb = build_traj_emb_from_batch(
            x_in,
            seq_len,
            device,
            traj_enc,
            use_traj_cond=True,
            traj_drop_out=0.0,
            training_dropout=False,
        )
        self.assertIsNotNone(traj_emb)
        self.assertEqual(traj_emb.shape[:2], (B, seq_len))
        self.assertTrue(torch.isfinite(traj_emb).all())

    def test_stream_style_xyz_slice_encoding(self):
        """对齐 stream 里 ``xyz_traj_to_features_4d(traj_slice)`` + ``traj_encoder``。"""
        from models.tools.traj_encoder import TrajEncoder
        from utils.traj_batch import xyz_traj_to_features_4d

        b = self.batch
        if "traj" not in b:
            self.skipTest("batch has no traj (stream buffer path uses xyz)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B = b["traj"].shape[0]
        seq_len = int(b["token_length"].max().item())
        traj = b["traj"].to(device=device, dtype=torch.float32)

        traj_enc = TrajEncoder(in_dim=4, hidden_dim=64, out_dim=32).to(device)
        # 模拟 stream 窗口已满 seq_len：取每条样本前 seq_len 帧（不足则左侧 pad，与 step 中逻辑同型）
        traj_slice = traj[:, :seq_len]
        if traj_slice.size(1) < seq_len:
            pad = seq_len - traj_slice.size(1)
            traj_slice = torch.cat(
                [
                    traj_slice.new_zeros(B, pad, 3),
                    traj_slice,
                ],
                dim=1,
            )
        feats = xyz_traj_to_features_4d(traj_slice)
        emb = traj_enc(feats)
        self.assertEqual(emb.shape, (B, seq_len, 32))
        self.assertTrue(torch.isfinite(emb).all())


if __name__ == "__main__":
    unittest.main()
