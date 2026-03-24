"""
Task2：轨迹特征几何、FlexTraj 模块与 Wan 前向（合成 + 真实数据两条线）。

**测试内容**
- ``TestPathHeading``：numpy 路径朝向 ``[x,z,cos,sin]`` 在直线上的数值；torch ``xyz_traj_to_features_4d``
  与 numpy 同输入一致（保证数据侧与模型侧同一语义）。
- ``TestWanModelFlexTraj``：``traj_enc_dim>0`` 时 ``traj_in_proj`` / traj LoRA 等存在；``wan_model_cross_rope``
  仅重导出 ``wan_model.WanModel``（无第二套 backbone）。
- ``TestWanForwardCuda``：**合成** latent 列表 + 零 ``traj_emb``，小 ``WanModel`` 在 CUDA 上整段 forward，检查输出形状。
- ``TestWanForwardHumanML3DBatch``：**val batch** 的 ``token`` pad 成 ``(C,T,1,1)``，``traj_features`` 经
  ``build_traj_emb_from_batch``（与 DiffForcing 一致）得 ``traj_emb``，再 Wan forward；检查有限性与形状。

**逻辑**
- 合成用例不依赖磁盘，适合 CI；数据集用例依赖 ``humanml3d_fixtures.try_get_val_batch``，无数据则 ``SkipTest``。
- 数据集用例限制 ``max_token_length<=64`` 控制序列长度；CUDA 类在无 GPU 时整类 skip。

**运行**（项目根）::

    python -m unittest tests.test_task2_flextraj -v
"""

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


class TestPathHeading(unittest.TestCase):
    def test_path_heading_straight_x(self):
        from utils.traj_batch import path_heading_features_from_root_xyz

        p = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        f = path_heading_features_from_root_xyz(p)
        self.assertEqual(f.shape, (3, 4))
        np.testing.assert_allclose(f[1, 2:], [1.0, 0.0], atol=1e-5)

    def test_xyz_traj_torch_matches_numpy(self):
        from utils.traj_batch import path_heading_features_from_root_xyz, xyz_traj_to_features_4d

        p = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=np.float32)
        fn = path_heading_features_from_root_xyz(p)
        tt = torch.from_numpy(p).unsqueeze(0)
        ft = xyz_traj_to_features_4d(tt)[0].cpu().numpy()
        np.testing.assert_allclose(fn, ft, atol=1e-5)


class TestWanModelFlexTraj(unittest.TestCase):
    def test_traj_branch_modules(self):
        from models.tools.wan_model import WanModel

        m = WanModel(
            patch_size=(1, 1, 1),
            in_dim=4,
            dim=64,
            ffn_dim=128,
            text_dim=32,
            out_dim=4,
            num_heads=4,
            num_layers=2,
            text_len=16,
            traj_enc_dim=16,
            traj_lora_rank=4,
        )
        self.assertIsNotNone(m.traj_in_proj)
        self.assertIsNotNone(m.traj_type_embed)
        blk = m.blocks[0]
        self.assertGreater(blk.self_attn.traj_lora_rank, 0)
        self.assertIsNotNone(blk.self_attn.lora_q_a)

    def test_cross_rope_reexport(self):
        from models.tools import wan_model_cross_rope as cro
        from models.tools import wan_model as wm

        self.assertIs(cro.WanModel, wm.WanModel)


@unittest.skipUnless(torch.cuda.is_available(), "flash_attention requires CUDA")
class TestWanForwardCuda(unittest.TestCase):
    def test_traj_concat_forward(self):
        from models.tools.wan_model import WanModel

        device = torch.device("cuda")
        B, seq_len, C = 2, 8, 4
        m = WanModel(
            patch_size=(1, 1, 1),
            in_dim=C,
            dim=64,
            ffn_dim=128,
            text_dim=32,
            out_dim=C,
            num_heads=4,
            num_layers=2,
            text_len=16,
            traj_enc_dim=8,
            traj_lora_rank=0,
        ).to(device=device, dtype=torch.float32)

        x_list = [
            torch.randn(C, seq_len, 1, 1, device=device, dtype=torch.float32)
            for _ in range(B)
        ]
        # 与 DiffForcing 一致：(B, seq_len) 噪声水平，避免 1D t 在 expand 上的歧义
        t = torch.rand(B, seq_len, device=device, dtype=torch.float32)
        context = [
            torch.randn(4, 32, device=device, dtype=torch.float32) for _ in range(B)
        ]
        traj_emb = torch.zeros(B, seq_len, 8, device=device, dtype=torch.float32)

        out = m(x_list, t, context, seq_len, traj_emb=traj_emb)
        self.assertEqual(len(out), B)
        self.assertEqual(out[0].shape[1], seq_len)


@unittest.skipUnless(torch.cuda.is_available(), "Wan FlexTraj + dataset test needs CUDA")
class TestWanForwardHumanML3DBatch(unittest.TestCase):
    """用 val dataloader 的真实 ``traj_features`` / ``token`` 跑 Wan forward（贴近训练）。"""

    @classmethod
    def setUpClass(cls):
        cls.batch = try_get_val_batch(batch_size=2, max_token_length=64)
        if cls.batch is None:
            raise unittest.SkipTest("HumanML3D val batch unavailable")
        if "traj_features" not in cls.batch or "token" not in cls.batch:
            raise unittest.SkipTest("batch missing traj_features or token")

    def test_wan_forward_with_dataset_traj_emb(self):
        from models.tools.wan_model import WanModel
        from models.tools.traj_encoder import TrajEncoder
        from utils.traj_batch import build_traj_emb_from_batch

        device = torch.device("cuda")
        b = self.batch
        B = b["token"].shape[0]
        C = b["token"].shape[-1]
        seq_len = int(b["token_length"].max().item())
        tok = b["token"].to(device=device, dtype=torch.float32)

        x_list = []
        for i in range(B):
            Ti = int(b["token_length"][i].item())
            t_i = tok[i, :Ti]
            if Ti < seq_len:
                t_i = torch.cat(
                    [t_i, t_i.new_zeros(seq_len - Ti, C)], dim=0
                )
            x_list.append(t_i.transpose(0, 1).reshape(C, seq_len, 1, 1))

        traj_enc_dim = 32
        traj_enc = TrajEncoder(in_dim=4, hidden_dim=64, out_dim=traj_enc_dim).to(device)
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
        self.assertEqual(traj_emb.shape, (B, seq_len, traj_enc_dim))

        m = WanModel(
            patch_size=(1, 1, 1),
            in_dim=C,
            dim=128,
            ffn_dim=256,
            text_dim=32,
            out_dim=C,
            num_heads=4,
            num_layers=2,
            text_len=16,
            traj_enc_dim=traj_enc_dim,
            traj_lora_rank=0,
        ).to(device=device, dtype=torch.float32)

        t = torch.rand(B, seq_len, device=device, dtype=torch.float32)
        context = [
            torch.randn(4, 32, device=device, dtype=torch.float32) for _ in range(B)
        ]
        out = m(x_list, t, context, seq_len, traj_emb=traj_emb)
        self.assertEqual(len(out), B)
        self.assertEqual(out[0].shape[1], seq_len)
        self.assertTrue(torch.isfinite(out[0]).all())


if __name__ == "__main__":
    unittest.main()
