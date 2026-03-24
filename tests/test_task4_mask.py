"""
Task4：FlexTraj self-attn **块稀疏掩码**（traj 段 query 不可 attend latent key）及 SDPA / Flash 双实现一致性。

**测试内容**
- ``TestFlexTrajAttnBias``：读 ``flextraj_self_attn_bias`` 生成的加性 bias——traj 行对 latent 列为强负；
  padding latent 列对全体 query 关闭。
- ``TestFlexTrajMaskEffect``：**函数级 mask**——扰动 latent 段 ``V`` 后 traj 有效行输出不变；手算一行 softmax，
  traj query 对 latent key 列质量≈0。
- ``TestFlexTrajFlashVsSdpa``：CUDA + flash-attn 可用时，``flextraj_flash_split_self_attention`` 与
  ``flextraj_sdpa_self_attention`` 数值接近；并用真实扰动 ``V`` 验证 flash 路径同样满足 traj 段不变量。
- ``TestFlexTrajSdpaGrad``：loss 只反传 traj 行输出时，**latent 段 K** 无梯度（无 traj→latent 边）。
- ``TestFlexTrajBench``（``FLEXTRAJ_BENCH=1``）：打印稠密 SDPA vs 双路 flash 的 **墙钟**；默认不断言快慢；
  ``FLEXTRAJ_BENCH_STRICT=1`` 才断言 ``t_flash < t_sdpa``。
- ``TestWanForwardWithMask``：小 ``WanModel`` + ``traj_emb`` 在 CUDA 上整网 forward 冒烟（走 ``flextraj_self_attention``）。

**逻辑**
- 掩码语义与 ``models/tools/attention.py``、``WanSelfAttention`` 中 Task4 设计一致；RoPE/文本 cross-attn 不在此文件单独测。
- Flash 路径为「latent→拼接 K」「traj→仅 traj K」两次 FA + 按 batch 循环（见 ``attention.py`` 注释）。

**运行**（项目根）::

    python tests/test_task4_mask.py -v
    FLEXTRAJ_BENCH=1 python tests/test_task4_mask.py TestFlexTrajBench.test_walltime_flash_vs_sdpa -v

---

**如何对外说明「双路 flash vs 稠密 SDPA」**

1. **复杂度**：稠密 SDPA 带 ``(B,1,L,L)`` bias 约为 **O(B·H·L²·D)**；双路 flash 在有效 ``vl,vt`` 上无完整 ``L×L`` 物化。
2. **实证**：微基准仅作参考；当前实现 **2B 次** FA launch，中小规模可能慢于单次大块 SDPA；报告应用 **全 step 时间 / 峰值显存**。
"""

import os
import sys
import time
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestFlexTrajAttnBias(unittest.TestCase):
    def test_traj_query_latent_key_is_masked(self):
        from models.tools.attention import flextraj_self_attn_bias

        # L_l=2, L_t=2, 全长 4，全部有效
        n_latent = 2
        L = 4
        seq_lens = torch.tensor([4], dtype=torch.long)
        bias, qv = flextraj_self_attn_bias(
            seq_lens, n_latent, L, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(bias.shape, (1, 1, L, L))
        neg = torch.finfo(torch.float32).min / 4
        # query 行 3 属于 traj 段，key 列 0 为 latent → 必须屏蔽
        self.assertLess(bias[0, 0, 3, 0].item(), neg / 2)
        # latent query 行 0 可看 latent key 0
        self.assertEqual(bias[0, 0, 0, 0].item(), 0.0)

    def test_padding_keys_masked(self):
        from models.tools.attention import flextraj_self_attn_bias

        n_latent, L = 2, 4
        seq_lens = torch.tensor([2], dtype=torch.long)  # 每段仅 1 帧有效 → vl=vt=1
        bias, _ = flextraj_self_attn_bias(
            seq_lens, n_latent, L, device=torch.device("cpu"), dtype=torch.float32
        )
        neg = torch.finfo(torch.float32).min / 4
        # latent pad 列 j=1：应对所有 query 关闭
        self.assertLess(bias[0, 0, 0, 1].item(), neg / 2)


class TestFlexTrajMaskEffect(unittest.TestCase):
    """用输出不变量刻画 mask：traj 段不能从 latent key 读到 V。"""

    def test_traj_output_invariant_when_latent_v_changes(self):
        from models.tools.attention import flextraj_sdpa_self_attention

        torch.manual_seed(1)
        B, L, H, D = 1, 4, 2, 8
        n_latent = 2
        seq_lens = torch.tensor([4], dtype=torch.long)
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D)
        v_base = torch.randn(B, L, H, D)
        v_perturb = v_base.clone()
        v_perturb[:, :n_latent] = v_perturb[:, :n_latent] + 50.0

        o0 = flextraj_sdpa_self_attention(
            q, k, v_base, seq_lens=seq_lens, n_latent=n_latent, causal=False
        )
        o1 = flextraj_sdpa_self_attention(
            q, k, v_perturb, seq_lens=seq_lens, n_latent=n_latent, causal=False
        )
        # traj 段（含 pad 槽位）与 latent 段 pad 行已由 query_valid 置零；比较 traj 有效行应完全一致
        torch.testing.assert_close(
            o0[:, n_latent : n_latent + 2],
            o1[:, n_latent : n_latent + 2],
            rtol=1e-5,
            atol=1e-5,
        )
        # latent 有效行应随 V 变化
        self.assertGreater((o0[:, 0:1] - o1[:, 0:1]).abs().max().item(), 10.0)

    def test_softmax_mass_traj_row_on_latent_keys_near_zero(self):
        """手算一行 attention：traj query 对 latent key 列 softmax 质量应≈0。"""
        from models.tools.attention import flextraj_self_attn_bias

        torch.manual_seed(2)
        n_latent, L = 2, 4
        seq_lens = torch.tensor([4], dtype=torch.long)
        H, D = 2, 8
        scale = D**-0.5
        q = torch.randn(1, L, H, D)
        k = torch.randn(1, L, H, D)

        bias, _ = flextraj_self_attn_bias(
            seq_lens,
            n_latent,
            L,
            device=q.device,
            dtype=torch.float32,
            causal=False,
        )
        # traj 上第一个有效 query 行 = n_latent
        qi = n_latent
        scores = torch.einsum("bhd,bjhd->bhj", q[:, qi], k) * scale
        scores = scores + bias[:, 0, qi, :].view(1, 1, L)
        w = torch.softmax(scores, dim=-1)
        mass_latent = w[:, :, :n_latent].sum(dim=-1)
        self.assertLess(mass_latent.max().item(), 1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "flash split path needs CUDA")
class TestFlexTrajFlashVsSdpa(unittest.TestCase):
    def test_flash_split_matches_sdpa(self):
        from models.tools.attention import (
            _flextraj_flash_available,
            flextraj_flash_split_self_attention,
            flextraj_sdpa_self_attention,
        )

        if not _flextraj_flash_available():
            self.skipTest("flash-attn not installed")

        torch.manual_seed(0)
        device = torch.device("cuda")
        B, L, H, D = 2, 8, 2, 16
        n_latent = 4
        seq_lens = torch.tensor([6, 8], device=device, dtype=torch.long)
        q = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
        k = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
        v = torch.randn(B, L, H, D, device=device, dtype=torch.float32)

        o_sdpa = flextraj_sdpa_self_attention(
            q, k, v, seq_lens=seq_lens, n_latent=n_latent, causal=False
        )
        o_flash = flextraj_flash_split_self_attention(
            q, k, v, seq_lens=seq_lens, n_latent=n_latent
        )
        torch.testing.assert_close(o_sdpa, o_flash, rtol=2e-2, atol=2e-2)

    def test_mask_invariant_flash_matches_sdpa(self):
        """与 CPU 不变量一致：在 CUDA 上 flash 路径下 traj 段也不应受 latent V 扰动影响。"""
        from models.tools.attention import (
            _flextraj_flash_available,
            flextraj_flash_split_self_attention,
        )

        if not _flextraj_flash_available():
            self.skipTest("flash-attn not installed")

        torch.manual_seed(3)
        device = torch.device("cuda")
        B, L, H, D = 1, 8, 2, 16
        n_latent = 4
        seq_lens = torch.tensor([8], device=device, dtype=torch.long)
        q = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
        k = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
        vb = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
        vp = vb.clone()
        vp[:, :n_latent] += 80.0

        o0 = flextraj_flash_split_self_attention(
            q, k, vb, seq_lens=seq_lens, n_latent=n_latent
        )
        o1 = flextraj_flash_split_self_attention(
            q, k, vp, seq_lens=seq_lens, n_latent=n_latent
        )
        vt = int((seq_lens[0] // 2).item())
        torch.testing.assert_close(
            o0[:, n_latent : n_latent + vt],
            o1[:, n_latent : n_latent + vt],
            rtol=1e-4,
            atol=1e-4,
        )


class TestFlexTrajSdpaGrad(unittest.TestCase):
    def test_traj_output_no_grad_through_latent_keys(self):
        from models.tools.attention import flextraj_sdpa_self_attention

        torch.manual_seed(0)
        B, L, H, D = 1, 4, 2, 8
        n_latent = 2
        seq_lens = torch.tensor([4], dtype=torch.long)
        q = torch.randn(B, L, H, D, requires_grad=True)
        k = torch.randn(B, L, H, D, requires_grad=True)
        v = torch.randn(B, L, H, D, requires_grad=True)

        out = flextraj_sdpa_self_attention(
            q, k, v, seq_lens=seq_lens, n_latent=n_latent, causal=False
        )
        # 仅 traj 段最后一个有效 query 的输出
        loss = out[:, 3, :, :].sum()
        loss.backward()

        # 不应通过 traj→latent 边传到 latent key（列 0,1）
        self.assertLess(k.grad[:, :n_latent].abs().max().item(), 1e-4)
        # latent key 仍可从 latent query 路径收到梯度（另一路）
        self.assertGreater(q.grad.abs().sum().item(), 0.0)


@unittest.skipUnless(
    os.environ.get("FLEXTRAJ_BENCH") == "1" and torch.cuda.is_available(),
    "set FLEXTRAJ_BENCH=1 and use CUDA to run wall-time microbenchmark",
)
class TestFlexTrajBench(unittest.TestCase):
    """墙钟对比：默认只打印，不判胜负（见模块说明）。"""

    def test_walltime_flash_vs_sdpa(self):
        from models.tools.attention import (
            _flextraj_flash_available,
            flextraj_flash_split_self_attention,
            flextraj_sdpa_self_attention,
        )

        if not _flextraj_flash_available():
            self.skipTest("flash-attn not installed")

        device = torch.device("cuda")
        torch.manual_seed(0)
        # 较大 L 时稠密 SDPA 劣势更明显
        n_latent = 128
        L = 2 * n_latent
        B, H, D = 4, 8, 64
        seq_lens = torch.full((B,), L, device=device, dtype=torch.long)
        q = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
        k = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
        v = torch.randn(B, L, H, D, device=device, dtype=torch.float32)

        def run_sdpa():
            flextraj_sdpa_self_attention(
                q, k, v, seq_lens=seq_lens, n_latent=n_latent, causal=False
            )

        def run_flash():
            flextraj_flash_split_self_attention(
                q, k, v, seq_lens=seq_lens, n_latent=n_latent
            )

        warmup, repeats = 5, 25
        for _ in range(warmup):
            run_sdpa()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            run_sdpa()
        torch.cuda.synchronize()
        t_sdpa = (time.perf_counter() - t0) / repeats

        for _ in range(warmup):
            run_flash()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            run_flash()
        torch.cuda.synchronize()
        t_flash = (time.perf_counter() - t0) / repeats

        ratio = t_sdpa / max(t_flash, 1e-9)
        print(
            f"\n[FLEXTRAJ_BENCH] sdpa {t_sdpa*1000:.3f} ms/iter, "
            f"flash {t_flash*1000:.3f} ms/iter, sdpa/flash = {ratio:.2f}x "
            f"(>1 表示 sdpa 更慢)。当前 flash 实现为按 batch 循环 FA，launch 多，"
            f"小任务上 sdpa 更快属正常现象；写报告请结合全 step 与显存。\n"
        )
        if os.environ.get("FLEXTRAJ_BENCH_STRICT") == "1":
            self.assertLess(
                t_flash,
                t_sdpa,
                msg=(
                    f"STRICT: expected t_flash < t_sdpa, got "
                    f"sdpa {t_sdpa*1e3:.3f} ms vs flash {t_flash*1e3:.3f} ms"
                ),
            )


@unittest.skipUnless(torch.cuda.is_available(), "Wan FlexTraj forward uses CUDA in project tests")
class TestWanForwardWithMask(unittest.TestCase):
    def test_traj_concat_forward_still_runs(self):
        from models.tools.wan_model import WanModel

        device = torch.device("cuda")
        B, seq_len, C = 1, 8, 4
        m = WanModel(
            patch_size=(1, 1, 1),
            in_dim=C,
            dim=64,
            ffn_dim=128,
            text_dim=32,
            out_dim=C,
            num_heads=4,
            num_layers=1,
            text_len=16,
            traj_enc_dim=8,
            traj_lora_rank=0,
        ).to(device=device, dtype=torch.float32)

        x_list = [
            torch.randn(C, seq_len, 1, 1, device=device, dtype=torch.float32)
        ]
        t = torch.rand(B, seq_len, device=device, dtype=torch.float32)
        context = [torch.randn(4, 32, device=device, dtype=torch.float32)]
        traj_emb = torch.zeros(B, seq_len, 8, device=device, dtype=torch.float32)
        out = m(x_list, t, context, seq_len, traj_emb=traj_emb)
        self.assertEqual(out[0].shape[1], seq_len)


if __name__ == "__main__":
    unittest.main()
