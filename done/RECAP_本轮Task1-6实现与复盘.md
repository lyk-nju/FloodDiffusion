# Floodcontrol 本轮 Task1-6 实现与复盘

> 这份文档汇总了本轮对 Task1-6 的实际落地内容、设计取舍、测试覆盖与已知不足，作为后续迭代基线。

---

## 1. 本轮目标与范围

- 将 Task1 数据校验从脚本迁入标准 `tests/`
- 完成 Task2/Task3 的“更贴近训练流程”的数据集集成测试
- 完成 Task4 的 FlexTraj 掩码实现与 flash/sdpa 双路径
- 完成 Task5 控制损失的抽象与单测覆盖
- 完成 Task6 关键配置键落地（含兼容别名）

---

## 2. 主要代码改动

### 2.1 Task1：数据与校验

- `datasets/verify_traj_batch.py`
  - 改为迁移提示入口（提示使用 `tests/test_task1_traj_batch.py`）。
- `tests/test_task1_traj_batch.py`
  - 接管原 Task1 校验逻辑，覆盖：
    - `feature_length == traj_length`
    - `token_length` 与 `feature_length` 的 VAE 4x 关系
    - `traj_features_length == token_length`
    - `traj_mask` 维度合法
  - 增加 `test_print_token_and_traj_masks`：
    - 打印 token 级 `traj_features_mask` 与 frame 级 `traj_mask`
    - 断言前段满足 `traj_mask == repeat(traj_features_mask, 4)`（考虑 pad/截断）。
- `tests/humanml3d_fixtures.py`
  - 新增通用 batch 夹具，统一 paths + ldf 配置合并与 val batch 获取逻辑。

### 2.2 Task2：模型结构与测试侧验证补强

- `tests/test_task2_flextraj.py`
  - 保留合成数据用例。
  - 新增 `TestWanForwardHumanML3DBatch`：
    - 真实 HumanML3D batch（`traj_features + token`）构建 `traj_emb`
    - 跑 `WanModel.forward(..., traj_emb=...)`，验证形状与数值稳定性。

### 2.3 Task3：推理流式路径一致性（tiny/full）

- `models/diffusion_forcing_wan.py`
- `models/diffusion_forcing_wan_tiny.py`
  - 流式路径新增/统一：
    - `traj_buffer`（xyz）
    - `traj_features_buffer` / `traj_features_mask_buffer`
    - `_stream_update_traj_buffers(...)`
    - `_stream_build_traj_emb(...)`
    - `_traj_stream_version` + `_traj_emb_cache`
  - 支持两类输入来源（`traj_features` 优先，缺失时回退 `traj`）。
  - 维护窗口滚动时轨迹 buffer 同步滚动并失效缓存。
  - `stream_generate_step` 不再只依赖单次输入 `x`，而是对齐到“流式窗口缓冲”语义。

> 说明：当前缓存的是 `traj_emb` 中间结果（编码侧），不是 Wan block 内部 K/V。

### 2.4 Task4：掩码与注意力后端

- `models/tools/attention.py`
  - 新增：
    - `flextraj_self_attn_bias`
    - `flextraj_query_valid`
    - `flextraj_sdpa_self_attention`
    - `flextraj_flash_split_self_attention`
    - `flextraj_self_attention`（统一入口，`auto/flash/sdpa`）
  - 支持环境变量：
    - `FLEXTRAJ_ATTN_BACKEND=auto|flash|sdpa`
- `models/tools/wan_model.py`
  - FlexTraj（`latent_pad_len is not None`）走 `flextraj_self_attention`。
  - 非 FlexTraj 路径保留原 `flash_attention`。
- `tests/test_task4_mask.py`
  - 增加 mask 语义、不变量、梯度、flash-vs-sdpa 对齐和可选 micro-bench 覆盖。

### 2.5 Task5：控制损失

- `train_ldf.py`
  - 将 control loss 抽成函数：
    - `CustomLightningModule._compute_control_loss_xz(...)`
  - `CustomLightningModule._step(...)` 调用该函数，保持 xz-only 监督语义。
- `tests/test_task5_control_loss.py`
  - 新增独立单测：
    - `y` 变化不影响 loss
    - mask=0 不计入
    - 按有效帧归一化

### 2.6 Task6：配置与兼容键

- `configs/ldf.yaml`
- `configs/ldf_tiny.yaml`
  - 新增/补齐：
    - `traj_encoder_in_dim`
    - `lora_rank_traj`
    - `traj_aggregate_mode`
    - `use_traj_kv_cache`
- `models/diffusion_forcing_wan.py`
- `models/diffusion_forcing_wan_tiny.py`
  - 构造函数接收并兼容别名键：
    - `traj_encoder_in_dim` 覆盖 `traj_in_dim`
    - `lora_rank_traj` 覆盖 `traj_lora_rank`
  - 加入 deprecation warning，提示优先使用新键。

---

## 3. 测试执行与结果

本轮已执行并通过（节选）：

- `python -m unittest tests.test_task1_traj_batch -v`
- `python -m unittest tests.test_task1_traj_batch.TestHumanML3DTrajBatch.test_print_token_and_traj_masks -v`
- `python tests/test_task2_flextraj.py -v`
- `python tests/test_task3_inference_batch.py -v`
- `python tests/test_task4_mask.py -v`
- `python -m unittest tests.test_task5_control_loss -v`

结果：全部通过；Task4 的基准测试默认按设计可跳过（需手动设置 `FLEXTRAJ_BENCH=1`）。

---

## 3.1 实验细节补充（环境、日志与观测）

### A. 运行环境与数据前提

- 主要运行环境：`/home/yuankai/.conda/envs/flooddiffusion`
- 数据：`raw_data/HumanML3D/val_debug.txt` 可读（本轮多次测试日志显示加载 `97` 样本，`Ignored 3 samples due to length mismatch`）
- 注意：Task2/3 的“数据集集成测试”依赖 HumanML3D 数据；无数据时会 `SkipTest`，不判失败。

### B. 关键命令与观测结果（本轮实跑）

1) **Task2（合成 + 数据集）**

- 命令：`python tests/test_task2_flextraj.py -v`
- 结果：`Ran 6 tests ... OK`
- 观测：
  - CUDA forward 用例通过（FlexTraj 主链可跑）
  - 数据集用例通过（`traj_features -> traj_emb -> WanModel.forward` 形状与数值稳定）

2) **Task3（数据集推理一致性）**

- 命令：`python tests/test_task3_inference_batch.py -v`
- 结果：`Ran 2 tests ... OK`
- 观测：
  - `build_traj_emb_from_batch(..., training_dropout=False)` 路径可用
  - `stream` 同型 `xyz -> features4d -> TrajEncoder` 路径可用

3) **Task4（mask + flash/sdpa）**

- 命令：`python tests/test_task4_mask.py -v`
- 结果：`Ran 9 tests ... OK (skipped=1)`
- 观测：
  - 掩码语义、不变量、梯度、flash-vs-sdpa 对齐均通过
  - bench 用例默认 skip（需设置 `FLEXTRAJ_BENCH=1`）

4) **Task1（mask 可视化确认）**

- 命令：`python -m unittest tests.test_task1_traj_batch.TestHumanML3DTrajBatch.test_print_token_and_traj_masks -v`
- 结果：`Ran 1 test ... OK`
- 观测：
  - 输出了每样本 `traj_features_mask`（token 级）与 `traj_mask`（frame 级）摘要
  - 显式验证 `traj_mask[:compare_len] == repeat(traj_features_mask, 4)[:compare_len]`

5) **Task5（控制损失单测）**

- 命令：`python -m unittest tests.test_task5_control_loss -v`
- 结果：`Ran 3 tests ... OK`
- 覆盖：
  - `y` 维不影响 loss（xz-only）
  - mask=0 帧不计损
  - 按有效帧归一化

### C. Task4 性能实验记录（重要）

- 命令：`FLEXTRAJ_BENCH=1 python tests/test_task4_mask.py TestFlexTrajBench.test_walltime_flash_vs_sdpa -v`
- 一次实测输出（本轮）：
  - `sdpa ≈ 0.550 ms/iter`
  - `flash ≈ 8.556 ms/iter`
  - `sdpa/flash ≈ 0.06x`
- 解释：
  - 当前 flash 路径为“分段 + 按样本循环调用 FA（约 2B 次 launch）”
  - 在中小规模任务上，launch 开销可能显著，导致墙钟不如单次稠密 SDPA
- 因此：
  - bench 默认只打印不强制快慢断言
  - 若需要硬性断言可设 `FLEXTRAJ_BENCH_STRICT=1`（仅建议本地实验）

### D. 复现实验建议

- 固定随机种子（脚本内已多处 `torch.manual_seed` / `seed_everything`）
- Task4 bench 建议同时报告：
  - 单算子墙钟（本测试）
  - 端到端 `stream_generate_step` 每步耗时
  - 峰值显存（`torch.cuda.max_memory_allocated`）
- 对外结论建议基于端到端结果，不仅看 micro-bench。

---

## 4. 关键设计取舍（本轮）

- Task3 先做 `traj_emb` 缓存，而非 block 内 K/V 缓存
  - 原因：侵入面较小、与现有 forward 接口兼容、风险低。
  - 代价：对 attention 主耗时优化有限。
- Task4 flash 路径采取“分段 + 按样本循环”方案
  - 原因：规避当前 varlen 封装在不等长 `q_lens` 下的批量还原问题。
  - 代价：中小规模任务上 kernel launch 开销可能抵消 flash 优势。

---

## 5. 当前不足与下一步建议

### 5.1 模型架构侧

- FlexTraj flash 路径仍有 Python 循环，尚非最终高吞吐实现。
- 真正的“traj KV cache（Wan block 内）”尚未落地，当前是编码侧缓存。

### 5.2 训练侧

- 控制损失主逻辑已单测化，但可继续补：
  - 多样本 batch、不同 `traj_length` 边界值
  - `control_aux` 缺失分支行为（dropout）

### 5.3 推理侧

- `use_traj_kv_cache` 命名与实现语义存在偏差（当前缓存 `traj_emb`）。
- 建议后续将配置名拆分为：
  - `use_traj_emb_cache`（当前实现）
  - `use_traj_kv_cache`（未来实现，需 Wan block 接口改造）

---

## 6. 涉及文件清单（本轮）

- `tests/__init__.py`
- `tests/humanml3d_fixtures.py`
- `tests/test_task1_traj_batch.py`
- `tests/test_task2_flextraj.py`
- `tests/test_task3_inference_batch.py`
- `tests/test_task4_mask.py`
- `tests/test_task5_control_loss.py`
- `datasets/verify_traj_batch.py`
- `models/tools/attention.py`
- `models/tools/wan_model.py`
- `models/diffusion_forcing_wan.py`
- `models/diffusion_forcing_wan_tiny.py`
- `train_ldf.py`
- `configs/ldf.yaml`
- `configs/ldf_tiny.yaml`
- `Todolists/Task1-input.md`
- `Todolists/Task2-model.md`
- `Todolists/Task3-inference.md`
- `Todolists/target.md`
- `done/RECAP_轨迹控制实现复盘.md`（同步更新引用）

---

## 7. 结论

本轮已将 Task1-6 从“设计文档阶段”推进到“可测试、可回归、可维护”的工程状态。  
核心链路（数据→模型→训练→推理）已连通，且关键语义（Task4 mask、Task5 xz-only）已有测试保护。  
后续应优先处理性能型工作（flash 批量化与真 KV cache）以释放推理吞吐。

