# Task3 — 推理（调度与轨迹 KV 缓存）

对应 `target.md` **§3 推理策略**。

---

## 实现目标

- **保持** FloodDiffusion 的 **下三角噪声调度**与 **活动窗口**更新逻辑（`_get_noise_levels`、`start_index:end_index`、ODE 步进），不改变训练目标语义。
- **新增**：在 **去噪循环**内，对 **固定不变的轨迹 token** 缓存其 **Key/Value**（或等价缓存中间结果），减少每步重复计算；**latent token** 每步仍全量重算 Q/K/V。
- **对齐**：`generate` / `stream_generate` / `stream_generate_step` 行为一致，**均需**传入与训练同分布的轨迹条件（修复 tiny 版 `stream_generate_step` 未传轨迹的问题）；轨迹须为 **Task1** 约定的 **token 分辨率** 与 **$x,z,\phi$ 语义**（或与旧 `traj` 插值路径一致直至迁移完成）。

---

## 实现步骤

1. **梳理调用链**
   - `diffusion_forcing_wan_tiny.py`：`generate`、`stream_generate`、`stream_generate_step`。
   - `diffusion_forcing_wan.py`：`traj_buffer` + `stream_generate_step` 已有部分流式轨迹逻辑，评估 **合并 tiny 与 full** 减少分叉。

2. **在 `WanModel` 或 `WanAttentionBlock` 增加可选 `cache_traj_kv`**
   - 前向参数：`traj_segment_slice` 或 `(n_latent, n_traj)`。
   - 第一步（或 `t` 第一步）：计算 traj 段的 `k,v` 并缓存；后续步若 `traj_tokens` 未变，复用缓存。
   - 注意：**多步扩散中** `x` 整体变，但 traj 段输入若不变，仅 latent 段的 Q/K/V 更新；实现时要分清 **从哪一维切分**。

3. **与 FlashAttention 对接**
   - 若使用融合 kernel，可能需 **分两次 attention**（latent-latent + latent→traj）或支持 **prefix KV cache** 的 API；提前查 `flash_attention` 是否支持 `past_kv`。
   - 若短期不支持：可仅在 **非 flash** 路径或 **reference attention** 实现缓存，用配置开关切换。

4. **修正 `stream_generate_step`（tiny）**
   - 与 `generate` 一致：维护 `traj_buffer` 或每步传入 `traj`，调用 `WanModel(..., traj_tokens=...)`。
   - `seq_len`、`end_index` 与轨迹切片长度一致，避免越界。

5. **`web_demo/model_manager.py`**
   - 流式生成路径使用统一 API；轨迹长度与 token 步对齐（依赖 Task1）。

6. **数据一致性测试**：`tests/test_task3_inference_batch.py` 使用 HumanML3D val batch，校验 `build_traj_emb_from_batch(..., training_dropout=False)` 与 `stream_generate_step` 同型的 xyz 切片 → `TrajEncoder`（不拉 T5 / 整图 DiffForcing）。

7. **性能与正确性验证**
   - 对比 **开/关 KV 缓存** 的生成结果（数值容差内应一致）；记录每步耗时。

---

## 可能遇到的细节问题

| 问题 | 说明与建议 |
|------|------------|
| **缓存失效条件** | 用户中途改轨迹、或 `traj_drop` 仅训练时有；推理时轨迹固定，但若 **动态长度** 变，需 invalidate 缓存。 |
| **CFG 双前向** | 有条件/无条件两次 `model` 调用；traj 通常两次相同，**缓存可共享**（注意线程安全仅单会话推理）。 |
| **变长 batch** | `List[Tensor]` 逐样本长度不同；KV 缓存需 **per-sample** 或 pad 到 `seq_len` 与 mask 一致。 |
| **EMA 权重** | 缓存逻辑不参与梯度，但推理应使用 **EMA copy** 的权重（与现有 `update_metrics` 一致）。 |
| **数值漂移** | 分块 attention 与合并 KV 的浮点顺序可能微差；以误差阈值验收。 |
| **tiny / full 行为不一致** | 优先 **抽象共用函数** `_build_model_inputs(...)`，减少只修一端导致的 demo 与训练脱节。 |
| **demo / 用户只画 xz** | `web_demo` 若只有平面轨迹，$\phi$ 需默认（如沿用上一帧或 0）并与训练缺省策略一致，否则分布偏移。 |
