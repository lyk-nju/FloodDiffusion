# Task4 — 自注意力掩码

对应 `target.md` **§4 掩码设计**。

**实现状态（代码已落地）**

- `models/tools/attention.py`：
  - **默认（`FLEXTRAJ_ATTN_BACKEND=auto`）**：在 `causal=False`、CUDA、已安装 flash-attn 时，FlexTraj 走 **`flextraj_flash_split_self_attention`**（两次 flash：latent→拼接 K、traj→仅 traj K；按 batch 逐条调用以兼容不等长 `q_lens`）；否则走 **`flextraj_sdpa_self_attention`**（稠密 SDPA + 加性 bias）。
  - 环境变量 **`FLEXTRAJ_ATTN_BACKEND`**：`auto` | `flash` | `sdpa`（强制 SDPA 便于对齐/调试；`flash` 条件不满足会报错）。
  - `flextraj_self_attn_bias`、`flextraj_query_valid`：掩码与有效 query 行定义。
- `models/tools/wan_model.py`：`WanSelfAttention` 在 `latent_pad_len is not None` 时调用 **`flextraj_self_attention`**；`window_size != (-1,-1)` 与 FlexTraj 组合会 `NotImplementedError`。
- `tests/test_task4_mask.py`：掩码、梯度、CUDA 整网冒烟、**flash 与 SDPA 数值对齐**（有 flash 时）。

---

## 实现目标

- 在 **拼接后的 self-attention 序列**上施加 **块稀疏掩码**：**轨迹 token 只能 attend 轨迹段内部**（traj→traj），**不能** attend latent 段；**latent token** 可 attend **latent + traj**。
- **文本** 仍仅通过 **`WanCrossAttention`** 进入，**不**写入 self-attn 的 $L\times L$ 掩码（与 `target.md` 一致）。
- 掩码与 Task3 的 **轨迹 KV 缓存** 兼容：traj 段 K/V 在「traj 不看 latent」的前提下可预计算并复用。

---

## 实现步骤

1. **形式化掩码**
   - 设 token 索引 $0..L_l-1$ 为 latent，$L_l..L_l+L_t-1$ 为 traj。
   - 对任意 query 位置 $i$、key 位置 $j$：若 $i \ge L_l$ 且 $j < L_l$，权重为 $-\infty$（或极大负 bias）；其余按双向允许。
   - 可选：traj 内部是否 **全连接** 或 **因果**（FlexTraj/EasyControl 为条件自洽，一般为 traj 段内全连接）。

2. **`models/tools/attention.py`**
   - 阅读 `flash_attention` 的 `window_size`、`causal`、是否支持 **attn_bias**（4D 或 (B,1,L,L)）。
   - 若不支持任意 mask：实现 **fallback**：`scaled_dot_product_attention` + 显式 mask（仅 debug 或小模型）；或拆成 **两次 attention**（latent 更新用 concat([Q_lat, Q_lat],[K_lat,K_traj],[V_lat,V_traj])）。

3. **`WanAttentionBlock` / `WanModel`**
   - 传入 `segment_info: (n_latent, n_traj)` 或 `attn_mask`。
   - 确保 **每层 block** 使用同一掩码（除非某层特意不同，一般不推荐）。

4. **与 RoPE 结合**
   - mask 只作用在 attention logits；RoPE 仍按位置施加在 Q/K 上，**勿**把 segment 混用错误索引；**时间索引与 `target.md` §2「latent/traj 共享同一时间轴 RoPE」一致**，掩码不改变 RoPE 规则。

5. **测试用例**
   - 构造 $L_l=2,L_t=2$ 手工张量，检查梯度是否仅从允许边传播；traj query 对 latent key 的 logit 应为 $-\infty$。

---

## 可能遇到的细节问题

| 问题 | 说明与建议 |
|------|------------|
| **FlashAttention 限制** | 部分版本仅支持因果或局部窗口；任意二值 mask 可能需 **xformers** / **PyTorch SDPA** / 自定义 CUDA。 |
| **性能** | 显式 $L^2$ mask 在 $L$ 大时显存与算力上升；优先 **融合 kernel** 或 **分块方案**。 |
| **padding 位置** | `WanModel` 将序列 pad 到 `seq_len`；掩码必须对 **pad token** 同时屏蔽为不 attend、不被 attend（或与现有 `seq_lens` 逻辑统一）。 |
| **与 cross-attn 混淆** | cross-attn 的 `context` 长度与 self-attn 的 $L$ 无关；不要误把 text token 拼进 self-attn 再掩码。 |
| **仅 latent 出 head** | 掩码不改变 head 切片；需保证 **traj 段不参与 loss**（Task2），但可参与中间层表示。 |
| **推理与训练掩码一致** | 任何 `stream_generate_step` 的长度截断必须与训练时 **同一规则**，否则 traj/latent 边界错位。 |
