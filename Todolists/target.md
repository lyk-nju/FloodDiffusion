# FlexTraj 式轨迹控制：设计规范与代码映射

本文档每条设计都对应到仓库中的**具体模块**；**当前**表示已实现行为，**待实现**表示与目标架构的差异（需在后续提交中补齐）。

---

## 与 Task 文档对应

| 章节 | 文档 | 内容摘要 |
|------|------|----------|
| §1 | [`Task1-input.md`](Task1-input.md) | `traj_features`（$x,z,\phi^{emb}$）、token 对齐、与 GT `traj` 分工 |
| §2 | [`Task2-model.md`](Task2-model.md) | 序列拼接、共享 RoPE、traj-only LoRA、`traj_proj` 迁移 |
| §3 | [`Task3-inference.md`](Task3-inference.md) | 下三角调度、traj KV 缓存、`stream_generate_step` 对齐 |
| §4 | [`Task4-mask.md`](Task4-mask.md) | traj 段不看 latent、与 FlashAttention 对接 |
| §5 | [`Task5-loss.md`](Task5-loss.md) | **xz-only** control loss、`traj_mask`、与 decode 长度 |
| §6 | [`Task6-config.md`](Task6-config.md) | YAML 键、新旧架构互斥、冻结与 LoRA |

---

## 全局一致性约定（避免「类似问题」再出现）

以下约定在全仓库文档与实现中应保持一致；若改其一，须同步改关联 Task 与代码。

1. **条件 vs GT 字段**：**`traj_features`**（或经 `TrajEncoder` 前）= 模型条件，**4 维** $(x,z,\cos\psi,\sin\psi)$；**`traj`** = `(T,3)` 完整根位置，供 **GT 与可视化**，**不**强制等于条件维数。`traj_features_mask` 在 token 级采样，再 4× 扩展为 frame 级 `traj_mask` 并补齐到 `traj_length`。
2. **竖直轴与 loss**：HumanML3D 根轨迹 `(...,3)` 约定为 **索引 0、2 → 地面 $x,z$**，**索引 1 → 竖直高度 $y$**。**Control loss 仅对 $x,z$**（与条件不含 $y$ 一致）；见 `train_ldf.py` 与 `Task5-loss.md`。
3. **时间**：**不**使用显式 `t_emb`；**token 索引 $k$** 即时间，**traj 段与 latent 段共享同一时间索引的 RoPE**（§2）。
4. **可选 $\phi$ 监督**：若增加 yaw 项，须 **单独 loss + 权重**，**不得**与三维位置平方和混写。
5. **命名**：「根绕 **Y 轴** yaw」中的 **Y** 为世界竖直轴；与轨迹向量里 **分量 $y$（索引 1）** 勿混为一谈（前者是旋转轴，后者是位置分量）。

---

## 1. 轨迹输入 `traj_input = [x, z, \phi^{emb}]`

### 规范

- **平面坐标 $(x,z)$**：根在地面投影上的位移，与 HumanML3D 根平移语义一致（与 `recover_root_rot_pos` 得到的 `r_pos` 的 $x,z$ 分量一致）。可选在数据管线中对根轨迹做高平滑（Kimodo 式 Smoothed Root）以抑制骨盆摆动；若平滑，须与 $\psi$ **同一管线重算**。
- **朝向嵌入 $\phi^{emb}$（当前实现）**：$\psi_t$ 为 **xz 平面路径朝向**（根轨迹位移差分单位化，与仅路径控制、流式推理可重构的信息一致），$\phi^{emb}_t=[\cos\psi_t,\sin\psi_t]$。由 `utils/traj_batch.path_heading_features_from_root_xyz` 在帧级计算，再经 `aggregate_to_token_last` 与 token 对齐。**非** `rot_vel` 积分的身体面向；若需面向，须另开数据字段或损失项。
- **与 token 时间对齐**：轨迹各通道在 **VAE 时间下采样后的长度**上与 `token` / `token_length` 一致。对每个 token 步覆盖的帧窗口：采用 **窗口末帧**的 $(x,z,\cos\psi,\sin\psi)$（与现 `humanml3d` 一致）。
- **时间信息（不显式注入 `t_emb`）**：轨迹与 latent **按 token 索引 $k$ 一一对齐**时，**序列位置即时间**；在模型内由 **traj 段与 latent 段共享同一时间索引的 RoPE**（见 §2）提供，避免与 RoPE 重复的标量时间通道。
- **MLP**：对拼接后的轨迹特征 $(x,z,\phi^{emb})$ 做「升维—非线性—降维」，对应 **`models/tools/traj_encoder.py`** 中的 **`TrajEncoder`**（当前为 `Linear(3→64)→GELU→Linear(64→out_dim)`）；目标 **`in_dim=4`**（$2+2$），与下游轨迹 token 维度衔接；可在 **Dataset 侧预计算** 或 **模型入口前** 调用，由实现选定。

### 实现映射

| 条目 | 当前代码 | 待实现 |
|------|----------|--------|
| 从 263D 取根位置 | `datasets/humanml3d.py` → `extract_root_trajectory_263(feature)` → `output["traj"]` `(T,3)`；供 **Task5** 在 **$x,z$** 上与 pred 对齐 | `traj` **保留** 为完整 3D GT |
| 取 $\psi$ / $\phi^{emb}$ | `path_heading_features_from_root_xyz(output["traj"])` → `aggregate_to_token_last` → `traj_features` | 若需身体面向，可另用 `extract_root_xz_phi_features_263` 产第二路特征（需与推理约定一致） |
| Batch 字段 | `collate_fn`：`traj`、`traj_mask`、`traj_length`；`train_ldf.py` 将 `traj*` 拷入 `model_batch` | 增加与 token 对齐的 `traj_features` 与 `traj_features_mask`；**不**传 `t_emb`。其中 `traj_features_mask` 先在 token 级采样，`traj_mask` 由其 4× 扩展并补齐到 `traj_length`；`tests/test_task1_traj_batch.py` 同步校验 |
| 与 `seq_len` 对齐 | `models/diffusion_forcing_wan_tiny.py`（及 `diffusion_forcing_wan.py`）`forward` / `generate`：`F.interpolate` 将 `traj` 对齐到当前 `seq_len`（token 长度） | 若数据集直接产出 token 分辨率，可去掉插值或改为仅在缺失时 pad |
| 轨迹 MLP | `TrajEncoder`：`in_dim=3`，`out_dim=cfg.traj_out_dim` | `in_dim=4`（$x,z,\cos\psi,\sin\psi$）；与下游 **轨迹 token 维度**一致 |

---

## 2. 模型结构：序列拼接、双向注意力、轨迹专用 LoRA

### 规范

- **序列拼接（FlexTraj 思路）**：轨迹不作为 patch 后单步 `concat+Linear` 的旁路，而是与 **latent token 序列**在自注意力中组成 **单条序列**（文本仍走 cross-attention）。
- **RoPE（时间索引）**：在 **1:1 对齐** 下（第 $k$ 个 latent token 与第 $k$ 个 traj token 对应同一时间步），**latent 段与 traj 段对 `rope_apply` 使用相同的时间维索引**（共享同一时间轴的 RoPE 频率），使几何条件与噪声潜变量在时间上对齐；若需区分模态，可另加 **可学习的 type embedding**（latent vs traj），但**不**为 traj 单独使用另一套时间 RoPE 索引。
- **双向自注意力**：在 FloodDiffusion **活动窗口**内，latent 段使用全连接自注意力；与配置 **`causal: false`** 一致。
- **LoRA**：仅在 **轨迹 token 对应的 Q/K/V 计算**上引入可训练低秩项（或对轨迹 hidden 仅加增量），**不**通过「整条序列共享的 `q_proj` 上挂一组 LoRA」改变 latent 段的投影；主干权重冻结或尽量少动。

### 实现映射

| 条目 | 当前代码 | 待实现 |
|------|----------|--------|
| DiT 骨干 | `models/tools/wan_model.py`：`WanModel`，`patch_embedding` → `blocks` → `head`；`rope_apply` 按 `grid_sizes`/序列位置 | 在 `blocks` 前构造 `[latent_tokens ∥ traj_tokens]`；**扩展/重排 RoPE 位置映射**，使 traj 位置 $k$ 与 latent 位置 $k$ **共用同一时间索引**（见上节） |
| 条件注入 | **当前**：`forward` 里 `patch` 后 `torch.cat([x, traj_emb], dim=-1)` → `traj_proj`（`nn.Linear(dim+traj_dim, dim)`），属 MotionStream 式 | **目标**：去掉逐 token `traj_proj` 拼接维，改为 **独立 traj token 序列** + 统一 `dim` |
| 双向 | `configs/ldf_tiny.yaml`：`causal: False`；`WanSelfAttention(..., causal=False)` | 保持不变 |
| LoRA | **无** | 在 `models/tools/wan_model.py` 或 `attention.py` 为 **traj 索引区间**单独建 `q/k/v` 的 LoRA 前向；训练入口与 `freeze_backbone_for_traj` 逻辑对齐（当前冻结除 `traj_proj` 外全部，见下） |
| 冻结策略 | `diffusion_forcing_wan_tiny.py`：`freeze_backbone_for_traj` 时 `traj_proj` + `TrajEncoder` 可训 | 改为「traj token 相关 LoRA + 轨迹编码器」可训，与旧 `traj_proj` 二选一迁移 |

---

## 3. 推理：FloodDiffusion 调度与轨迹 KV 缓存

### 规范

- **下三角噪声调度 / 活动窗口**：与 FloodDiffusion 一致，逐段更新 latent。
- **KV 缓存**：仅缓存 **轨迹 token 子序列**的 Key/Value；**每个扩散步**仍须重算 **latent token** 的 Q/K/V。

### 实现映射

| 条目 | 当前代码 | 待实现 |
|------|----------|--------|
| 噪声水平 | `models/diffusion_forcing_wan_tiny.py`：`_get_noise_levels`、`add_noise` | 不变 |
| 训练窗口 MSE | 同上 `forward`：仅末尾 `chunk_size` 步 | 不变 |
| 整段生成 | `generate` / `stream_generate`：ODE 步进 + `start_index:end_index`；传入 `traj_emb` 至 `WanModel` | 改为 traj token 序列 + mask；在 attention 内缓存 traj 的 k/v |
| 逐步流式 | `stream_generate_step`（**tiny**） | **当前未传 `traj_emb`**（`model(..., y=None)` 无轨迹），与 `generate` 不一致；待与 `diffusion_forcing_wan.py` 中带 `traj_buffer` 的逻辑对齐或统一补全 |

---

## 4. 掩码（自注意力）

### 规范

- **轨迹 token**：仅在 **轨迹段内部**做自注意力（或 traj 互看），**不** attend 到 latent、**不** attend 到文本序列（文本不在 self-attn 里）。
- **Latent token**：可 attend **自身段 + 轨迹段**；**文本**仍通过 **`WanCrossAttention`** 注入（`context` / `encode_text_with_cache`），不强行并入 self-attn 掩码矩阵。

### 实现映射

| 条目 | 当前代码 | 待实现 |
|------|----------|--------|
| Cross-attn 文本 | `WanModel.forward`：`WanAttentionBlock` 内 cross | 保持 |
| Latent/traj 分段 mask | **无**（全体 token 同一 self-attn） | `models/tools/attention.py` 中 `flash_attention` 或 block 传入 **attn_bias / block_mask**，实现 FlexTraj/EasyControl 式「traj 不看 latent」 |

---

## 5. 显式控制损失 `Loss_control`

### 规范

在 latent 扩散损失之外，将模型给出的 **x0 或等价干净 latent** 经 **VAE decode** 回到 263D，在 **根轨迹的水平分量 $(x,z)$** 上与 GT 做 **L2**（**不**对竖直分量计损，以与条件仅含 $x,z,\phi^{emb}$ 一致）；可按 `traj_mask` 在帧维稀疏加权。实现上 **GT 与 pred** 均来自 `extract_root_trajectory_263(_torch)`，取 **通道索引 `[0, 2]`** 参与误差（**索引 1 为 $y$**，不参与此项）。

### 实现映射

| 条目 | 当前代码 |
|------|----------|
| 辅助输出 | `diffusion_forcing_wan_tiny.py`：`forward` 末尾 `loss_dict["control_aux"]["pred_x0_latent_list"]`（`prediction_type` 为 `vel`/`x0` 且 `use_traj_cond`） |
| 损失计算 | `train_ldf.py`：`CustomLightningModule._step`，`vae.decode`、`extract_root_trajectory_263_torch` → **仅索引 0、2（$x,z$）** 的 L2 与 `traj_mask` 归一化 |
| 权重 | `configs/ldf_tiny.yaml`：`control_loss_weight` |

条件侧为 $[x,z,\phi^{emb}]$ 时，GT 仍用同一提取函数得到 `traj`；**位置监督仅 xz**；若日后增加 **$\phi$ 监督**，须单独一项（见 [`Task5-loss.md`](Task5-loss.md)），与 xz loss 分权重。

---

## 6. 配置入口（当前）

- **`configs/ldf_tiny.yaml`**：`model.params` 中 `use_traj_cond`、`traj_out_dim`、`traj_drop_out`、`control_loss_weight`、`freeze_backbone_for_traj`、`causal`、`chunk_size`、`noise_steps` 等与上述行为直接对应；目标架构新增项（如 `traj_token_dim`、`traj_encoder_in_dim`）见 [`Task6-config.md`](Task6-config.md)。
- **Control loss 轴向**：当前在代码中 **写死 `[0,2]`（只约束根水平面 $x,z$）**，不保留额外配置开关。
