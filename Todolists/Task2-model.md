# Task2 — 模型结构（序列拼接、LoRA）

对应 `target.md` **§2 模型结构**。

**实现记录（代码）**：`WanModel` 已移除 `traj_proj`，改为 `traj_in_proj` + `[latent‖traj]` 序列拼接；`rope_apply_concat_latent_traj` 共享时间 RoPE；`traj_lora_rank>0` 时仅在 traj 段对 Q/K/V 加 LoRA；`Head` 只取 latent 半段。`DiffForcingWanModel`（tiny / full）经 `utils/traj_batch.build_traj_emb_from_batch` 优先用 `traj_features`，否则由 `traj` xyz 经 `xyz_traj_to_features_4d`（路径朝向，与数据集一致）；`train_ldf` 传入 `traj_features*`。`wan_model_cross_rope.py` 已改为对 `wan_model` 的重导出，避免双份 backbone。冒烟测试：`tests/test_task2_flextraj.py`（合成张量 + **HumanML3D val batch** 的 `TestWanForwardHumanML3DBatch`，CUDA 上整段 forward）。

---

## 实现目标

- 将轨迹条件从 **MotionStream 式**（patch 后 `concat(traj_emb)` + `traj_proj`）迁移为 **FlexTraj 式**：**latent token 与轨迹 token 在同一条自注意力序列**中交互；文本仍仅通过 **cross-attention** 注入。
- 在 **轨迹 token 对应的 Q/K/V** 上接入 **LoRA**（或等价「仅 traj 段生效」的低秩增量），**不**让整条序列共享的 `q_proj` LoRA 改变 latent 段的表示。
- 保持 **`causal: false`** 与 FloodDiffusion 活动窗口语义兼容（窗口内双向）。
- **与 Task5**：条件为水平面 + yaw；**control loss** 仅在 GT 根轨迹 **$x,z$** 上计算，与「head 只监督 latent、不监督 traj 段」分工一致（见 `target.md` 全局约定）。

---

## 实现步骤

1. **接口设计（`WanModel.forward`）**
   - 约定输入：除现有 `noisy_input` 列表外，增加 **轨迹 token 序列** `traj_tokens`（形状与 batch 内 `seq_len` 一致），或单张量 `(B, L_traj, D)` 与 latent `(B, L_latent, D)` 分离传入，在模型内 `cat`。
   - 明确 **segment 边界**：`l_latent`、`l_traj`（通常 `l_latent == l_traj == T_token` 或按设计 1:1）。

2. **`patch_embedding` 与轨迹分支**
   - **Latent**：保持 `Conv3d` patch 对 noisy latent。
   - **Traj**：轨迹已是 `dim` 或需 `nn.Linear(traj_enc_dim → dim)` 投到与 patch 后 `dim` 一致；**不再**使用 `traj_proj(dim+traj_dim→dim)` 与 latent 拼在最后一维。
   - 在 **进入 `blocks` 之前** `torch.cat([x_latent, x_traj], dim=1)`，并记录 `n_latent`、`n_traj` 供 mask 与 LoRA 使用（Task4 配合）。

3. **RoPE / 位置编码（与 `target.md` 一致）**
   - **共享同一时间索引**：拼接序列为 `[latent_0…latent_{T-1} ∥ traj_0…traj_{T-1}]` 时，对 RoPE 的 **时间维** 使用映射：`latent_k` 与 `traj_k` 使用 **同一绝对时间索引 $k$**（与 patch 后 latent 的帧下标一致），**不**对 traj 段从 0 另起一套 RoPE 轴。
   - **实现落点**：`models/tools/wan_model.py` 中 `rope_apply` 目前按 `grid_sizes` 与序列下标展开；拼接后需为 **后半 traj 段** 传入与 **前半对应 latent 位置** 相同的 temporal `freqs` 索引（或等价：在 `cat` 前分别为两段赋相同 `t_idx` 再合并进 attention）。
   - **模态区分（可选）**：若需让模型区分「噪声潜变量」与「轨迹条件」，增加 **可学习 type embedding**（加在 patch/traj linear 之后、进 block 之前），**不改变** RoPE 时间索引规则。

4. **LoRA 实现路径（择一）**
   - **A**：`WanSelfAttention` 内对 `q,k,v` 计算后，按 token 索引对 `i >= n_latent` 的行加上 `B@A@h`；或
   - **B**：轨迹段单独一套 `q_proj_lora` / `k_proj_lora` / `v_proj_lora`，仅对 `x_traj` 切片调用。
   - 注册参数名便于 `freeze_backbone_for_traj` 改为：**冻结** `blocks.*` 基座权重，**仅训练** `traj_in_proj`、`lora_*`、`TrajEncoder`。

5. **移除或降级旧路径**
   - 移除（或在代码中强制关闭）旧的 MotionStream 式注入：`traj_proj` 与 `cat([x, traj_emb], dim=-1)` 拼接维路径；本目录仅保留 FlexTraj 式「轨迹 token 与 latent token 组成同一 self-attn 序列」。

6. **`diffusion_forcing_wan_tiny.py`**
   - `forward` / `generate` / `stream_generate`：构造 `traj_tokens` 传入 `WanModel`；删除对旧 `traj_emb` 的逐帧拼维（与配置联动）。

7. **单元测试 / 过拟合单 batch**
   - 见 `tests/test_task2_flextraj.py`：路径朝向与 torch 一致、FlexTraj 模块存在、`wan_model_cross_rope` 重导出；**CUDA** 下可跑 `traj_emb` 全零的短序列 forward（无 LoRA）。若需严格「无 traj≈原模型」，依赖 `traj_in_proj` 零初始化（已实现）。

---

## 可能遇到的细节问题

| 问题 | 说明与建议 |
|------|------------|
| **序列长度翻倍** | self-attn 复杂度 $O((L_l+L_t)^2)$；若 $L_l=L_t=T$，计算量约为原来 4×；可考虑 traj 下采样或共享 traj token（折中）。 |
| **RoPE 与 pad** | `WanModel` 将序列 pad 到 `seq_len`；traj/latent 的 **有效** 时间索引应对齐真实 token 帧，pad 段 RoPE 应与现有 latent pad 规则一致，避免 traj 有效位与 latent 有效位时间错位。 |
| **Head 只输出 latent 段** | `Head` 当前对整序列输出；需 **只取前 `n_latent`** 个 token 做 `out_dim` 预测，避免对 traj 段算重建损失。 |
| **FlashAttention 与 mask** | 自定义 block mask 需与 `flash_attention` API 一致（Task4）；部分实现只支持因果或全 1，需查 `attention.py`。 |
| **LoRA rank 与 lr** | traj 段数据量少时易过拟合；rank 宜小，可与 backbone 不同学习率。 |
| **Checkpoint 兼容** | 旧 ckpt 无 `traj_token` 相关键；`load_state_dict(strict=False)` 与 `on_load_checkpoint` 逻辑已在 `train_ldf.py`，需扩展新参数名。 |
| **`wan_model_cross_rope.py`** | 若项目使用多文件变体，需同步改或统一入口，避免训练与推理 backbone 不一致。 |
