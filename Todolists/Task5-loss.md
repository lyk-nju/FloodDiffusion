# Task5 — 显式控制损失 `Loss_control`

对应 `target.md` **§5 显式控制损失**。

---

## 实现目标

- 在 **`prediction_type`** 为 `vel` 或 `x0` 且开启轨迹条件时，继续从模型辅助输出得到 **pred 干净 latent**，经 **VAE decode** 到 263D，在 **根轨迹的水平分量 $(x,z)$** 上与 GT 计算 **加权 L2**（**不**对 $y$ 计损：条件仅约束地面投影，高度由身体姿态与 VAE 主损失决定）。
- **`traj_mask`** 仅在被监督帧上累计误差；归一化方式与现有 `train_ldf.py` 一致（按 mask 有效 **帧**数平均；每帧误差为 **两维 $x,z$** 之和）。**已实现**：`train_ldf.py` 中 `pred_traj[..., [0, 2]]` 与 `gt_traj[..., [0, 2]]`。
- 若 Task1 将条件改为 **token 分辨率**，GT 监督仍应对齐 **decode 后的动作长度**：与 `traj_length`、`feature_length` 关系要明确，避免 off-by-one。

---

## 实现步骤

1. **确认模型侧契约**
   - `diffusion_forcing_wan_tiny.py`（及 full）：`loss_dict["control_aux"]["pred_x0_latent_list"]` 每个元素 `(T_token, 4)` 与 batch 中 token 对齐。
   - 若 head 只回传 latent 段，确认 **无 traj 段混入**（Task2）。

2. **`train_ldf.py` — `CustomLightningModule._step`**
   - 保持：`vae.decode(pred_latent.unsqueeze(0))` → `extract_root_trajectory_263_torch` → 与 `batch["traj"]` 在 **维度 0、2（$x,z$）** 上 L2；**实现**：`pred_traj[..., [0,2]]` 与 `gt_traj[..., [0,2]]` 逐差平方再求和（与 `extract_root_trajectory_263` 的轴约定一致：中间维为竖直高度）。
   - 若 GT `traj` 仍为 **动作长度** `(T,3)`：在 token 级 pred 解码后，动作长度约为 `T_token * 4`（或 VAE 实际倍率），需在 **同一尺度** 上比较：
     - **方案 A**：将 pred 根轨迹 **下采样/池化**到与 `batch["traj"]` 的 `L` 一致；
     - **方案 B**：将 GT `traj` **上采样/重复**到 decode 长度（不推荐，易模糊）；
     - **方案 C**：Task1 同时提供 **动作级 GT 轨迹** 与 **token 级条件**，control loss 只用动作级 GT（推荐与现实现一致）。

3. **`traj_mask`**
   - 若 mask 在动作帧：在比较前将 pred 轨迹在帧维对齐到 `L`，再应用 mask（当前逻辑）。
   - 若改为 token 级 mask：需新字段 `traj_mask_token`，并在 decode 后映射到帧 mask。

4. **$\phi$ 监督（可选扩展）**
   - 若需与 **$\phi^{emb}$** 显式对齐，可另加 **yaw L2** 项并单独设权重；与 **xz-only** 根位置 loss 并列，勿混在三维位置平方和里。

5. **权重与日志**
   - `control_loss_weight`（`ldf_tiny.yaml`）；`wandb` 单独 log `control` 与 `mse`。

6. **梯度检查**
   - `freeze_backbone_for_traj` 时确保 **pred_x0 路径** 仍对 **可训练参数** 有梯度（TrajEncoder / LoRA / traj_in_proj）。

---

## 可能遇到的细节问题

| 问题 | 说明与建议 |
|------|------------|
| **长度 min 裁剪** | 现有代码 `L = min(L_motion, L_gt, traj_length)`；VAE decode 长度与 token 数 * 因子 必须 **文档化**（`wan_vae_1d` 下采样率）。 |
| **bf16 / fp32** | decode 与轨迹提取用 `float()` 与设备一致；避免 control loss 半精度不稳定。 |
| **`traj_drop` 训练** | dropout 时无 `traj_emb` 可能无 `control_aux`；需分支不计算 control loss 或与主损失一致。 |
| **pred 为 vel** | `pred_x0 = pred + noise` 与当前一致；若改 `noise` 预测需改公式。 |
| **验证集可视化** | `make_composite_compare_videos` 使用的 `traj_mask` 路径需与新的 mask 分辨率一致。 |
| **与 Task1 条件字段分离** | 条件用 `traj_features`，GT 仍用 `traj` `(T,3)`，可减少一处长度地狱。 |
| **轴约定** | 若将来根轨迹表示改为非标准 `xyz` 顺序，须同步修改 `[..., [0,2]]` 索引；当前与 `extract_root_trajectory_263_torch` 输出 `(..., 3)` 一致。 |
| **仅 xz 后梯度** | $y$ 不参与 control loss，根高度仍可通过 **主扩散 MSE** 与 VAE 学习；若发现足滑与高度不一致，可再调主损失或接触约束（超出本 Task）。 |
