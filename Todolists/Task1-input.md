# Task1 — 轨迹输入（`traj_input`）

对应 `target.md` **§1 轨迹输入**。

---

## 实现目标

- 在 **token 时间分辨率**（与 `token` / `token_length` 一致，相对 263D 约 4× 下采样）上，为每条样本提供 **`traj_input = [x, z, \phi^{emb}]`**（$\phi^{emb}=[\cos\psi,\sin\psi]$，$\psi$ 与 `recover_root_rot_pos` 的 `r_rot_ang` 一致）。
- **不显式构造 `t_emb`**：时间与 token 索引 $k$ 对齐，由模型内 **traj 与 latent 共享同一时间索引的 RoPE**（Task2）表达。
- 数据管线产出字段可被 `collate_fn` 批处理，并与 `train_ldf.py` 传入 `model_batch` 的键一致；校验脚本可断言形状与 `token_length` 对齐。
- **`TrajEncoder`** 输入维 **`in_dim=4`**（$x,z,\cos\psi,\sin\psi$），其中 $\psi$ 为 **xz 路径朝向**（与 `utils/traj_batch.path_heading_features_from_root_xyz` 一致），输出维与下游轨迹 token / `hidden_dim` 衔接（Task2）。**非** 263 `rot_vel` 积分的身体面向。
- **与 Task5 分工**：**模型条件**只用水平面 + 朝向（`traj_features`）；**batch 里仍可保留** `traj` 为 `(T,3)` 的完整根轨迹，供 `train_ldf.py` 里 **control loss** 使用——该损失已改为 **仅对 $x,z$ 维**与预测根轨迹对齐（**不**监督 $y$），与「条件不含高度」一致。

---

## 实现步骤

1. **`utils/motion_process.py`**
   - 新增 `recover_root_yaw_sincos(feature_263)` 或等价函数：在 **与 `recover_root_rot_pos` 相同设备/dtype 约定**下，返回逐帧 $\psi_t$ 或直接的 $[\cos\psi_t,\sin\psi_t]$（避免重复实现 cumsum 逻辑，可内部调用 `recover_root_rot_pos` 并取 `r_rot_ang`）。
   - 新增 `aggregate_root_features_to_token_time(root_xz, phi_emb, motion_length, token_length, mode="last"|"sincos_mean")`：按 VAE 时间因子 4 将 **动作帧** 聚合到 **token 帧**（末帧采样或 cos/sin 向量平均后 L2 归一化），输出形状 `(T_token, ·)`。
   - （可选）平滑根轨迹：对聚合前的 $x,z$ 做 1D 平滑后，**对同一 `feature` 重算** $\psi$ 或基于平滑后的速度重积分，保证与 `target.md` 一致。

2. **`datasets/humanml3d.py`**
   - 在 `_process` 中：在已有 `feature` / `token` 裁剪对齐逻辑下，计算 **token 对齐**的 `traj_xz`、`traj_phi`（或合并为 `traj_features` 四维通道）；**不**生成 `t_emb`。
   - 保留 **`traj` `(T,3)`**（`extract_root_trajectory_263`）：供 **Task5** 与 GT 对比；control loss **只取索引 0、2（$x,z$）**，**索引 1 为竖直 $y$**，不参与该项损失。字段命名建议：`traj_features` = 条件（4 维/经 MLP 前）；`traj` = 完整根位置用于监督与可视化。
   - **稀疏 mask 统一策略（当前实现）**：先在 **token 时间轴**采样 `traj_features_mask`（20%-30%），再按 4× 扩展回 frame 级得到 `traj_mask`，并在末尾 **补零到 `traj_length`**（保证长度对齐且尾帧语义显式）。这样可避免 frame→token 下采样导致的比例漂移。

3. **`datasets/humanml3d.py` — `collate_fn`**
   - 对新张量做 `pad_sequence`（与 `token` 同一 `T_max`），`traj_features_length` 与 `token_length` 共用或显式一致。

4. **`tests/test_task1_traj_batch.py`**（原 `datasets/verify_traj_batch.py`）
   - 断言 `traj_features` 与 `token` 对齐、`traj_length` 与 `feature_length` 一致等；运行：
     `python -m unittest tests.test_task1_traj_batch -v`

5. **`models/tools/traj_encoder.py`**
   - 将 `in_dim` 设为 **4**（与配置 `traj_encoder_in_dim` 一致，Task6）。

6. **`models/diffusion_forcing_wan_tiny.py`（及 full 版）**
   - 第一阶段可仍接收旧 `traj` 或切换到新键；若数据集已是 token 分辨率，**删除或条件化** `F.interpolate` 对齐，避免双次重采样引入模糊。

---

## 可能遇到的细节问题

| 问题 | 说明与建议 |
|------|------------|
| **crop 不对齐** | `process_feature` 随机裁剪后，`token` 已通过 `crop_start//4` 对齐；轨迹聚合必须使用 **同一 `crop_start` 与 `feature_length`**，否则条件与 latent 错位。 |
| **$\psi$ 首帧与 cumsum 边界** | `r_rot_ang` 与 `rot_vel` 的切片、`cumsum` 维度必须与现有 `recover_root_rot_pos` **逐行一致**，否则 control loss 与条件 yaw 语义分叉。 |
| **末帧 vs sin/cos 均值** | 窗口跨越 $2\pi$ 跳变时，末帧更稳；sin/cos 均值再归一化更平滑但实现要防除零。团队内 **二选一写死**。 |
| **与 RoPE 的契约** | 第 $k$ 个 `traj_features` 必须与第 $k$ 个 latent token **同一时间索引**（Task2 才能在拼接后共用 RoPE）；若中途插值改变长度，索引映射要显式文档化。 |
| **`traj_mask` 分辨率** | 当前实现采用「**token 级采样** → 4× 扩展到 frame 级并补齐到 `traj_length`」；若改回 frame 级采样，再下采样到 token 可能引入 keep 比例漂移。 |
| **推理端 `web_demo`** | `model_manager` 若构造轨迹数组，需同样 **插值/聚合到 token 步**，与训练分布一致。 |
| **条件 vs 监督维数** | 条件为 4 维 $(x,z,\cos\psi,\sin\psi)$；GT `traj` 仍为 3D，Task5 仅在 **索引 0、2** 上算 loss（**勿**写成「第 3 维为 $y$」——$y$ 为 **索引 1**）。 |
