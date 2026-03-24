# FloodDiffusion 根轨迹控制实现复盘

> 本文档复盘了为 FloodDiffusion 添加 MotionStream 式轻量级根轨迹稀疏控制及 MotionLCM 显式轨迹对齐损失的完整实现过程。

---

## 一、目标与背景

- **目标**：在 FloodDiffusion 上实现根轨迹稀疏控制，支持用户指定根节点 3D 轨迹的稀疏约束（20–30% 帧）
- **参考方法**：MotionStream 的 concat 式轻量控制、MotionLCM 的显式 control loss
- **训练方式**：从预训练 LDF post-train，ControlNet 式零初始化，保持原有无轨迹控制能力

---

## 二、实现阶段概览

### 阶段 0：数据层 (Data)

| 内容 | 文件 | 说明 |
|------|------|------|
| 根轨迹提取 | `utils/motion_process.py` | `extract_root_trajectory_263()` 从 263D motion 提取根 3D 轨迹 |
| 数据集处理 | `datasets/humanml3d.py` | `_process()` 中计算 `traj`、`traj_length`，生成 20–30% 稀疏 `traj_mask` |
| Collate | `datasets/humanml3d.py` | `collate_fn` 对 `traj`、`traj_mask`、`traj_length` 做 pad/stack |
| 校验脚本 | `tests/test_task1_traj_batch.py` | 验证 dataloader 输出与 feature–token 对齐 |

### 阶段 1：模型层 (Model)

| 内容 | 文件 | 说明 |
|------|------|------|
| 轨迹编码器 | `models/tools/traj_encoder.py` | TrajEncoder：3→64→2，轻量升维再降维 |
| Wan 主干 | `models/tools/wan_model.py` | 增加 `traj_dim`、`traj_proj`，对 traj 列零初始化 |
| LDF 主模型 | `models/diffusion_forcing_wan.py` | 集成 traj 条件、TrajEncoder、对齐与 dropout |
| 训练脚本 | `train_ldf.py` | 传递 traj 相关字段，`on_load_checkpoint` 支持 `strict=False` |
| 配置 | `configs/ldf.yaml` | `use_traj_cond`、`traj_out_dim`、`traj_drop_out` 等参数 |

### 阶段 2：Feature–Token 时间对齐修复

| 问题 | 原因 | 处理 |
|------|------|------|
| traj 与 token 错位 | `process_feature`、`process_token` 各自随机 crop | 统一 crop：`process_feature` 返回 `crop_start`，`process_token` 用同一窗口裁剪 |

- `process_feature`：返回 `(feature, feature_length, crop_start)`
- `process_token`：新增 `crop_start`、`feature_length`，按 `token_start = crop_start//4`、`token_len = feature_length//4` 对齐

### 阶段 3：MotionLCM 显式 Control Loss

| 内容 | 文件 | 说明 |
|------|------|------|
| 根轨迹提取 | `utils/motion_process.py` | `extract_root_trajectory_263_torch()` 供 control loss 使用 |
| 预测潜变量 | `models/diffusion_forcing_wan.py` | 在 forward 中构造 `pred_x0_latent` 并通过 `control_aux` 返回 |
| Control Loss | `train_ldf.py` | VAE 解码 → 提取根轨迹 → 按 `traj_mask` 计算 L2 损失 |
| 配置 | `configs/ldf.yaml` | `control_loss_weight: 1.0` |

**Control Loss 公式**（与 MotionLCM 一致）：

$$
\mathcal{L}_{\text{control}} = \frac{\sum_{i,t} m_{it} \| \hat{p}_t - g_t \|^2}{\sum_{i,t} m_{it}}
$$

其中 $m_{it}$ 为 `traj_mask`，$\hat{p}$ 为预测根轨迹，$g$ 为 GT 根轨迹。

---

## 三、涉及的主要文件

```
FloodDiffusion/
├── configs/ldf.yaml                    # use_traj_cond, control_loss_weight 等
├── datasets/
│   ├── humanml3d.py                    # traj 生成、feature-token 对齐、collate
│   └── (已迁) test_task1_traj_batch.py # 数据与对齐校验
├── models/
│   ├── diffusion_forcing_wan.py        # traj 条件、control_aux
│   └── tools/
│       ├── traj_encoder.py             # TrajEncoder
│       └── wan_model.py                # traj_proj、零初始化
├── train_ldf.py                        # traj 传入、control loss、checkpoint 加载
├── utils/
│   └── motion_process.py               # extract_root_trajectory_263(_torch)
└── docs/
    └── RECAP_轨迹控制实现复盘.md        # 本文档
```

---

## 四、如何训练

### 4.1 环境准备

```bash
# 激活 conda 环境
conda activate flooddiffusion

# 确保 paths 配置正确
# 检查 configs/paths.yaml 或 configs/paths_default.yaml 中的：
# - dirs.raw_data  (HumanML3D 数据路径)
# - dirs.outputs   (输出目录)
# - dirs.deps      (VAE、T5 等依赖)
```

### 4.2 从头训练

```bash
python train_ldf.py --config configs/ldf.yaml
```

### 4.3 从预训练 LDF 继续训练（Post-train）

在 `configs/ldf.yaml` 中设置：

```yaml
resume_ckpt: "${dirs.outputs}/20251106_063218_ldf/step_step=50000.ckpt"
# 或使用绝对路径
```

然后执行：

```bash
python train_ldf.py --config configs/ldf.yaml
```

- `on_load_checkpoint` 会以 `strict=False` 加载预训练权重
- `traj_encoder`、`traj_proj` 中与 traj 相关部分保持零初始化
- 行为类似 ControlNet 的 post-train

### 4.4 主要配置项

| 配置项 | 说明 | 默认 |
|--------|------|------|
| `use_traj_cond` | 是否启用轨迹条件 | `true` |
| `traj_out_dim` | 轨迹嵌入输出维度 | `2` |
| `traj_drop_out` | 轨迹条件 dropout 概率 | `0.1` |
| `control_loss_weight` | Control loss 权重 | `1.0` |
| `control_loss_weight=0` | 关闭显式 control loss | - |

### 4.5 校验数据

```bash
python -m unittest tests.test_task1_traj_batch -v
```

预期输出包括：batch keys、shape 检查、feature–traj 对齐、feature–token 对齐（VAE 4×）、traj_mask 稀疏比例等。

---

## 五、训练日志说明

- `train_loss/total`：总损失（mse + λ × control）
- `train_loss/mse`：扩散重建损失
- `train_loss/control`：轨迹对齐损失（存在时）
- `val_loss/*`：对应的验证损失

---

## 六、已知限制与待完善

1. **MultiDataset**：`datasets/multi.collate_fn` 尚未支持 traj，`ldf_babel.yaml` 等需要扩展
2. **推理脚本**：`generate_ldf.py` 暂未支持 traj 输入
3. **Tiny 模型**：`diffusion_forcing_wan_tiny` 未接入轨迹条件
