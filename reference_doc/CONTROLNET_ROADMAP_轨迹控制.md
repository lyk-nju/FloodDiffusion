# 为 FloodDiffusion 添加轨迹 ControlNet：新手准备与推进指南

作为新手，建议**分阶段推进**：先打牢基础、再做最小可用的“轨迹条件”，最后再考虑完整的 ControlNet 结构。这样每一步都能验证，不容易卡死。

---

## 一、准备阶段（先做完再动手改代码）

### 1.1 确保已经理清的两件事

1. **LDM 两条线**（见 `PAPER_TO_CODE_FloodDiffusion.md`）  
   - VAE：263D ↔ 4D 潜在，只做压缩/解压。  
   - LDF：只在 4D 潜在上做“加噪→去噪”，条件目前只有文本。  
   - 轨迹条件要加在 **LDF 这一侧**，和文本一样作为“条件”，不改 VAE。

2. **数据流到 LDF 的入口**  
   - 训练：`train_ldf.py` → 配置里的 `model` → `DiffForcingWanModel.forward(x)`。  
   - `x` 是一个 dict，目前至少有：`feature`（4D token）、`feature_length`、`text`（及可选的 `feature_text_end`）。  
   - **轨迹条件**之后也会从 `x["traj"]` 之类的地方进来，和 `x["feature"]`、`x["text"]` 一起用。

### 1.2 明确“轨迹”在你项目里指什么

选一种先做，避免一开始就做太复杂：

- **方案 A（推荐起步）**：只用 **根节点 3D 轨迹**  
  - 263 维里：根相关在前几维（见 `utils/motion_process.py` 的 `recover_root_rot_pos` 等）。  
  - 可以从 263 里解析出 root 的 3D 位置序列，形状例如 `(T, 3)`，作为“轨迹”。  
- **方案 B**：全身 22 关节 3D 轨迹  
  - 即 `(T, 22, 3)`，或展平为 `(T, 66)`。  
  - 可以做“随机 mask 一部分关节”的训练，但实现和调试都更重，建议在 A 跑通后再做。

建议：**第一阶段只做根轨迹 3D**，先让“有轨迹条件 vs 无轨迹条件”在训练/推理里跑通，再考虑全身+mask。

### 1.3 需要提前会一点的技能

- 会用 **PyTorch**：`nn.Module`、`forward`、tensor 的 `shape`、`device`。  
- 会改 **OmegaConf / YAML**：在 `configs/ldf.yaml` 里加几项、改 `model.params`。  
- 会 **从 263 维里取一段**：`utils/motion_process.py` 里已有从 263 恢复 root 的代码，你可以先写一个小脚本，输入一个 `(T, 263)` 的 npy，输出 `(T, 3)` 的 root 位置，确认和可视化一致。

---

## 二、推进路线总览（三阶段）

```
阶段 0：准备（文档 + 轨迹解析脚本）
    ↓
阶段 1：数据里带轨迹 + 简单条件（“轨迹当额外 context” 或 “concat 进 hidden”）
    ↓  验证：能训练、能推理、轨迹对生成有影响
阶段 2：改成完整 ControlNet（可选：复制一层 backbone + residual 注入）
```

先做 **阶段 1**，再考虑阶段 2，这样即使不做 ControlNet，你也已经有一个“带轨迹条件的 LDF”可用。

---

## 三、阶段 0：准备（建议 1–2 天）

### 3.1 从 263 里解析轨迹并写进数据

目标：**不改模型**，只在数据 pipeline 里增加“轨迹”字段，保证后面模型能直接读。

1. **确定 263 里 root 的维度**  
   - 看 `utils/motion_process.py`：  
     - `recover_root_rot_pos` 里用到了 `data[..., 0]`（rot_vel）、`data[..., 1:3]`（linear_vel）、`data[..., 3]`（root_y）。  
   - HumanML3D 的 263 定义里，root 相关在前若干维；你需要得到的是 **每帧的 root 3D 位置**。  
   - 一种做法：用现有的 `recover_root_rot_pos` + 累积位置，得到 `(T, 3)` root 位置；或者直接用 263 里某几维的简单组合（如 1:4 的某种变换）作为“轨迹”先跑通，再换更精确的解析。

2. **在 `datasets/humanml3d.py` 里加轨迹**  
   - 在 `_process()` 里，当有 `feature` 时：  
     - 从 `feature`（或 `data["feature"]`）解析出 root 轨迹，得到 `(T, 3)` 或 `(T, C_traj)`。  
     - `output["traj"]` = 该数组，`output["traj_length"]` = 长度（一般等于 `feature_length`）。  
   - 在 `collate_fn` 里：  
     - 对 `traj` 做和 `feature` 一样的 pad（`pad_sequence`），得到 `(B, T, C_traj)`；  
     - `traj_length` 拼成 tensor。

3. **验证**  
   - 写一个小脚本：读一个 batch，检查 `output["feature"].shape`、`output["traj"].shape`、`output["traj_length"]`，再画一两帧的轨迹（例如 XZ 平面），确认合理。

### 3.2 可选：轨迹 dropout（和文本 CFG 对齐）

- 训练时有时“不用轨迹”（类似文本的 `drop_out`），推理时才能做“无轨迹 / 弱轨迹”的 CFG。  
- 可以在 **阶段 1 的模型里** 再做：先实现“始终用轨迹”，再在 `forward` 里对 `x["traj"]` 做随机置零或替换为 zeros。

---

## 四、阶段 1：最小可用的“轨迹条件”LDF（建议 3–5 天）

目标：**不破坏原有训练/推理**，让模型多一个输入 `traj`，并让预测依赖它。推荐两种实现方式二选一（先做 A 更省事）。

### 4.1 方案 A：轨迹当“额外 context”（和文本一起进 cross-attention）

思路：把轨迹编码成一段“伪文本”embedding，和现有文本 context 拼在一起，让 WanModel 的 cross-attention 同时看到文本+轨迹。

1. **轨迹编码器**（新模块，例如放在 `models/tools/traj_encoder.py`）  
   - 输入：`(B, T, C_traj)`，例如 C_traj=3。  
   - 用 1D Conv 或 MLP 按时间编码成 `(B, T, D)`，再线性映射到和文本一样的维度（如 4096 或 768），得到 `(B, T, text_dim)`。  
   - 输出可以按“每帧一个向量”组织，和现有 `all_text_context` 的“每帧一个 text embedding”一致。

2. **在 `DiffForcingWanModel` 里接上**  
   - `__init__`：  
     - 若 `use_traj_cond=True`，则 `self.traj_encoder = TrajEncoder(...)`，并设 `traj_dim`、`traj_drop_out` 等。  
   - `forward`（及 `generate` / `stream_generate`）：  
     - 从 `x` 里取 `x["traj"]`、`x["traj_length"]`（若没有则用 zeros 或 None 表示无轨迹）。  
     - 用 `traj_encoder` 得到轨迹 embedding，按**帧对齐**拼到 `all_text_context`：例如每帧的 context = [text_emb; traj_emb] 再过一个线性压回原 context 维，或者直接 concat 成更长的 context 序列（注意 `context_lens` 要一起改）。  
   - 训练时：  
     - 可选 `traj_drop_out`：以一定概率把轨迹置零，做“无轨迹”条件，方便以后做 CFG。

3. **配置**  
   - 在 `configs/ldf.yaml` 的 `model.params` 里加 `use_traj_cond: true`、`traj_dim: 3`、`traj_drop_out: 0.1` 等，并在 `DiffForcingWanModel` 里从 `**kwargs` 或显式参数读入。

4. **验证**  
   - 训练几个 step，loss 能正常下降；  
   - 推理时给一条直线轨迹 vs 一条曲线轨迹，生成动作在轨迹上应有差别（目视或简单定量）。

### 4.2 方案 B：轨迹编码后与 latent 在隐层 concat（MotionStreamer 风格）

思路：在 backbone 的**输入**或**第一层之后**，把“轨迹编码”和“当前 latent”在特征维上 concat，再线性压回原维度，即 `[T, D] + [T, L] → [T, D+L] → Linear → [T, D]`。

1. **轨迹编码器**  
   - 输入 `(B, T, C_traj)`，输出 `(B, T, L)`，L 为轨迹隐维度。

2. **在 WanModel 的 patch embedding 之后**  
   - 当前：`x` 经过 `patch_embedding` 得到 `(B, L, D)`（L=序列长，D=hidden_dim）。  
   - 轨迹编码要和时间对齐：若 VAE 有下采样，轨迹可能是 (B, T_orig, C_traj)，需要先下采样或插值到 (B, L)，再编码成 (B, L, L_traj)。  
   - 然后：`x = concat([x, traj_enc], dim=-1)` 得到 (B, L, D+L_traj)，再 `x = self.traj_proj(x)` 压回 (B, L, D)。  
   - 这需要改 `WanModel.forward` 的接口：多一个可选参数 `traj_emb`，并在第一层后做 concat+proj；同时 `DiffForcingWanModel` 里在调用 `self.model(..., traj_emb=...)` 时传入轨迹编码。

3. **时间对齐**  
   - LDF 里用的是 **token 序列**（4D latent 按时间），长度和 VAE 的潜在长度一致（约 T/4）。  
   - 若轨迹是 (B, T, 3)，需要先对齐到 token 长度：例如线性插值，或重复，得到 (B, L_token, 3)，再编码成 (B, L_token, L)。

4. **验证**  
   - 同方案 A：训练能跑、推理时改轨迹能看出生成差异。

建议：**先实现方案 A**（轨迹当额外 context），改动的文件少、不动 WanModel 内部结构；方案 B 需要改 WanModel 的 forward 和调用处，适合在 A 跑通后再做。

---

## 五、阶段 2：完整 ControlNet（可选，建议在阶段 1 稳定后再做）

目标：**复制一份“轻量 backbone”**，输入只有轨迹条件，输出与主 backbone 每层加 residual，实现“轨迹控制分支”。

1. **ControlNet 分支**  
   - 新建 `models/tools/controlnet_traj.py`（或类似名字）：  
     - 输入：轨迹编码 `(B, T, L)` 或 (B, L, L)。  
     - 结构：几层与 WanModel 类似的 block（可更浅、更窄），**不**做文本 cross-attention，只做 self-attention + 与时间对齐的 conditioning。  
     - 输出：每层一个 residual 张量，shape 与主 backbone 该层一致（B, L, D）。

2. **在主 backbone 里接 residual**  
   - 修改 `WanModel`（或 `wan_model_cross_rope.py`）：  
     - `forward` 增加可选参数 `control_residuals: List[Tensor]`。  
     - 在 `for block in self.blocks` 里，每层：`x = x + control_residuals[i]`（若提供）。  
   - 在 `DiffForcingWanModel` 里：  
     - 先跑 ControlNet 分支得到 `control_residuals`，再调用 `self.model(..., control_residuals=control_residuals)`。

3. **训练策略**  
   - 先**冻结**主 backbone，只训 ControlNet 分支，这样不会破坏原有文本→动作能力；  
   - 再视情况解冻主 backbone 做端到端微调。

4. **与文本 CFG 一致**  
   - 训练时用 `traj_drop_out` 随机丢掉轨迹，推理时可以做“有轨迹 vs 无轨迹”的 CFG（和 `cfg_scale` 类似）。

---

## 六、需要改动的文件清单（按阶段）

| 阶段 | 文件 | 改动要点 |
|------|------|----------|
| 0 | `utils/motion_process.py`（或新 util） | 从 263 解析 root 轨迹 (T, 3) |
| 0 | `datasets/humanml3d.py` | `_process` 里算 `traj`/`traj_length`；`collate_fn` 里 pad `traj` |
| 1A | `models/tools/traj_encoder.py`（新建） | 轨迹 → embedding (B,T,D) 或 (B,T,text_dim) |
| 1A | `models/diffusion_forcing_wan.py` | 读 `x["traj"]`，调用 traj_encoder，拼进 all_text_context 或拼成 extra context |
| 1A | `configs/ldf.yaml` | `model.params` 增加 use_traj_cond、traj_dim、traj_drop_out |
| 1B | 同上 + `models/tools/wan_model.py` | WanModel.forward 增加 traj_emb，patch 后 concat+proj |
| 2 | `models/tools/controlnet_traj.py`（新建） | ControlNet 分支，每层输出 residual |
| 2 | `models/tools/wan_model.py` | forward 接受 control_residuals，每层加 residual |
| 2 | `models/diffusion_forcing_wan.py` | 建 ControlNet、算 control_residuals、传入 model |

---

## 七、建议的时间与顺序

1. **第 1 周**：阶段 0 — 跑通数据 pipeline，确认 `feature`、`traj`、`traj_length` 在 dataloader 里正确；可选写一个“从 263 解析轨迹”的脚本并画图。  
2. **第 2 周**：阶段 1A — 实现轨迹 encoder + 在 LDF 里当“额外 context”，能训练、能推理，并观察“换轨迹会改变生成”。  
3. **第 3 周及以后**：视需要做 1B（concat 进 hidden）或阶段 2（ControlNet）；同时可加 traj_drop_out 和推理时的轨迹 CFG。

这样推进，每一步都有可验证的小目标，遇到问题也容易定位是在数据、还是在模型、还是在配置。如果你愿意，我可以下一步帮你具体写：**阶段 0 里从 263 解析 root 轨迹的函数** 或 **阶段 1A 里 TrajEncoder + 在 DiffForcingWanModel 里接上的最小代码片段**（含需要改动的行号或补丁）。
