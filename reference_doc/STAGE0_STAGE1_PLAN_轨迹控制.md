# 阶段 0 与阶段 1 实施计划大纲（轨迹控制）

本文档是**可执行计划**：阶段 0 在数据侧加入根轨迹与 **traj_mask**（支持稀疏点控制）；阶段 1 在 LDF 中按 **MotionStreamer 方式**（隐藏层 concat + linear projection 还原）接入轨迹条件。并明确 **当前 dataset 返回 vs 阶段 0 目标返回** 的对照与各字段含义。

---

## 〇、Dataset 返回内容对照（当前 vs 阶段 0 目标）

### 当前：单样本 `_process()` 返回（list 中一个元素）

| Key | 类型 / Shape | 含义 | 下游谁用 |
|-----|----------------|------|----------|
| `dataset` | str | 数据集名（如 HumanML3D） | 日志 / 区分 |
| `name` | str | 样本 ID（文件名） | 日志 / 保存 |
| `feature` | np.ndarray `(T, 263)` 或 `(T_token, 4)` | **dataset 层**：`(T, 263)` 为原始 motion（来自 `feature_path`，crop 后）；**在 LDF 输入层**，LightningModule 会把 `batch["token"]`（`(T_token, 4)`）拷贝到 `model_batch["feature"]`，此时 `feature` 实际变成 4 维 token。 | VAE 训练用 263；LDF 训练多用 4 维 token |
| `feature_length` | int | 当前样本 **motion 有效长度** T（crop 后），与 `feature.shape[0]` 一致。 | Pad、mask、loss |
| `token` | np.ndarray `(T_token, 4)` | VAE 预计算潜在序列，T_token ≈ T/4。 | LDF 训练（作为“潜在”输入） |
| `token_length` | int | 当前样本 token 有效长度 T_token。 | Pad、截断 |
| `text` | str 或 List[str] | 单条描述（非 stream）或 `[一条]`（stream）。 | LDF 文本条件 |
| `text_tokens` | ... | 文本 token 化结果（若用）。 | 可选 |
| `feature_text_end` | List[int]（仅 stream） | 多段文本时每段结束帧，与 `feature_length` 等配合。 | LDF 多段文本 |

说明：**LDF 训练时**，若配置了 `token_path`，batch 里往往用 **token** 作为“动作序列”喂给模型（即 `x["feature"]` 实际可能是 token）；263 维的 `feature` 仍会从 `feature_path` 加载在 `data["feature"]`，用于在阶段 0 中**推导轨迹**。

### 当前：Collate 后一个 batch

| Key | 类型 / Shape | 含义 |
|-----|----------------|------|
| `feature` | tensor `(B, T_max, 263)` 或 `(B, T_token_max, 4)` | 按 batch 内最大长度 pad，padding 填 0。 |
| `feature_length` | tensor `(B,)` | 每个样本的有效长度。 |
| `token` | tensor `(B, T_token_max, 4)` | 同上，若存在则 pad。 |
| `token_length` | tensor `(B,)` | 每个样本的 token 有效长度。 |
| `dataset` / `name` / `text` / `feature_text_end` 等 | list 或保持原样 | 按 key 逐项 list 或 stack。 |

### 阶段 0 目标：单样本 `_process()` 新增/保持

在**保持上表所有现有 key** 的前提下，**新增**以下与轨迹相关的项：

| Key | 类型 / Shape | 含义 | 下游谁用 |
|-----|----------------|------|----------|
| `traj` | np.ndarray `(T, 3)` | **根节点 3D 轨迹**，与当前 crop 后的 motion 帧数 T 一致；由 263 维 `data["feature"]` 解析得到。 | 阶段 1 LDF 轨迹条件 |
| `traj_length` | int | 轨迹有效长度，等于当前 `feature_length`（crop 后 motion 长度 T）。 | Pad、对齐 |
| `traj_mask` | np.ndarray `(T,)` bool 或 float | **该帧轨迹是否被“观测/给定”**：1/True = 该帧轨迹有效、参与控制；0/False = 该帧未给定（稀疏控制时由模型补全或忽略）。训练时可设为：全 1（稠密）；或随机稀疏（如只保留 10%～30% 为 1），以支持**稀疏点控制**。 | 阶段 1 编码/融合时区分观测与未观测 |

### 阶段 0 目标：Collate 后一个 batch 新增

| Key | 类型 / Shape | 含义 |
|-----|----------------|------|
| `traj` | tensor `(B, T_max, 3)` | 轨迹 pad 到 batch 内最大 motion 长度 T_max；**padding 位置填 0**。 |
| `traj_length` | tensor `(B,)` | 每个样本的轨迹有效长度。 |
| `traj_mask` | tensor `(B, T_max)` bool 或 float | 与 `traj` 对齐；**padding 位置为 0/False**，有效区内按单样本逻辑（全 1 或稀疏）。 |

### 各部分含义小结

- **feature / feature_length**：当前表示 263 维 motion 或其替代（token）；长度 T 或 T_token，取决于配置。
- **token / token_length**：VAE 潜在序列，LDF 的“动作”输入多为它；长度 T_token。**T 与 T_token 关系**：T 为 motion 帧数，T_token 为潜在帧数，VAE 时间下采样 4 倍，故 **T_token ≈ T/4**。
- **text / feature_text_end**：文本条件，LDF 的 cross-attention 使用。
- **traj**：根轨迹 (x,y,z)，长度 **T**（与 263 维 motion 同长）。
- **traj_mask**：哪些帧的轨迹是“给定”的；**1 = 给定，0 = 未给定**；支持仅有少量稀疏点时仍能控制；长度 T。
- **traj_length**：轨迹有效长度，用于 pad 与对齐。

---

## 一、阶段 0：数据侧提供轨迹 + traj_mask（支持稀疏控制）

**目标**：在不大改训练脚本的前提下，让每个 batch 里多出 `traj`、`traj_length`、**`traj_mask`**，且与现有 `feature`/`feature_length` 对齐（轨迹为运动帧率，与 263 维 motion 同长）；**traj_mask 用于标记“哪些帧有轨迹观测”**，以便后续支持稀疏点控制。

### 1.1 轨迹定义与来源

- **轨迹内容**：根节点 3D 位置序列，形状 `(T, 3)`，T 为当前样本 motion 帧数（与 `feature_length` 一致）。
- **来源**：从 263 维 motion 中解析。263 维结构见 `utils/motion_process.py` 注释：  
  `root_rot_velocity(1), root_linear_velocity(2), root_y(1), ric_data(...), ...`  
  已有 `recover_root_rot_pos(data)`：输入 `(..., 263)`，返回 `r_rot_quat` 与 **`r_pos` (..., 3)**，即根位置。
- **traj_mask**：与 `traj` 同长度，`(T,)`。**1 表示该帧轨迹为观测值（参与控制），0 表示未给定（稀疏）**。训练时可选：全 1（稠密）；或按比例随机置 0（如保留 10%～30% 为 1），使模型学会在少量稀疏点下也能控制。
- **约定**：阶段 0 输出的轨迹与 **motion 帧数 T** 一致；LDF 训练时用的是 token（潜在序列），长度为 T_token ≈ T/4，对齐与 mask 下采样在阶段 1 的模型内做。

### 1.2 实施步骤

| 步骤 | 内容 | 文件/位置 |
|------|------|------------|
| 0.1 | **工具函数**：从 263 维得到根轨迹 (T, 3)。可复用 `recover_root_rot_pos`，输入 `(1,T,263)`，取 `r_pos.squeeze(0).cpu().numpy()`。若希望接口更清晰，可在 `utils/motion_process.py` 中新增 `extract_root_trajectory_263(feature_263: np.ndarray) -> np.ndarray`，返回 `(T, 3)`。 | `utils/motion_process.py` |
| 0.2 | **数据集**：在 `HumanML3DDataset._process()` 中，当存在 `data["feature"]`（263 维 motion）时：1）调用上述函数得到 `traj` (T, 3)；2）生成 **traj_mask** (T,)：默认全 True；可选在训练时随机稀疏（如 `traj_mask = np.zeros(T, dtype=bool)`，再随机选约 10%～30% 下标置 True，或保留首尾+均匀采样若干帧）。写 `output["traj"] = traj`，`output["traj_length"] = len(traj)`，`output["traj_mask"] = traj_mask`。注意：需确认 `_load_file_list` 中 263 维被加载（`feature_path`），以便用 `data["feature"]` 算 traj。 | `datasets/humanml3d.py` |
| 0.3 | **Collate**：在 `collate_fn` 中，对 `traj` 做与 `feature` 相同的 pad（按 `traj_length` pad 到 batch 内最大长度），得到 `output["traj"]` shape `(B, T_max, 3)`，**padding 填 0**；`output["traj_length"]` 为 `(B,)`。对 **traj_mask** 同样 pad 到 `(B, T_max)`，**padding 填 False/0**。若某样本无 traj（兼容未改动的数据），可跳过上述 key 或填 zeros/False。 | `datasets/humanml3d.py` → `collate_fn` |
| 0.4 | **校验**：写一小脚本（或 pytest）：从 HumanML3D 取一个 batch，检查 `output["feature"].shape`、`output["traj"].shape`、`output["traj_length"]`、**`output["traj_mask"].shape` 与 dtype**；可选地画一条轨迹的 XZ 平面图，并在图上标出 `traj_mask==1` 的点，确认稀疏/稠密合理。 | 新建脚本或 `datasets/` 下 test |

### 1.3 注意事项

- LDF 训练时若用的是 **token**，batch 里 `feature` 的 shape 可能为 `(B, T_token, 4)`，`feature_length` 为 token 长度。此时 `traj` / `traj_mask` 仍为 **motion 帧率** `(B, T_max, 3)` 与 `(B, T_max)`（T_max 为 motion 长度），与 T_token 不一致，在阶段 1 模型内做插值/下采样及 mask 对齐。
- **traj_mask** 在阶段 1 中：对齐到 T_token 后，未观测位置（mask=0）可在编码前将 traj 值置 0 或单独学 “missing” embedding，仅在有观测位置注入真实轨迹，以支持稀疏点控制。

### 1.4 阶段 0 交付物

- 数据 pipeline 输出包含：`traj` `(B, T_max, 3)`、`traj_length` `(B,)`、**`traj_mask` `(B, T_max)`**（T_max 为 batch 内最大 motion 长度）。
- 工具函数：`extract_root_trajectory_263`（或等价）可被数据集与后续脚本复用。
- 一次简单校验（脚本或用例）通过；可选验证稀疏 traj_mask（仅部分点为 1）的 batch 正确。

---

## 二、阶段 1：LDF 内 MotionStreamer 式轨迹条件（concat + linear projection）

**目标**：在 LDF 的 backbone（WanModel）中，将轨迹编码为与时间对齐的向量，在**隐藏层**与当前 latent 特征 **concat**，再用 **linear projection** 压回原维度，即完成「[T, D] + [T, L] → [T, D+L] → Linear → [T, D]」；不改变原有文本条件与噪声调度逻辑。

### 2.1 数据流与长度对齐

- LDF 的 `x["feature"]`：shape `(B, T_token, 4)`，即 **潜在序列**（T_token 为 VAE 下采样后长度，T_token ≈ T/4）。
- LDF 的 `x["traj"]`：shape `(B, T_max, 3)`，**运动帧率**（T_max 为 batch 内最大 motion 长度）。
- LDF 的 `x["traj_mask"]`：shape `(B, T_max)`，**哪些帧有轨迹观测**（1=给定，0=未给定）；用于稀疏点控制。
- **对齐**：在模型内将 traj 与 traj_mask 从 **T_max（motion 帧）** 插值/下采样到 **T_token**，使与 backbone 的序列长度一致。traj 可用 `F.interpolate` 或线性插值得到 `traj_aligned` (B, T_token, 3)；traj_mask 可插值为 float (B, T_token) 或 nearest 得到 bool，得到 `traj_mask_aligned` (B, T_token)。
- **稀疏处理**：在编码前，可将 `traj_aligned` 中 `traj_mask_aligned==0` 的位置置 0（或乘 mask），再送 TrajEncoder；或编码器内部接受 mask，对未观测位置输出零/可学习的 “missing” 向量，使模型仅在观测点受轨迹约束、未观测点自由生成。

### 2.2 架构约定（MotionStreamer 式）

- **轨迹编码器**：输入 `(B, T_token, 3)`，输出 `(B, T_token, L)`，L 为轨迹隐维度（超参，如 64 或 128）。
- **融合位置**：在 WanModel 中，**patch embedding 之后、第一个 Transformer block 之前**，当前 `x` 的 shape 为 `(B, L, D)`（L=T_token，D=hidden_dim）。  
  - 将轨迹编码 `(B, L, L_traj)` 与 `x` 在最后一维 concat：`x = cat([x, traj_emb], dim=-1)` → `(B, L, D+L_traj)`。  
  - 再经过一个 Linear：`x = self.traj_proj(x)` → `(B, L, D)`，恢复原维度，后续 blocks 不变。
- **无轨迹时**：若 `traj_emb` 为 None 或未提供，则不做 concat，等价于 `traj_proj` 仅作用在 `x` 上（或该分支不创建，见下「可选」）。

### 2.3 实施步骤

| 步骤 | 内容 | 文件/位置 |
|------|------|------------|
| 1.1 | **轨迹编码器**：新建 `models/tools/traj_encoder.py`。实现 `TrajEncoder`：输入 (B, T, C_in)（C_in=3），用 1D Conv 或 MLP 按时间编码，输出 (B, T, L)。例如：`Linear(3, 64) -> GELU -> Linear(64, L)` 逐帧；或 `Conv1d(3, 64, k) -> ... -> Linear -> (B, T, L)`。输出维度 L 由配置指定（如 64）。 | `models/tools/traj_encoder.py`（新建） |
| 1.2 | **WanModel 支持 traj 融合**：在 `models/tools/wan_model.py` 的 `WanModel` 中：  
  - `__init__`：增加可选参数 `traj_dim=0`（0 表示不使用）。若 `traj_dim > 0`，则新增 `self.traj_proj = nn.Linear(dim + traj_dim, dim)`（或通过外部传入 `traj_hidden_dim` 作为 L）。  
  - `forward`：增加可选参数 `traj_emb=None`。若 `traj_emb` 非 None，则在 **patch embedding 之后、得到 x (B, L, D) 之后**，做 `x = torch.cat([x, traj_emb], dim=-1)`，再 `x = self.traj_proj(x)`；否则不 concat，若存在 `traj_proj` 则仅对 `x` 做线性（或不做，见下方「可选」）。  
  - 注意：`traj_emb` 的序列长度必须与当前 `x` 的 L 一致（由调用方保证已对齐到 T_token）。 | `models/tools/wan_model.py` |
| 1.3 | **DiffForcingWanModel 侧**：  
  - `__init__`：若启用轨迹（如 `use_traj_cond=True`），则创建 `TrajEncoder`（输出维度 L），并计算 `traj_hidden_dim=L`；将 `traj_hidden_dim` 传给 WanModel（若 Wan 需要），并在 WanModel 中创建 `traj_proj = Linear(dim + L, dim)`。  
  - `forward`：从 `x` 取 `x["traj"]`、`x["traj_length"]`、**`x["traj_mask"]`**；若 traj 不存在则置为 None，不传 traj。若有 traj：1）将 traj 从 (B, T_max, 3) 对齐到 T_token；2）**将 traj_mask 从 (B, T_max) 对齐到 (B, T_token)**（插值或 nearest）；3）在未观测位置（mask=0）将 traj 置 0 或乘 mask，再送入 `TrajEncoder` 得到 `traj_emb` (B, T_token, L)；4）调用 `self.model(..., traj_emb=traj_emb)`。  
  - **对齐方式**：batch 内对 traj / traj_mask 按最大 T_token 做插值后 pad，与 WanModel 内 pad 后的 x 一致。 | `models/diffusion_forcing_wan.py` |
| 1.4 | **配置**：在 `configs/ldf.yaml` 的 `model.params` 中增加例如：`use_traj_cond: true`、`traj_dim: 3`、`traj_hidden_dim: 64`、`traj_drop_out: 0.1`（可选，用于训练时随机丢弃轨迹以做 CFG）。在 `DiffForcingWanModel` 与 `WanModel` 的实例化处读取这些参数。 | `configs/ldf.yaml`、模型构造函数 |
| 1.5 | **训练时轨迹 dropout（可选）**：在 `forward` 中，若 `use_traj_cond` 且 `traj_drop_out > 0`，以概率 `traj_drop_out` 将 `traj_emb` 置零或设为 None（不注入轨迹），与文本 CFG 一致。 | `models/diffusion_forcing_wan.py` |
| 1.6 | **推理**：`generate` / `stream_generate` 中同样从 `x` 读 traj（若有），做同样的对齐与编码，传入 `self.model(..., traj_emb=...)`。若无 traj，则 `traj_emb=None`，行为与当前无轨迹版本一致。 | `models/diffusion_forcing_wan.py` |

### 2.4 WanModel 与 stream / 变长

- 当前 WanModel 的 `x` 是 list 的 patch 后拼接为 (B, seq_len, D)，seq_len 为 pad 后的长度。`traj_emb` 应与之一致：shape `(B, seq_len, L)`，这样 concat 后 `(B, seq_len, D+L)` 再投影回 `(B, seq_len, D)`。
- 若使用 `wan_model_cross_rope.py`（stream 用），需在**同一位置**（patch 后、blocks 前）加入相同的 concat + traj_proj，并保证 `traj_emb` 的 seq_len 与当前 step 的 x 一致。

### 2.5 阶段 1 交付物

- **TrajEncoder** 模块，输入 (B, T, 3)，输出 (B, T, L)。
- **WanModel** 支持可选 `traj_emb`，在隐藏层 concat + linear 还原维度。
- **DiffForcingWanModel** 中：traj 长度对齐 → TrajEncoder → 传入 WanModel；forward/generate/stream_generate 均支持有无 traj。
- 配置项：`use_traj_cond`、`traj_dim`、`traj_hidden_dim`、`traj_drop_out`。
- 训练与推理可跑通；可通过「换轨迹不换文本」观察生成差异，做一次简单验证。

---

## 三、两阶段总览表

| 阶段 | 主要改动 | 新增文件 | 验证方式 |
|------|----------|----------|----------|
| **0** | 从 263 解析根轨迹；dataset 输出 traj / traj_length / **traj_mask**；collate 对 traj、traj_mask pad | 无（或仅小 util） | 取一个 batch 检查 shape 与 mask；可选画轨迹并标出 mask=1 的稀疏点 |
| **1** | TrajEncoder；WanModel concat+proj；DiffForcingWanModel 对齐 traj 与 traj_mask、稀疏处置 0 后编码并传入 traj_emb；配置项 | `models/tools/traj_encoder.py` | 训练若干 step；推理时改 traj（及可选稀疏 traj_mask）看生成变化 |

---

## 四、建议执行顺序

1. **阶段 0**：先做 0.1 → 0.2 → 0.3 → 0.4，确认 dataloader 输出正确再进入阶段 1。  
2. **阶段 1**：按 1.1 → 1.2 → 1.3 → 1.4 实现；1.5、1.6 与 stream 支持可随后补全。  
3. 若项目中有 **diffusion_forcing_wan_tiny** 或其它 backbone 变体，需在**同一融合位置**为 WanModel 增加相同 concat+proj 逻辑，并在对应 DiffForcing 模型中传入 `traj_emb`。

按此计划即可分步实现阶段 0 与阶段 1；阶段 1 严格按 MotionStreamer 的「隐藏层 concat + linear projection 还原」方式接入轨迹。
