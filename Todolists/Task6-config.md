# Task6 — 配置与开关

对应 `target.md` **§6 配置入口**。

---

## 实现目标

- 在 **`configs/ldf_tiny.yaml`**（及 `ldf.yaml` / `babel` 等变体）中集中管理 **轨迹条件维度、训练策略、LoRA/KV/聚合规则**，并与代码中 `cfg.model.params` 读取点 **一一对应**。
- 本文件默认只覆盖 FlexTraj 式轨迹条件；不保留 MotionStream 分支的切换开关（放在另一个目录实现）。

---

## 实现步骤

1. **梳理现有键（保持文档化）**
   - `use_traj_cond`、`traj_out_dim`、`traj_drop_out`、`control_loss_weight`、`freeze_backbone_for_traj`、`causal`、`chunk_size`、`noise_steps`、`prediction_type`、`cfg_scale`、`drop_out` 等。
   - 在 `train_ldf.py`、`generate_ldf.py`、`web_demo` 中 `grep` 所有 `cfg.model.params.get`，确保无遗漏。

2. **建议新增键（命名供参考）**
   - `traj_token_dim: int` — 轨迹 token 隐藏维（通常等于 `hidden_dim`）。
   - `traj_encoder_in_dim: int` — 默认 **4**（$x,z,\cos\psi,\sin\psi$）；不显式 `t_emb`，时间由共享 RoPE 提供。
   - `traj_aggregate_mode: "last" | "sincos_mean"` — 与 Task1 一致。
   - `lora_rank_traj: int`、`lora_alpha_traj: float` — Task2。
   - `use_traj_kv_cache: bool` — Task3，默认推理 true、训练可 false 便于调试。

3. **`utils/initialize.py` / `instantiate`**
   - 确认 `DiffForcingWanModel` 构造函数 **kwargs** 接收新参数；`OmegaConf` 未声明键时不要静默丢弃（必要时在 `load_config` 后 print unknown keys）。

4. **多配置一致性**
   - `configs/ldf_babel_tiny.yaml`、`stream_tiny.yaml` 等若用于训练，同步增加注释与默认值，避免仅 `ldf_tiny.yaml` 可跑。

5. **文档**
   - 在 `Todolists/target.md` 或 `README` 增加 **配置表**：键名、类型、默认值、影响模块（Task1–5）。

6. **实验管理**
   - `exp_name` 或 `wandb` tag 标明 `flextraj_v1`，便于与 baseline 区分。

---

## 可能遇到的细节问题

| 问题 | 说明与建议 |
|------|------------|
| **旧 checkpoint** | 新键无默认值时 `load_state_dict` 报错；保持默认 **关闭** 新路径直至首次训练保存新 ckpt。 |
| **冻结策略与 LoRA** | `freeze_backbone_for_traj` 需改为识别 `lora_` 前缀；否则 LoRA 被误冻结。 |
| **YAML 类型** | `devices: [0,1]` 与 `traj_aggregate_mode: last` 引号；字符串枚举避免被解析为 bool。 |
| **HuggingFace / 导出** | 若后续导出 `trust_remote_code`，配置需可序列化；避免 lambda。 |
| **demo 独立配置** | `web_demo` 的 yaml 路径与训练 yaml 分离时，**轨迹相关键** 手动同步，否则 demo 与训练分布不一致。 |
