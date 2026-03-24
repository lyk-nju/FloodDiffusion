# Todolists — FlexTraj 式轨迹改造

- **[`target.md`](target.md)**：总规范、**全局一致性约定**、与代码映射、Task 文档索引。
- **[`Task1-input.md`](Task1-input.md)**：`traj_features`（$x,z,\phi^{emb}$）、token 对齐、`traj` `(T,3)` 与 Task5 分工。
- **[`Task2-model.md`](Task2-model.md)**：序列拼接、**共享 RoPE**、traj-only LoRA、替换 `traj_proj`。
- **[`Task3-inference.md`](Task3-inference.md)**：FloodDiffusion 调度、traj KV 缓存、流式 API 对齐。
- **[`Task4-mask.md`](Task4-mask.md)**：traj 不看 latent、与 FlashAttention。
- **[`Task5-loss.md`](Task5-loss.md)**：**xz-only** control loss、索引 0/2 vs 1（$y$）、可选 $\phi$ 项。
- **[`Task6-config.md`](Task6-config.md)**：YAML 键、新旧路径互斥。

修改任一约定（如 loss 轴向、是否使用 `t_emb`、RoPE 规则）时，请同步更新 **`target.md` 全局一致性约定**与相关 Task。
