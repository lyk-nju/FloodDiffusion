# FloodDiffusion：论文与代码对应说明

本文档帮助初学者理清：**（1）LDM 整体结构（VAE + Latent Diffusion）及两者关系**；**（2）论文中的噪声调度与文本相关掩码机制在代码中的具体实现**。对应论文：*FloodDiffusion: Tailored Diffusion Forcing for Streaming Motion Generation*。

---

## 一、整体结构：这是一个 LDM（Latent Diffusion Model）

### 1.1 两阶段流水线

论文与代码都采用 **Latent Diffusion** 结构，分两步：

```
原始动作空间 (263维/帧)  →  [VAE]  →  潜在空间 (4维/帧)  →  [Latent Diffusion]  →  去噪后的潜在  →  [VAE 解码]  →  重建动作
```

- **VAE**：只做“压缩/解压”，不参与扩散。训练时单独训（`train_vae.py`），推理时先 encode 再 decode。
- **Latent Diffusion（LDF）**：只在 **潜在空间** 里做“加噪→去噪”，条件为文本。训练脚本是 `train_ldf.py`，模型在 `models/diffusion_forcing_wan.py`（或 `diffusion_forcing_wan_tiny.py`）。

所以：**数据流是「263D → VAE encode → 4D latent → LDF 去噪 → 4D latent → VAE decode → 263D」**。LDF 的输入输出维度都是 `input_dim=4`（潜在维度），不是 263。

### 1.2 为什么要先过 VAE？

- 263 维直接做扩散计算量大，且高维噪声难学。
- 在 4 维潜在空间做扩散，步数少（如 10 步）、速度快，适合流式生成。
- 论文强调 **Causal VAE**：解码时时刻 \(t\) 只依赖 \(\leq t\) 的潜在，这样流式解码时不需要“未来帧”。

---

## 二、VAE 部分：结构与代码对应

### 2.1 论文描述（Section 3.4, Causal VAE）

- 输入：263 维动作序列，做标准化（mean/std）。
- 编码器：**因果** 1D 卷积 + 时间下采样，输出高斯参数 \((\mu, \sigma^2)\)，重参数化得到潜在 \(z\)。
- 潜在：时间下采样率 4，通道数 4 → 即 **每 4 帧得到 1 个潜在向量，维度 4**。
- 解码器：因果结构，从 \(z\) 重建 263 维。

### 2.2 代码位置与对应关系

| 论文概念 | 代码位置 | 说明 |
|----------|----------|------|
| 263D 输入 / 标准化 | `models/vae_wan_1d.py` → `forward()` 中 `(features - self.mean) / self.std` | 使用数据集的 mean/std（或 buffer） |
| 编码器 / 解码器 | `models/tools/wan_vae_1d.py` → `WanVAE_` | 内部是因果 1D Conv + ResBlock，`encode` / `decode` |
| 时间下采样 | `vae_wan_1d.py` 中 `temperal_downsample=[True, True]` | 两次 True → 下采样 2×2=4 倍 |
| 潜在维度 4 | 配置 `configs/vae_wan_1d.yaml` 中 `z_dim: 4`，或 `ldf.yaml` 里 `test_vae.params.z_dim` | 与论文 “latent channel dimension to 4” 一致 |
| 因果性 | `wan_vae_1d.py` 中 `CausalConv1d`（左侧 padding，不看到未来） | 解码时不用未来帧，支持流式 decode |
| 损失 | `vae_wan_1d.py` → recons (SmoothL1) + velocity 项 + KL | 论文中的重建 + root/velocity 等 |

**数据形状（便于对照）：**

- 输入 `feature`: `(B, T, 263)`，经 `preprocess` 成 `(B, 263, T)` 给 `WanVAE_`。
- 编码输出 `mu/log_var`: `(B, z_dim, T')`，其中 \(T' \approx T/4\)。
- 解码输出再 `postprocess` 回 `(B, T, 263)`（长度可能对齐到 \(T\)）。

**小结**：VAE 只负责「263 ↔ 4 维潜在」的映射；扩散部分完全不碰 263 维，只碰 4 维潜在。

---

## 三、Latent Diffusion 部分：噪声调度（论文重点之一）

### 3.1 论文中的向量化时间调度（Section 3.1–3.2）

- 对序列中**每个位置** \(k\) 有一个噪声水平，由**标量时间** \(t\) 和**位置** \(k\) 共同决定：
  - \(\alpha^k_t = \mathrm{clamp}(t - k/n_s,\, 0,\, 1)\)
  - \(\beta^k_t = 1 - \alpha^k_t\)
- \(n_s\)：流式步长（论文里 “streaming step-size parameter”），代码里即 **`chunk_size`**（如 5）。
- 加噪形式：\(x^k_t = \alpha^k_t z^k + \beta^k_t \epsilon^k\)（\(z\) 为干净数据，\(\epsilon\) 为标准高斯）。
- 因此：
  - \(k\) 很小（序列左侧）：\(t - k/n_s\) 大 → \(\alpha^k_t=1\) → 几乎全是干净数据；
  - \(k\) 很大（序列右侧）：\(\alpha^k_t=0\) → 全是噪声；
  - 中间一段是“过渡带”（三角形），这就是 **lower triangular / 下三角** 调度。

**Active window（论文 Lemma 3.6 / 3.8）**：

- \(m(t) = \lceil (t-1) n_s \rceil\)：已完全去噪的最后一帧下标；
- \(n(t) = \lceil t n_s \rceil\)：当前“激活”的最后一帧下标；
- 训练/推理时只需对 **\([m(t), n(t))\)** 这一段计算速度场并更新，左侧已固定，右侧仍是噪声。

### 3.2 代码中的实现（与论文一一对应）

**（1）噪声水平公式**

```python
# models/diffusion_forcing_wan.py 或 diffusion_forcing_wan_tiny.py

def _get_noise_levels(self, device, seq_len, time_steps):
    # 对应论文: β^k_t = 1 - α^k_t，其中 α^k_t = clamp(t - k/n_s, 0, 1)
    # 即 β^k_t = clamp(1 - (t - k/n_s), 0, 1) = clamp(1 + k/n_s - t, 0, 1)
    noise_level = torch.clamp(
        1
        + torch.arange(seq_len, device=device) / self.chunk_size   # k/n_s, 论文 n_s = chunk_size
        - time_steps.unsqueeze(1),                                 # t
        min=0.0,
        max=1.0,
    )
    return noise_level
```

- 这里 **`noise_level` 就是论文里的 \(\beta^k_t\)**（噪声权重）。
- `1 - noise_level` 即 \(\alpha^k_t\)（干净数据权重），与论文一致。

**（2）加噪**

```python
def add_noise(self, x, noise_level):
    noise = torch.randn_like(x)
    noise_level = noise_level.unsqueeze(-1)
    noisy_x = x * (1 - noise_level) + noise_level * noise   # α*x + β*noise
    return noisy_x, noise
```

- 对应论文 \(x_t = \alpha_t z + \beta_t \epsilon\)。

**（3）训练时时间 \(t\) 与 active window**

- 训练时每个 batch 随机一个 \(t\)（在合法范围内）：
  - `max_time = valid_len / self.chunk_size`，`t ~ Uniform(0, max_time)`。
- 对每个样本取 `end_index = int(chunk_size * t) + 1`，只把 `x[..., :end_index]` 送入 backbone，并在 **最后 `chunk_size` 个时间步** 上算 loss（论文：只在 active window 内算目标）。
- 对应论文：只在 \([m(t), n(t))\) 上算 velocity 的 MSE。

**（4）推理时的 ODE 步进（Algorithm 2）**

- `generate()` 里：`dt = 1 / num_denoise_steps`，`t` 从 0 逐步增加到 `max_t = 1 + (seq_len-1)/chunk_size`。
- 每一步：
  - `start_index = max(0, int(chunk_size * (t - 1)) + 1)`，
  - `end_index = int(chunk_size * t) + 1`，
  - 只对 `generated[:, start_index:end_index]` 用模型预测并更新（velocity 或 x0/noise 形式），其余不动。
- 这就是论文 Theorem 3.8 的 “streaming locality”：只更新当前 active window。

**对应小结**：

- **`chunk_size`** = 论文的 **\(n_s\)**（streaming step-size）。
- **`_get_noise_levels`** = 论文的 **\(\beta^k_t\)（及隐含的 \(\alpha^k_t\)）**。
- **`add_noise`** = 论文的 **\(x_t = \alpha_t z + \beta_t \epsilon\)**。
- 训练时只对「当前窗口末尾的 `chunk_size` 帧」算 loss；推理时只对 `[start_index:end_index]` 更新，与论文一致。

---

## 四、文本条件与“掩码”机制（论文重点之二）

### 4.1 论文描述（Section 3.4 & Figure 2）

- “**Continuous time-varying text conditioning**”：
  - 文本可以随时间变化（例如前一段是 “walk”，后一段是 “sit”）。
- “**Frame-wise text conditioning**” + “**attention mask**”：
  - 每个**运动帧**只允许 attend 到**当前时刻生效的那条文本**；
  - 即：第 \(k\) 帧只看到 \(c_k\)，不能看到其它时间段的文本。

这样做的目的：新文本到来时，模型能立刻根据**当前帧对应的文本**去去噪，而不被其它段的文本干扰，从而对齐“流式、按段切换”的设定。

### 4.2 代码实现方式（不是显式 mask，而是“每帧一个 context”）

论文里的 “each motion frame is only allowed to attend to the text prompt **active at that time**” 在代码里是通过 **为每一帧分配一个独立的 text embedding** 实现的，而不是在 attention 里写一个 0/1 的 mask 矩阵。

**（1）单段文本（例如 HumanML3D：整段一个描述）**

- `all_text_context` = 长度为 B 的 list，每个元素是对应样本的 T5 编码 `[L_text, 4096]`。
- 送入 WanModel 后，`context` 的 batch 维 = B，即**每个序列一个 context**，该序列内**所有帧**都 attend 到同一条文本。
- 对应论文中 “整段同一描述” 的情况。

**（2）多段 / 时间变化文本（例如 BABEL 或 stream：多段 [text1, text2, ...] + 每段结束帧）**

- 对每个样本：根据 `feature_text_end` 得到每段的时间范围，把该段的 T5 编码 **按帧数 repeat**：
  - “for u, duration in zip(single_text_context, single_text_length_list): all_text_context.extend([u for _ in range(duration)])”
- 这样 `all_text_context` 的长度 = **B × seq_len**（每个 (batch, frame) 一个 embedding）。
- 送入 WanModel 时，`context` 的 shape 为 `[B*seq_len, text_len, text_dim]`；在 DiT 的 cross-attention 里，**第 (b, l) 帧的 query 只和 context 的第 (b*seq_len+l) 行（即该帧对应的那条文本）做 attention**。
- 因此：**第 l 帧只会看到第 l 段对应的文本**，等价于论文说的 “biased mask” / frame-wise conditioning。

**（3）Classifier-Free Guidance（CFG）用的 “dropout”**

- 论文提到用 CFG 提升质量；训练时需要“有时不用文本”。
- 代码里用 **`drop_out`**（如 0.1）随机把文本条件丢掉：
  - 单段：`all_text_context[i] = "" if random > drop_out else text_list[i]`；
  - 多段：以概率 `drop_out` 把整段换成 `single_text_list = [""]`、`single_text_end_list = [0, seq_len]`，即整段都变成空文本。
- 推理时再用 `cfg_scale` 做 “有条件预测 - 无条件预测” 的插值，与论文一致。

**对应小结**：

- **Frame-wise 条件**：多段文本时，`all_text_context` 被展开成 **每帧一个 embedding**，cross-attention 结构上就实现了“每帧只看当前段文本”。
- **“掩码”**：不是额外的 attention mask 矩阵，而是 **context 的编排方式**（每帧一个 context 向量）天然限制了每帧只能 attend 到自己的文本。
- **CFG**：`drop_out` 训练时随机清空文本；推理时 `text_null_context` 与 `all_text_context` 用 `cfg_scale` 组合。

---

## 五、LDF 与 WanModel 的配合（简要）

- **DiffForcingWanModel**（`diffusion_forcing_wan.py`）负责：
  - 采样 \(t\)、算 `noise_level`、加噪、组好 **per-frame 的 all_text_context**；
  - 调用 **WanModel**（DiT backbone）做一次前向，得到对 velocity / x0 / noise 的预测；
  - 在 active window 上算 loss（训练）或更新 latent（推理）。
- **WanModel**（`models/tools/wan_model.py` 或 `wan_model_cross_rope.py`）负责：
  - 把 latent 当作 3D 的 (C, T, 1, 1) 做 patch embedding（1D 时间），然后 self-attention + **cross-attention to context** + FFN；
  - `context` 的 batch 维 = B 或 B*seq_len，由上面 “单段 / 多段” 决定，实现 frame-wise 文本条件。

因此：**噪声调度和 active window 在 DiffForcingWanModel 里实现；文本的“按帧绑定”在 DiffForcingWanModel 组好 context、WanModel 做 cross-attention 时实现。**

---

## 六、快速对照表（论文 ↔ 代码）

| 论文概念 | 代码位置 / 变量 |
|----------|------------------|
| 263D motion | 数据集 feature；VAE 输入/输出 |
| Causal VAE，4D latent，下采样 4× | `vae_wan_1d.py`，`z_dim=4`，`temperal_downsample` |
| \(\alpha^k_t, \beta^k_t\)，\(n_s\) | `_get_noise_levels`，`chunk_size`（n_s） |
| \(x_t = \alpha z + \beta \epsilon\) | `add_noise()` |
| Active window \(m(t), n(t)\) | `start_index`, `end_index`；训练时 `feature_ref` 的切分与 loss 只取最后 `chunk_size` |
| Frame-wise text conditioning | `all_text_context` 长度为 B×seq_len（多段时），WanModel cross-attention |
| CFG 训练 | `drop_out` 随机置空文本 |
| CFG 推理 | `cfg_scale`，`text_null_context` 与有条件预测的线性组合 |

把上述几点串起来，就能从“深度学习新手”视角把论文里的 **LDM 结构、噪声调度和文本掩码机制** 与当前代码对应起来；若要改或扩展（例如加轨迹控制），只需在 **LDF 的 condition 构建** 和 **WanModel 的输入** 上接续设计即可。
