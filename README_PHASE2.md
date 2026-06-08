# Phase 2：Qwen2.5-Omni 本地推理 + LoRA 微调

> **状态：📋 未开始（方案备忘）**  
> 截至 2025-06-09，阶段二脚本与配置已搭建，但 **尚未执行** LoRA 训练、API 标注或 Omni 全量推理。本文档供后续租服务器时参考，避免遗忘参数与流程。

本文档说明 **HumOmni EmpathyEval Track 1** 阶段二的计划工作内容、关键配置参数、推荐流程与优化方向。

阶段二目标：在 **合规前提下**（测试集仅本地推理），用 **Qwen2.5-Omni** 做多模态共情选择，并通过 **LoRA 微调** 和可选的 **API 训练标注** 提升准确率，最终产出正式提交文件。

前置条件：阶段一已完成（见 [README_PHASE1.md](README_PHASE1.md)）。可复用 `sft_train.jsonl`、`outputs/features/`、`para_classifier.pt`。

环境：`conda activate humomni`（阶段二额外依赖 `peft`、`qwen-omni-utils`，见 `environment.yml`）

---

## 1. 阶段定位

| 维度 | 阶段一 | 阶段二 |
|------|--------|--------|
| 推理模型 | Whisper + 规则 + para 分类器 | Qwen2.5-Omni-3B（+ LoRA） |
| 是否理解 context | 弱（规则层几乎不用） | 强（prompt 含完整上下文） |
| 是否听音频 | para 韵律特征 | 端到端多模态 |
| 训练方式 | wav2vec2 二分类 | LoRA on Omni + 可选 API 标注 |
| 硬件 | Mac CPU/MPS 即可 | 建议租 GPU 服务器做 LoRA；Mac 48G 可跑 3B 推理 |
| 测试集 API | 禁止 | 禁止（与阶段一相同） |

---

## 2. 整体流程

```
阶段一产出（可复用，无需重做）
  outputs/training/sft_train.jsonl     # 4892 条 A/B 训练对
  outputs/features/                    # 可选，供 prompt 增强 / 错例分析
  checkpoints/para_classifier.pt         # 可选，供集成或分析

阶段二新增
  ┌─ [可选] API 标注训练集 ─────────────────────────────┐
  │  generate_training_labels.py                        │
  │  → outputs/training/api_labels.jsonl                │
  └─────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─ LoRA 微调（仅训练数据）─────────────────────────────┐
  │  train_omni_lora.py                                 │
  │  → checkpoints/omni_lora/                           │
  └─────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─ 本地 Omni 推理（测试集，无 API）────────────────────┐
  │  run_omni_inference.py                              │
  │  → outputs/predictions/{dataset}/predictions.jsonl  │
  └─────────────────────────────────────────────────────┘
                          │
                          ▼
  export_submission.py → outputs/submission/track1_submission.jsonl
```

一键编排：

```bash
python scripts/run_submission.py --backend omni
```

---

## 3. 工作内容详解

### 3.1 训练数据准备（阶段一已完成，可跳过）

**脚本：** `scripts/build_training_sft.py`

从 `empatheticDialogue_*` 的 `goodPara` / `badPara` 音频对构建监督样本：

| 字段 | 说明 |
|------|------|
| `task` | `context_variant`（3020 条）或 `tone_variant`（1872 条） |
| `context` / `utterance_text` / `reference_response` | 文本上下文 |
| `utterance_audio` | 用户语音 |
| `options` | `opt-A`、`opt-B` 两条候选回复音频 |
| `answer` | `A` 或 `B`（goodPara 所在侧，随机打乱） |

产出：`outputs/training/sft_train.jsonl`（**4892 条**）

> **注意：** 训练集只有 **A/B 二选一**；测试集 `meld` 有 **A/B/C 三选一**。LoRA 未专门训练三选项场景，需在 prompt 或数据增强上弥补（见 §6）。

### 3.2 API 训练标注（可选，合规：仅训练数据）

**脚本：** `scripts/generate_training_labels.py`  
**配置：** `configs/omni.yaml` → `api_training`

在已知 `goodPara` 标签的前提下，调用 DashScope（`qwen-plus` 等）为训练样本生成 **共情理由**（rationale），用于：

- 丰富 LoRA 训练 prompt（需自行接入 rationale 字段）
- 人工审核训练质量
- 未来做 CoT / 蒸馏

产出：`outputs/training/api_labels.jsonl`

```json
{
  "example_id": "...",
  "answer": "B",
  "rationale": "The good option matches the user's frustration with warmer tone...",
  "paths": [".../user_audio/...", ".../opt-A.wav", ".../opt-B.wav"]
}
```

**合规：** 脚本内置 `assert_api_training_only`，传入 `phase1-test` / `gigaspeech` / `meld` / `emovdb` 路径会直接报错。

### 3.3 LoRA 微调

**脚本：** `scripts/train_omni_lora.py`  
**配置：** `configs/omni.yaml` → `train` + `model_id`

训练时构造多模态对话：

```
[text] instruction + context
[audio] utterance
[text] Candidate opt-A:
[audio] opt-A
[text] Candidate opt-B:
[audio] opt-B
→ 监督标签：answer（A 或 B）
```

技术要点：

- 基座：`Qwen/Qwen2.5-Omni-3B`，`enable_audio_output=False`（关闭语音生成，省约 2GB 显存）
- LoRA 注入：`q_proj`、`k_proj`、`v_proj`、`o_proj`
- 仅对 **answer token** 计算 loss（`labels` 其余位置为 -100）
- 产出目录：`checkpoints/omni_lora/`（含 adapter + processor + `train_meta.json`）

微调完成后，在 `configs/omni.yaml` 设置：

```yaml
lora_adapter: checkpoints/omni_lora
```

### 3.4 本地 Omni 推理（测试集提交）

**脚本：** `scripts/run_omni_inference.py`  
**核心类：** `empathy_eval/omni/inference.py` → `OmniEmpathyJudge`

对每道测试题：

1. 用 `build_selection_instruction()` 构造 prompt（含 context、utterance 文本提示、reference response、候选列表）
2. 依次附加 **用户语音** + 各候选 **回复语音**（支持 A/B/C）
3. `model.generate()` 生成短回答，`max_new_tokens` 默认 16
4. 正则解析输出中的 `A` / `B` / `C` → `opt-A` / `opt-B` / `opt-C`

产出：`outputs/predictions/{dataset}/predictions.jsonl`（`source: omni_local`）

### 3.5 导出提交

```bash
python scripts/export_submission.py
# → outputs/submission/track1_submission.jsonl（530 行）
```

可用阶段二预测 **覆盖** 阶段一 rules 预测后重新导出；`export_submission.py` 会合并三个测试集。

---

## 4. 关键配置参数（`configs/omni.yaml`）

### 4.1 推理相关

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `model_id` | `Qwen/Qwen2.5-Omni-3B` | HuggingFace 模型 ID | GPU 充足可试 `7B`；**换模型必须重训 LoRA** |
| `device` | `auto` | `auto` / `mps` / `cuda` / `cpu` | 服务器用 `cuda`；Mac 用 `mps`；OOM 时改 `cpu`（极慢） |
| `max_new_tokens` | `16` | 生成长度 | 模型偶尔输出多余文字时可降到 `8`；解析失败可略增到 `32` |
| `lora_adapter` | `null` | LoRA 目录 | 微调后设为 `checkpoints/omni_lora` |
| `datasets` | 三个测试集 | 推理范围 | 调试时可只留一个数据集 |

### 4.2 LoRA 训练相关（`train:`）

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `sft_dataset` | `outputs/training/sft_train.jsonl` | 训练 jsonl 路径 | 一般不需改 |
| `tasks` | `[context_variant, tone_variant]` | 纳入哪些任务 | 可只训 `tone_variant` 专攻 emovdb |
| `limit` | `null` | 训练样本上限 | **首次 dry-run 设 `100~200`**，确认能跑通再 `null` 全量 |
| `lora_r` | `8` | LoRA 秩 | 欠拟合 ↑ 到 `16`/`32`；过拟合 ↓ 到 `4` |
| `lora_alpha` | `16` | LoRA 缩放 | 通常设为 `2 × lora_r` |
| `lora_dropout` | `0.05` | Dropout | 过拟合时 ↑ 到 `0.1` |
| `epochs` | `1` | 训练轮数 | 全量 4892 条可试 `2~3`；注意过拟合 |
| `batch_size` | `1` | 每步样本数 | 受显存限制，多模态音频通常只能 1 |
| `learning_rate` | `0.0002` | AdamW 学习率 | 不稳定 ↓ `1e-4`；收敛慢 ↑ `3e-4`（谨慎） |
| `gradient_accumulation_steps` | `8` | 梯度累积 | 等效 batch=8；显存紧时可 ↑ 到 `16` |
| `output_dir` | `checkpoints/omni_lora` | 保存路径 | 多次实验建议加版本后缀 |

### 4.3 API 训练标注（`api_training:`）

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `enabled` | `false` | 是否允许跑 API 脚本 | 跑 `generate_training_labels.py` 前改 `true` |
| `model` | `qwen-plus` | DashScope 模型 | 质量优先 `qwen-max`；成本优先 `qwen-turbo` |
| `limit` | `50` | API 调用条数 | 先小批量验证 prompt，再扩大到全量 |
| `temperature` | `0.1` | 采样温度 | 标注任务保持低温度 |
| `output` | `outputs/training/api_labels.jsonl` | 输出路径 | — |

### 4.4 Prompt 相关（代码内，非 yaml）

文件：`empathy_eval/omni/prompts.py`

| 函数 | 用途 |
|------|------|
| `build_selection_instruction()` | 测试推理 prompt |
| `build_training_instruction()` | LoRA 训练 prompt |

优化 prompt 是阶段二 **性价比最高** 的手段之一（不改训练即可影响推理）。

---

## 5. 硬件与环境建议

| 场景 | 最低配置 | 推荐配置 |
|------|----------|----------|
| Omni-3B 推理（单题） | Mac MPS 16GB+ | Mac M4 48GB / 单卡 24GB GPU |
| Omni-3B LoRA 全量（4892 条） | 不推荐 Mac | A100 40GB / A10 24GB（gradient_accum=8） |
| Omni-7B | 48GB 统一内存勉强 | 多卡或 80GB GPU |

首次运行会从 HuggingFace 下载 Omni-3B（数 GB），确保网络与磁盘空间充足。

依赖安装：

```bash
conda activate humomni
conda env update -f environment.yml --prune
```

验证 API（仅训练用途）：

```bash
python scripts/check_api_config.py
```

---

## 6. 特点与局限

### 优势

1. **端到端多模态**：直接听用户语音与候选回复，不依赖 Whisper ASR 转写。
2. **理解 context**：prompt 含完整对话背景，弥补阶段一规则层短板。
3. **合规提交路径**：`run_omni_inference.py` + `compliance.py` 保证测试集不经 API。
4. **LoRA 轻量微调**：只训少量 adapter 参数，比全参微调省显存。
5. **阶段一资产可叠加**：特征 JSON、para 分数、API rationale 可注入 prompt 做集成。

### 局限与风险

| 风险 | 说明 | 缓解 |
|------|------|------|
| **训练只有 A/B，测试 meld 有 C** | LoRA 未见过三选项 | 强化 prompt 规则；或构造伪三选项训练数据 |
| **LoRA 与模型版本绑定** | 3B → 7B 需重训 | 确定模型后再训 LoRA |
| **解析失败** | 模型输出非单字母 | 调 `max_new_tokens`、改 prompt「ONLY one letter」、加 fallback |
| **显存 OOM** | 多段音频拼接很长 | `batch_size=1`、关 `audio_output`、用 3B 而非 7B |
| **Mac 训练极慢** | 4892 条 LoRA 不现实 | 租 GPU；或 `train.limit` 子集 + 多 epoch |
| **无 group 一致性** | 同组独立预测 | 后处理多数投票（与阶段一相同） |

---

## 7. 优化建议

### 7.1 Prompt 优化（零训练成本）

- 在 `build_selection_instruction()` 中注入阶段一特征摘要，例如：
  - `option_text_similarity` 高时强调「听韵律而非措辞」
  - 附上 `naturalness_para_prob` 作为 hint（注意：这是训练集上学来的信号，测试集可用本地特征提取，**不经过 API**）
- 针对 Task 2（emovdb）增加「语调与情绪匹配」指令
- 针对 meld 三选项明确「三个候选均需比较」

### 7.2 LoRA 训练优化

| 目标 | 建议 |
|------|------|
| 快速验证 | `train.limit: 200`，`epochs: 2`，观察 loss 下降 |
| 提升泛化 | 全量 4892 条，`lora_r: 16`，`epochs: 2~3`，早停 |
| 专攻 emovdb | `tasks: [tone_variant]` 单独训一版 LoRA，推理时按 dataset 切换 adapter |
| 利用 API 标签 | 将 `api_labels.jsonl` 的 rationale 拼入训练 instruction，做 CoT 式 SFT |
| 难例重采样 | 对 rules 与 Omni 预测不一致的题（若有 dev 集）过采样 |

### 7.3 推理优化

| 目标 | 建议 |
|------|------|
| 提速 | 先 `limit: 5` smoke test；全量分批 `--datasets gigaspeech` 等 |
| 稳定性 | `do_sample=False`（已默认）；降低 `max_new_tokens` |
| 准确率 | 加载 LoRA；对比 base vs LoRA 在文档样例上的输出 |
| 集成 | rules 与 Omni 一致则采纳；分歧时以 Omni 为准或触发二次 prompt |

### 7.4 数据优化

- **三选项增强：** 从 meld 测试集结构出发，用训练集 goodPara 音频 + 合成/裁剪构造「伪 A/B/C」样本（仍须来自训练许可数据）
- **任务平衡：** `context_variant` 与 `tone_variant` 样本数约 6:4，可按测试集比例 380:120 加权采样
- **答案打乱：** 训练时已 `shuffle_bad_good`，可固定 `seed: 42` 复现

### 7.5 评估与迭代（无官方 test label）

- 用 `Task 1: Audio Supplementation/` 文档样例（答案 B）做 smoke test
- 对比阶段一与阶段二预测差异：`diff` 两个 `predictions.jsonl`
- 关注 **group bonus**：按 `group_id` 统计组内是否一致

---

## 8. 推荐工作流

### Step 0：确认阶段一产出

```bash
ls outputs/training/sft_train.jsonl    # 应有 4892 行
ls checkpoints/para_classifier.pt    # 可选
```

### Step 1：API 标注（可选）

```bash
# configs/omni.yaml → api_training.enabled: true
python scripts/generate_training_labels.py --limit 50
```

### Step 2：LoRA dry-run（服务器）

```bash
# configs/omni.yaml → train.limit: 200
python scripts/train_omni_lora.py
```

### Step 3：LoRA 全量训练

```bash
# configs/omni.yaml → train.limit: null
python scripts/train_omni_lora.py
# 完成后设置 lora_adapter: checkpoints/omni_lora
```

### Step 4：单题 smoke test

```bash
python scripts/run_omni_inference.py --dataset meld --question-id meld_183_1
python scripts/demo_task1_example.py   # 若有对应 Omni 入口可对比
```

### Step 5：全量推理 + 提交

```bash
python scripts/run_omni_inference.py
python scripts/export_submission.py
```

或：

```bash
python scripts/run_submission.py --backend omni --skip-para-train
```

---

## 9. 产出物索引

### 已有（阶段一提供，可直接用）

| 文件 | 状态 |
|------|------|
| `outputs/training/sft_train.jsonl` | ✅ 4892 条 |
| `outputs/features/` | ✅ 530 题 |
| `checkpoints/para_classifier.pt` | ✅ val_acc 83.0% |
| `configs/omni.yaml` | ✅ 已配置 |

### 待生成（阶段二执行后才有）

| 文件 | 状态 |
|------|------|
| `outputs/training/api_labels.jsonl` | ❌ 未生成 |
| `checkpoints/omni_lora/` | ❌ 未生成 |
| `outputs/predictions/`（source=omni_local） | ❌ 未生成 |
| `outputs/submission/track1_submission.jsonl`（Omni 版） | ❌ 当前为 rules 版 |

---

## 10. 与阶段一的对照实验建议

完成阶段二推理后，建议做一张简单对照表：

| 指标 | 阶段一 rules | 阶段二 Omni | 阶段二 Omni+LoRA |
|------|-------------|-------------|------------------|
| 文档样例 | B / B | ? | ? |
| gigaspeech 选项分布 | 97A / 103B | — | — |
| meld 选项分布 | 62A / 56B / 92C | — | — |
| emovdb 选项分布 | 63A / 57B | — | — |
| 与 rules 不一致题数 | — | — | — |

阶段一的 `outputs/features/` 和 `rule_scores` 可用来解释 Omni 错例：是语义、韵律还是 context 理解问题。

---

## 11. 相关文档

- [README.md](README.md) — 项目总览
- [README_PHASE1.md](README_PHASE1.md) — 阶段一基线说明
- [COMPLIANCE.md](COMPLIANCE.md) — 竞赛合规规则
- [configs/omni.yaml](configs/omni.yaml) — 阶段二配置
- [configs/pipeline.yaml](configs/pipeline.yaml) — 阶段一特征 / 规则配置
