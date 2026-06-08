# Phase 1：本地小模型基线（Rules + Para Classifier）

本文档说明 **HumOmni EmpathyEval Track 1** 阶段一已完成的工作、设计特点、产出物与优化方向。

阶段一目标：在 **不下载 Qwen2.5-Omni、不使用 API 处理测试集** 的前提下，搭建合规的本地推理流水线，产出可提交的 baseline，并为阶段二（Omni + LoRA）积累训练数据与特征缓存。

环境：`conda activate humomni`

---

## 1. 任务回顾

给定 **对话上下文**、**用户语音** 和若干 **候选回复语音**（A/B 或 A/B/C），选出最具共情力的选项。

| 测试集 | 任务类型 | 题数 | 选项 |
|--------|----------|------|------|
| `gigaspeech` | Context-variant（Task 1） | 200 | A/B |
| `meld` | Context-variant（Task 1） | 210 | A/B/C |
| `emovdb` | Tone-variant（Task 2） | 120 | A/B |
| **合计** | | **530** | |

评分公式（含 group bonus）：

```
Final = (Accuracy + Bonus) / (530 + 200) = (Acc + Bonus) / 730
```

竞赛合规要求：**测试集不得经过任何 API**（含预处理、推理、评估）。详见 [COMPLIANCE.md](COMPLIANCE.md)。

---

## 2. 架构总览

```
训练数据（API 可用）
  empatheticDialogue jsonl + goodPara/badPara 音频
      ├─ build_training_sft.py        → 4892 条 SFT 样本
      └─ train_para_classifier.py     → wav2vec2 二分类器

测试数据（仅本地）
  context + utterance_audio + option_audios
        │
        ▼
  Layer 1  特征提取（Whisper ASR / 语义相似度 / 韵律 / para 分类器）
        │
        ▼
  Layer 2  自适应规则打分 → opt-A / opt-B / opt-C
        │
        ▼
  export_submission.py → track1_submission.jsonl
```

阶段一 **不使用** Layer 3（Qwen API Judge）和本地 Omni 推理。

---

## 3. 核心组件

### 3.1 Layer 1：特征提取

每道题缓存为独立 JSON：`outputs/features/{dataset}/{question_id}.json`

| 字段 | 来源 | 作用 |
|------|------|------|
| `utterance_transcript` | Whisper `base` | 用户语音转写 |
| `options[*].transcript` | Whisper | 各候选回复转写 |
| `semantic_sim_to_reference` | sentence-transformers | 候选 ASR 与参考 `response` 文本的语义相似度 |
| `option_text_similarity` | 候选文本两两相似度均值 | 判断 A/B/C 是否「说的是同一件事」 |
| `naturalness_para_prob` | wav2vec2 分类器 | P(goodPara)，区分自然/机械韵律 |
| `naturalness_heuristic` | f0 / rms / tempo 等 | 无 checkpoint 时的韵律启发式 |
| `prosody` | librosa | 基频、能量、节奏等原始特征 |

**设计要点：** 当 `option_text_similarity ≥ 0.85` 时，候选文本几乎相同（如官方文档样例：同一句回复、A=机器人 B=真人），语义分无法区分，系统自动 **提高 naturalness 权重至 75%**。

### 3.2 Para 分类器（goodPara vs badPara）

- **骨干：** `facebook/wav2vec2-base`（冻结）+ 轻量 MLP 头
- **训练数据：** `empatheticDialogue_t_multi-context` + `empatheticDialogue_n_multi-emotion` 下的 `response_audio/`
- **标签：** 文件名含 `goodPara` → 1，`badPara` → 0
- **Checkpoint：** `checkpoints/para_classifier.pt`
- **验证集准确率：** **83.0%**（3 epoch，全量样本）

推理时，规则层直接使用 `naturalness_para_prob` 作为 naturalness 分数（有 checkpoint 时不再用 heuristic 组合分）。

### 3.3 Layer 2：自适应规则打分

配置见 `configs/pipeline.yaml` → `rules`：

| 场景 | semantic 权重 | naturalness 权重 |
|------|---------------|------------------|
| Task 1，候选文本差异大 | 0.60 | 0.40 |
| Task 1，候选文本高度相似 | 0.25 | 0.75 |
| Task 2（emovdb 语调变体） | 0.35 | 0.65 |

综合分：`combined = w_sem × semantic + w_nat × naturalness`，取最高者为预测。

### 3.4 预测与提交格式

**预测输出：** `outputs/predictions/{dataset}/predictions.jsonl`

```json
{
  "question_id": "meld_183_1",
  "choice": "opt-C",
  "source": "rules",
  "rule_scores": [
    {"option_key": "opt-C", "semantic": 0.817, "naturalness": 0.649, "combined": 0.691},
    {"option_key": "opt-A", "semantic": 0.962, "naturalness": 0.369, "combined": 0.517}
  ]
}
```

**官方提交：** `outputs/submission/track1_submission.jsonl`（530 行）

```json
{"question_id": "gigaspeech_0_1", "answer": "B"}
```

注意：`answer` 为 `A`/`B`/`C`，不是 `opt-A`。

---

## 4. 已完成工作

| 步骤 | 状态 | 产出 |
|------|------|------|
| 构建 SFT 训练集 | ✅ | `outputs/training/sft_train.jsonl`（4892 条） |
| 训练 para 分类器 | ✅ | `checkpoints/para_classifier.pt`（val_acc 83.0%） |
| Task 1 文档样例验证 | ✅ | 两题均选 **B**（与官方一致） |
| 全量测试集特征提取 | ✅ | `outputs/features/`（530 题） |
| 全量规则预测 | ✅ | `outputs/predictions/`（530 题） |
| 导出提交文件 | ✅ | `outputs/submission/track1_submission.jsonl` |

**本次 baseline 预测分布：**

| 数据集 | A | B | C |
|--------|---|---|---|
| gigaspeech | 97 | 103 | — |
| meld | 62 | 56 | 92 |
| emovdb | 63 | 57 | — |

---

## 5. 阶段一特点

### 优势

1. **完全合规**  
   测试集推理全程本地，无 API 调用，可直接用于正式提交。

2. **可解释**  
   每题有完整特征 JSON 和 `rule_scores` 分解，便于分析错例（语义 vs 韵律哪一项拉胯）。

3. **自适应权重**  
   根据 `option_text_similarity` 自动切换「语义主导」与「韵律主导」，覆盖文档样例（同文本不同音色）和常规题（文本本身有差异）。

4. **特征可复用**  
   `outputs/features/` 已缓存 530 题，阶段二无需重跑 Whisper / para，可直接作为 Omni 或 LLM 的输入上下文。

5. **训练资产可迁移**  
   SFT jsonl（4892 条）和 para checkpoint 与模型无关，租服务器后可直接用于 LoRA 微调，无需重建。

6. **Mac 可跑**  
   无需 GPU 大模型；全量 530 题约 30–40 分钟（特征缓存后更快）。

### 局限

1. **ASR 误差传播**  
   Whisper 对医学术语、口语缩读等转写不准，会拉低 `semantic_sim_to_reference`（如 meld_183_1 中 craniotomy 被误识）。

2. **Para 分类器粒度粗**  
   仅区分 goodPara/badPara（自然 vs 机械），无法区分「两个都是真人但共情程度不同」的难例。

3. **规则层无上下文理解**  
   不读 `context` 文本做推理，只靠 ASR + 参考回复 + 韵律；对「同一句话在不同情境下哪个更合适」判断有限。

4. **三选项 meld 更难**  
   A/B/C 中常有多个高语义分选项，仅靠 para_prob 区分，容易在 C 上过度集中（本次 92/210 选 C）。

5. **无 group 一致性约束**  
   同一 `group_id` 下各题独立预测，未利用「同组应风格一致」的先验，可能损失 Bonus 分。

---

## 6. 优化建议

### 6.1 短期（仍在阶段一框架内）

| 方向 | 做法 | 预期收益 |
|------|------|----------|
| **调权重** | 在 `configs/pipeline.yaml` 中微调 `similar_text_*` / `tone_variant_*` 权重 | 针对 meld/emovdb 错例集中领域 |
| **升级 Whisper** | `whisper_model: small` 或 `medium` | 减少 ASR 误差对语义分的影响 |
| **Para 分类器重训** | 加 epoch、调 `learning_rate`、或换 `wav2vec2-large` | 提升韵律判别边界 |
| **错例分析** | 按 `rule_scores` 排序，找 combined 差距 < 0.05 的「胶着题」 | 定位规则失效模式 |
| **跳过 ASR 做对照** | `skip_asr: true`，直接用测试 JSON 里的 `utterance` / `response` 文本 | 量化 ASR 对最终准确率的影响 |

### 6.2 中期（训练数据 + 弱监督）

| 方向 | 做法 | 预期收益 |
|------|------|----------|
| **API 标注训练集** | `generate_training_labels.py`（仅 `empatheticDialogue_*`） | 获得高质量共情标签，供 LoRA 或蒸馏 |
| **特征 + 文本微调小分类器** | 用 4892 条 SFT + 已提取特征训练轻量 MLP / XGBoost | 比纯规则更强的本地判别器 |
| **Group 一致性后处理** | 同 `group_id` 内多数投票或约束同一选项 | 提升 Bonus 分 |
| **难例语义嵌入** | 将 `context` 编码进 embedding，而不只比 candidate vs reference | 改善 Task 1 情境适配 |

### 6.3 长期（阶段二：Omni + LoRA）

| 方向 | 做法 | 预期收益 |
|------|------|----------|
| **Qwen2.5-Omni-3B 本地推理** | `run_submission.py --backend omni` | 端到端多模态理解，显著超越规则基线 |
| **LoRA 微调** | `train_omni_lora.py` on SFT + API 标签 | 针对共情语音选择任务专项优化 |
| **特征作为 prompt 上下文** | 将 Layer 1 JSON 摘要注入 Omni prompt | 结合小模型可解释信号与大模型推理 |
| **集成** | rules 与 Omni 分歧时触发二次判断 | 兼顾合规、可解释与准确率 |

---

## 7. 复现命令

```bash
conda activate humomni
cd /path/to/HumOmni_EmpathyEval

# 1. 构建训练 SFT（若尚未生成）
python scripts/build_training_sft.py

# 2. 训练 para 分类器
python scripts/train_para_classifier.py

# 3. 验证官方文档样例（应两题均选 B）
python scripts/demo_task1_example.py

# 4. 单题调试（查看特征 + 规则分）
python scripts/predict.py --dataset meld --question-id meld_183_1 --no-qwen --force-features

# 5. 全量预测（建议按数据集分批前台运行，避免后台进程被中断）
python scripts/predict.py --no-qwen --datasets gigaspeech
python scripts/predict.py --no-qwen --datasets meld
python scripts/predict.py --no-qwen --datasets emovdb

# 6. 导出提交
python scripts/export_submission.py
# → outputs/submission/track1_submission.jsonl
```

或使用编排脚本（需确保进程不被中断）：

```bash
python scripts/run_submission.py --backend rules --skip-para-train
```

---

## 8. 产出物索引

```
outputs/
├── training/
│   └── sft_train.jsonl              # 4892 条训练样本（阶段二 LoRA 输入）
├── features/
│   └── {dataset}/{question_id}.json # 530 题 Layer 1 特征缓存
├── predictions/
│   └── {dataset}/predictions.jsonl  # 530 题规则预测 + 分项得分
└── submission/
    └── track1_submission.jsonl      # 530 行官方提交格式

checkpoints/
└── para_classifier.pt               # goodPara 分类器（val_acc 83.0%）

Task 1: Audio Supplementation/
└── demo_predictions.json            # 文档样例验证结果
```

---

## 9. 与阶段二的衔接

阶段一产出的以下内容 **无需重做**，可直接进入阶段二：

| 资产 | 阶段二用途 |
|------|------------|
| `sft_train.jsonl` | LoRA 微调数据集 |
| `outputs/features/` | Omni prompt 附加上下文 / 难例分析 |
| `para_classifier.pt` | 继续作为本地特征或集成信号 |
| `track1_submission.jsonl` | 规则基线成绩对照 |

阶段二推荐路径（详见 [README_PHASE2.md](README_PHASE2.md)）：

```bash
# 训练集 API 标注（合规）
python scripts/generate_training_labels.py

# 服务器上 LoRA 微调 + Omni 推理
python scripts/train_omni_lora.py
python scripts/run_submission.py --backend omni
```

---

## 10. 相关文档

- [README.md](README.md) — 项目总览与完整命令
- [COMPLIANCE.md](COMPLIANCE.md) — 竞赛合规规则
- [configs/pipeline.yaml](configs/pipeline.yaml) — 特征提取与规则权重配置
