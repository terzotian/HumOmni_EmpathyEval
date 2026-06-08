# HumOmni EmpathyEval

Track 1 pipeline for **contextualized affective speech** response selection.

Given a conversation **context**, a user **utterance audio**, and candidate **response audios** (A/B or A/B/C), the system selects the most empathetic option.

Works in the conda environment **`humomni`**.

---

## Project status（截至 2025-06-09）

| 阶段 | 状态 | 说明 |
|------|------|------|
| **阶段一** Rules + Para Classifier | ✅ **已完成** | 全量 530 题预测 + 提交文件已生成 |
| **阶段二** Omni + LoRA | 📋 **未开始** | 代码与配置已搭好，方案见 [README_PHASE2.md](README_PHASE2.md) |

### 阶段一已产出（可直接使用）

| 产出 | 路径 | 数量 |
|------|------|------|
| SFT 训练集 | `outputs/training/sft_train.jsonl` | 4892 条 |
| Para 分类器 | `checkpoints/para_classifier.pt` | val_acc **83.0%** |
| 测试集特征缓存 | `outputs/features/{dataset}/` | 530 题 |
| 规则预测 | `outputs/predictions/{dataset}/predictions.jsonl` | 530 题 |
| **提交文件** | `outputs/submission/track1_submission.jsonl` | **530 行** |
| 文档样例验证 | `Task 1: Audio Supplementation/demo_predictions.json` | 两题均选 **B** ✓ |

阶段一 baseline 预测分布：`gigaspeech` 97A/103B · `meld` 62A/56B/92C · `emovdb` 63A/57B

### 阶段二尚未执行

| 计划步骤 | 状态 |
|----------|------|
| API 训练标注（`generate_training_labels.py`） | 未运行 |
| LoRA 微调（`train_omni_lora.py`） | 未运行，`checkpoints/omni_lora/` 不存在 |
| 本地 Omni 推理（`run_omni_inference.py`） | 未运行 |

详细方案与参数备忘 → [README_PHASE2.md](README_PHASE2.md)  
阶段一复盘与优化建议 → [README_PHASE1.md](README_PHASE1.md)

---

## Task overview

| Split | Dataset | Task | Questions | Options |
|-------|---------|------|-----------|---------|
| Test | `gigaspeech` | Context-variant (Task 1) | 200 | A/B |
| Test | `meld` | Context-variant (Task 1) | 210 | A/B/C |
| Test | `emovdb` | Tone-variant (Task 2) | 120 | A/B |

Training data (under `data/`):

- Text: `empatheticDialogue_t_multi-context_flat.jsonl`, `empatheticDialogue_n_multi-emotion_flat.jsonl`
- Audio: `goodPara` vs `badPara` response pairs + `user_audio`

---

## 今日工作摘要（2025-06-09）

### 1. 代码与基础设施（已搭建）

```
empathy_eval/
├── compliance.py          # 竞赛合规守卫（禁止 test 集走 API）
├── data.py                # 测试集加载
├── training_data.py       # 训练集 goodPara/badPara 解析
├── submission.py          # opt-A → A 格式转换
├── audio.py
├── features/              # Layer 1：ASR / 语义 / 韵律 / para 分类
├── pipeline/              # Layer 2：规则打分；Layer 3：Qwen API judge（仅训练/调试）
└── omni/                  # 阶段二：Omni 推理 prompt + inference（代码就绪，未跑全量）

scripts/
├── build_training_sft.py       # 构建 SFT 训练集
├── train_para_classifier.py    # wav2vec2 goodPara 分类器
├── predict.py                  # 阶段一推理（--no-qwen）
├── extract_features.py         # 单独提取特征
├── export_submission.py        # 合并三数据集 → 提交 jsonl
├── run_submission.py           # 端到端编排
├── demo_task1_example.py       # 官方文档样例验证
├── check_api_config.py         # DashScope API 连通性测试
├── generate_training_labels.py # 阶段二：API 标注（仅训练集）
├── train_omni_lora.py          # 阶段二：LoRA 微调（未执行）
└── run_omni_inference.py       # 阶段二：本地 Omni 推理（未执行）

configs/
├── pipeline.yaml          # 阶段一：特征提取 + 规则权重
└── omni.yaml              # 阶段二：Omni + LoRA + API 标注配置
```

### 2. 阶段一执行（已完成）

1. `build_training_sft.py` → 4892 条训练样本（3020 context + 1872 tone）
2. `train_para_classifier.py` → 3 epoch，best val_acc = 83.0%
3. `predict.py --no-qwen` → 全量 530 题（gigaspeech / meld / emovdb 分批前台运行）
4. `export_submission.py` → `track1_submission.jsonl`
5. `demo_task1_example.py` → 文档样例 Context 1/2 均选 B

### 3. 阶段二（仅方案，未执行）

- 脚本与 `configs/omni.yaml` 已写好，供租服务器后按 [README_PHASE2.md](README_PHASE2.md) 执行
- DashScope API key 已配置（`check_api_config.py` 通过），但 **未对训练集跑 API 标注**

---

## Architecture

### 阶段一（当前提交方案）✅

```
context + utterance_audio + option_audios
        │
        ▼
 Layer 1  Whisper ASR / 语义相似度 / 韵律启发式 / para 分类器
        │
        ▼
 Layer 2  自适应规则打分（按 option_text_similarity 切换权重）
        │
        ▼
 export_submission.py → track1_submission.jsonl
```

### 阶段二（计划，未开始）📋

```
阶段一产出（sft_train.jsonl、features/、para_classifier.pt）
        │
        ├─ [可选] generate_training_labels.py  （API，仅训练集）
        ├─ train_omni_lora.py                （GPU 服务器）
        └─ run_omni_inference.py             （本地 Omni，测试集无 API）
                │
                ▼
        export_submission.py → track1_submission.jsonl
```

See [COMPLIANCE.md](COMPLIANCE.md) for rules enforced in code.

---

## Quick start

All commands assume `conda activate humomni`.

### 当前可用：阶段一复现 / 重新提交

```bash
# 验证文档样例（应两题均选 B）
python scripts/demo_task1_example.py

# 单题调试
python scripts/predict.py --dataset meld --question-id meld_183_1 --no-qwen

# 全量预测（建议按数据集分批前台运行）
python scripts/predict.py --no-qwen --datasets gigaspeech
python scripts/predict.py --no-qwen --datasets meld
python scripts/predict.py --no-qwen --datasets emovdb

# 导出提交（已有 530 行可直接上传）
python scripts/export_submission.py
```

当前提交文件：`outputs/submission/track1_submission.jsonl`

### 阶段二（待执行，见 README_PHASE2.md）

```bash
# 租 GPU 服务器后
python scripts/train_omni_lora.py
python scripts/run_omni_inference.py
python scripts/export_submission.py
```

---

## Environment setup

### 1. Create or update the environment

```bash
cd /path/to/HumOmni_EmpathyEval
conda env update -f environment.yml --prune
conda activate humomni
```

### 2. Optional: ffmpeg (for whisper CLI only; pipeline uses soundfile)

```bash
brew install ffmpeg
```

### 3. API key（阶段二训练标注用，测试集禁用）

```bash
cp .env.example .env
# edit .env and set DASHSCOPE_API_KEY

python scripts/check_api_config.py   # 仅 ping 测试，不碰测试集
```

### 4. Download data

```bash
python scripts/download_dataset.py
```

---

## Key commands reference

| 用途 | 命令 |
|------|------|
| 构建 SFT 训练集 | `python scripts/build_training_sft.py` |
| 训练 para 分类器 | `python scripts/train_para_classifier.py` |
| 规则推理（合规提交） | `python scripts/predict.py --no-qwen` |
| 导出提交 | `python scripts/export_submission.py` |
| 文档样例验证 | `python scripts/demo_task1_example.py` |
| 单独提特征 | `python scripts/extract_features.py --datasets meld --limit 5` |
| 端到端编排（阶段一） | `python scripts/run_submission.py --backend rules --skip-para-train` |

**注意：** `predict.py --use-qwen` 和 `demo_task1_example.py --use-qwen` 会调用 API，**不得用于测试集正式提交**。测试集提交只用 `--no-qwen` 或 `run_omni_inference.py`。

---

## Submission format

`outputs/submission/track1_submission.jsonl` — 530 lines, one json per line:

```json
{"question_id": "gigaspeech_0_1", "answer": "A"}
{"question_id": "meld_183_1", "answer": "C"}
```

- `answer` is `A` / `B` / `C`（not `opt-A`）
- `question_id` must match test JSON exactly
- Upload **one** file covering all three subsets

---

## Competition rules

### No API on test data

Test data must **not** be used in any API-related process. APIs may only be used for **training**.

- ✅ `predict.py --no-qwen` on test sets
- ✅ `run_omni_inference.py` (local Omni)
- ❌ `predict.py --use-qwen` on test sets for submission
- ❌ `generate_training_labels.py` with test paths

### Scoring (Track 1)

```
Final = (Accuracy + Bonus) / (530 + 200) = (Acc + Bonus) / 730
```

Groups: gigaspeech (100×2) + meld (70×3) + emovdb (30×4) = 200 groups

Test JSON has **no answer labels**; organizers score against hidden human annotations.

---

## Configuration

| File | 用途 | 阶段 |
|------|------|------|
| [`configs/pipeline.yaml`](configs/pipeline.yaml) | Whisper、embedding、para checkpoint、规则权重 | 一 ✅ |
| [`configs/omni.yaml`](configs/omni.yaml) | Omni 模型、LoRA、API 标注 | 二 📋 |
| [`configs/dataset.yaml`](configs/dataset.yaml) | Hugging Face 数据下载 | — |

---

## Project layout

```
HumOmni_EmpathyEval/
├── README.md                 # 本文件（项目总览 + 进度）
├── README_PHASE1.md          # 阶段一详细说明（已完成）
├── README_PHASE2.md          # 阶段二方案备忘（未开始）
├── COMPLIANCE.md             # 竞赛合规
├── configs/
│   ├── pipeline.yaml
│   ├── omni.yaml
│   └── dataset.yaml
├── empathy_eval/
│   ├── compliance.py
│   ├── data.py
│   ├── training_data.py
│   ├── submission.py
│   ├── audio.py
│   ├── features/             # Layer 1
│   ├── pipeline/             # Layer 2 & 3
│   └── omni/                 # Phase 2（代码就绪）
├── scripts/                  # 见上方「今日工作摘要」
├── data/                     # gitignored
├── outputs/                  # gitignored
│   ├── training/sft_train.jsonl
│   ├── features/             # 530 题
│   ├── predictions/          # 530 题（source=rules）
│   └── submission/track1_submission.jsonl
└── checkpoints/
    └── para_classifier.pt    # 阶段一 checkpoint
```

---

## Decision logic（阶段一，当前提交）

1. **Semantic fit** — candidate ASR vs reference `response`（sentence embeddings）
2. **Naturalness** — `naturalness_para_prob` from goodPara classifier（有 checkpoint 时）
3. **Adaptive weights**（`configs/pipeline.yaml` → `rules`）:
   - 候选文本差异大 → semantic 60% / naturalness 40%
   - 候选文本高度相似（≥ 0.85）→ semantic 25% / naturalness 75%
   - Task 2 emovdb → semantic 35% / naturalness 65%

---

## Next steps

| 优先级 | 动作 | 阶段 |
|--------|------|------|
| 现在可做 | 上传 `track1_submission.jsonl` 作为 rules baseline | 一 |
| 现在可做 | 按 `rule_scores` 分析错例、调 `pipeline.yaml` 权重 | 一 |
| 后续 | 租 GPU → LoRA 微调 → Omni 全量推理 | 二 |
| 后续 | API 标注训练集（`generate_training_labels.py`） | 二 |

---

## Dependencies

See `environment.yml`: `pytorch`, `transformers`, `peft`, `openai-whisper`, `sentence-transformers`, `qwen-omni-utils`, `dashscope`, `librosa`, `soundfile`, etc.

---

## Citation

Dataset: [gracehuggingface/EmpathyEval](https://huggingface.co/datasets/gracehuggingface/EmpathyEval)

Paper: [AEQ-Bench: Measuring Empathy of Omni-Modal Large Models](https://arxiv.org/abs/2601.10513)
