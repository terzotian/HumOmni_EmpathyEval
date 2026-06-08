# Competition Compliance Guide

## Rule (Track 1)

> Test data must **not** be used in any API-related process, including preprocessing,
> intermediate processing, inference, result prediction, or evaluation.
> APIs may only be used for **model training** purposes.

## Allowed

| Action | Data | Tool |
|--------|------|------|
| Local inference | gigaspeech / meld / emovdb | `run_omni_inference.py`, `predict.py --no-qwen` |
| Train para classifier | training `response_audio` | `train_para_classifier.py` |
| Build SFT dataset | training jsonl + audio | `build_training_sft.py` |
| API label enrichment | training jsonl + audio only | `generate_training_labels.py` |
| LoRA fine-tune | training audio pairs | `train_omni_lora.py` |
| Export submission | local predictions | `export_submission.py` |

## Forbidden

| Action | Why |
|--------|-----|
| `predict.py --use-qwen` on test sets | API + test data |
| `generate_training_labels.py` with test paths | API + test data |
| Sending test audio/text to DashScope | API + test data |

## Enforcement in code

- `empathy_eval/compliance.py` — guards for API vs local paths
- `qwen_judge.py` — raises `ComplianceError` on test datasets
- `generate_training_labels.py` — only accepts `empatheticDialogue_*` paths
- `run_submission.py` — local backends only

## Recommended submission command

```bash
conda activate humomni
python scripts/run_submission.py --backend omni
```

Upload: `outputs/submission/track1_submission.jsonl`
