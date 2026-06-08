"""Layer-3 Qwen API judge via DashScope (optional)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from empathy_eval.compliance import assert_api_training_only, is_test_dataset
from empathy_eval.features.extractor import QuestionFeatures
from empathy_eval.pipeline.rules import score_options


def _build_prompt(features: QuestionFeatures, rule_scores: list) -> str:
    options_block = []
    for key, opt in sorted(features.options.items()):
        rs = next((r for r in rule_scores if r.option_key == key), None)
        options_block.append(
            f"- {key}:\n"
            f"  transcript: {opt.transcript or '[ASR unavailable]'}\n"
            f"  semantic_sim_to_reference: {opt.semantic_sim_to_reference:.3f}\n"
            f"  naturalness_score: {opt.naturalness_combined:.3f}\n"
            f"  rule_combined_score: {(rs.combined if rs else 0):.3f}"
        )

    return f"""You are judging empathetic spoken response selection for a benchmark.

Task type: {features.task}
- context_variant: pick the response that best fits the conversational context.
- tone_variant: pick the response whose wording AND spoken tone best match the user's emotional state inferred from their utterance.

Context:
{features.context}

User utterance (text):
{features.utterance}

User utterance (ASR from audio):
{features.utterance_transcript}

Reference empathetic response (text anchor, not necessarily exact wording):
{features.reference_response}

Candidate spoken responses:
{chr(10).join(options_block)}

Rules:
1. Prefer responses that are contextually appropriate and emotionally attuned.
2. Flat, robotic delivery is usually wrong when a warmer human delivery exists.
3. When transcripts are nearly identical, trust naturalness / emotional fit over wording.
4. When transcripts differ, weigh semantic fit to context heavily.

Return ONLY valid JSON:
{{"choice": "opt-A|opt-B|opt-C", "reason": "one short sentence"}}
"""


def _parse_choice(text: str, valid_keys: set[str]) -> str | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        choice = obj.get("choice", "")
        if choice in valid_keys:
            return choice
    except json.JSONDecodeError:
        pass

    for key in valid_keys:
        if key in text:
            return key
    m = re.search(r"opt-[ABC]", text)
    return m.group(0) if m else None


def judge_with_qwen(
    features: QuestionFeatures,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Call Qwen via DashScope. Requires DASHSCOPE_API_KEY in environment.
    Falls back to rule winner if API unavailable.
    """
    cfg = cfg or {}
    rule_scores = score_options(features, cfg.get("rules", {}))
    fallback = rule_scores[0].option_key
    valid_keys = set(features.options.keys())

    if is_test_dataset(features.dataset):
        assert_api_training_only(
            dataset=features.dataset,
            paths=[v for opt in features.options.values() for v in [opt.audio_path]],
            operation="Qwen API judge",
        )

    api_key = os.getenv(cfg.get("api_key_env", "DASHSCOPE_API_KEY"), "")
    if not api_key:
        return {
            "choice": fallback,
            "reason": "DASHSCOPE_API_KEY not set; used rule-based fallback",
            "source": "rules_fallback",
            "rule_scores": [r.__dict__ for r in rule_scores],
        }

    try:
        import dashscope
        from dashscope import Generation
    except ImportError as exc:
        return {
            "choice": fallback,
            "reason": f"dashscope not installed: {exc}",
            "source": "rules_fallback",
            "rule_scores": [r.__dict__ for r in rule_scores],
        }

    dashscope.api_key = api_key
    prompt = _build_prompt(features, rule_scores)
    model = cfg.get("model", "qwen-plus")

    response = Generation.call(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        result_format="message",
        temperature=cfg.get("temperature", 0.1),
    )

    if getattr(response, "status_code", None) != 200:
        return {
            "choice": fallback,
            "reason": f"API error: {getattr(response, 'message', response)}",
            "source": "rules_fallback",
            "rule_scores": [r.__dict__ for r in rule_scores],
        }

    content = response.output.choices[0].message.content
    choice = _parse_choice(content, valid_keys) or fallback
    return {
        "choice": choice,
        "reason": content,
        "source": "qwen",
        "model": model,
        "rule_scores": [r.__dict__ for r in rule_scores],
    }
