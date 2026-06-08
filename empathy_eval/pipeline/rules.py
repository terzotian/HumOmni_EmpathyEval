"""Layer-2 rule-based scoring before optional LLM judge."""

from __future__ import annotations

from dataclasses import dataclass

from empathy_eval.features.extractor import QuestionFeatures


@dataclass
class RuleScore:
    option_key: str
    semantic: float
    naturalness: float
    combined: float
    rationale: str


def _weights(task: str, option_text_similarity: float, cfg: dict) -> tuple[float, float]:
    """Return (semantic_weight, naturalness_weight)."""
    high_text_sim = option_text_similarity >= cfg.get("text_similarity_threshold", 0.85)

    if task == "tone_variant":
        sem_w = cfg.get("tone_variant_semantic_weight", 0.35)
        nat_w = cfg.get("tone_variant_naturalness_weight", 0.65)
    elif high_text_sim:
        sem_w = cfg.get("similar_text_semantic_weight", 0.25)
        nat_w = cfg.get("similar_text_naturalness_weight", 0.75)
    else:
        sem_w = cfg.get("context_variant_semantic_weight", 0.60)
        nat_w = cfg.get("context_variant_naturalness_weight", 0.40)

    total = sem_w + nat_w
    return sem_w / total, nat_w / total


def score_options(features: QuestionFeatures, cfg: dict | None = None) -> list[RuleScore]:
    cfg = cfg or {}
    sem_w, nat_w = _weights(features.task, features.option_text_similarity, cfg)
    scores: list[RuleScore] = []

    for key, opt in sorted(features.options.items()):
        semantic = opt.semantic_sim_to_reference
        if opt.naturalness_para_prob >= 0:
            naturalness = opt.naturalness_para_prob
        else:
            naturalness = opt.naturalness_combined

        combined = sem_w * semantic + nat_w * naturalness
        rationale = (
            f"semantic={semantic:.3f} (w={sem_w:.2f}), "
            f"naturalness={naturalness:.3f} (w={nat_w:.2f}), "
            f"option_text_sim={features.option_text_similarity:.3f}"
        )
        scores.append(
            RuleScore(
                option_key=key,
                semantic=semantic,
                naturalness=naturalness,
                combined=combined,
                rationale=rationale,
            )
        )

    scores.sort(key=lambda s: s.combined, reverse=True)
    return scores


def pick_by_rules(features: QuestionFeatures, cfg: dict | None = None) -> RuleScore:
    scores = score_options(features, cfg)
    return scores[0]
