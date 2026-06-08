"""End-to-end prediction: features -> rules -> optional Qwen."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from empathy_eval.data import Question, TEST_DATASETS, load_questions
from empathy_eval.features.extractor import FeatureExtractor, QuestionFeatures
from empathy_eval.pipeline.qwen_judge import judge_with_qwen
from empathy_eval.pipeline.rules import pick_by_rules, score_options


@dataclass
class Prediction:
    question_id: str
    dataset: str
    group_id: str
    choice: str
    source: str
    reason: str
    rule_winner: str
    rule_scores: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    def __init__(self, project_root: Path, config_path: Path):
        self.project_root = project_root
        with config_path.open(encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        paths = self.cfg.get("paths", {})
        self.features_dir = project_root / paths.get("features_dir", "outputs/features")
        self.predictions_dir = project_root / paths.get("predictions_dir", "outputs/predictions")

        fe_cfg = self.cfg.get("feature_extractor", {})
        para_ckpt = fe_cfg.get("para_checkpoint")
        if para_ckpt:
            para_ckpt = project_root / para_ckpt

        self.extractor = FeatureExtractor(
            whisper_model=fe_cfg.get("whisper_model", "base"),
            embedding_model=fe_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            para_checkpoint=para_ckpt,
            skip_asr=fe_cfg.get("skip_asr", False),
        )

    def _feature_path(self, dataset: str, question_id: str) -> Path:
        return self.features_dir / dataset / f"{question_id}.json"

    def _task_for_question(self, question: Question) -> str:
        if question.task:
            return question.task
        if question.dataset in TEST_DATASETS:
            return str(TEST_DATASETS[question.dataset]["task"])
        return "context_variant"

    def extract_and_cache(self, question: Question, force: bool = False) -> QuestionFeatures:
        out = self._feature_path(question.dataset, question.question_id)
        if out.exists() and not force:
            return QuestionFeatures.load(out)

        task = self._task_for_question(question)
        features = self.extractor.extract(question, task=task)
        features.save(out)
        return features

    def predict_one(
        self,
        question: Question,
        use_qwen: bool | None = None,
        force_features: bool = False,
    ) -> Prediction:
        features = self.extract_and_cache(question, force=force_features)
        rules_cfg = self.cfg.get("rules", {})
        rule_scores = score_options(features, rules_cfg)
        rule_winner = pick_by_rules(features, rules_cfg).option_key

        qwen_cfg = self.cfg.get("qwen", {})
        if use_qwen is None:
            use_qwen = qwen_cfg.get("enabled", False)

        if use_qwen:
            result = judge_with_qwen(features, self.cfg)
            choice = result["choice"]
            source = result["source"]
            reason = result.get("reason", "")
        else:
            choice = rule_winner
            source = "rules"
            reason = rule_scores[0].rationale

        return Prediction(
            question_id=question.question_id,
            dataset=question.dataset,
            group_id=question.group_id,
            choice=choice,
            source=source,
            reason=reason,
            rule_winner=rule_winner,
            rule_scores=[r.__dict__ for r in rule_scores],
        )

    def predict_dataset(
        self,
        dataset: str,
        limit: int | None = None,
        use_qwen: bool | None = None,
        force_features: bool = False,
    ) -> list[Prediction]:
        preds: list[Prediction] = []
        for i, q in enumerate(load_questions(self.project_root, [dataset])):
            if limit is not None and i >= limit:
                break
            preds.append(self.predict_one(q, use_qwen=use_qwen, force_features=force_features))
        return preds

    def save_predictions(self, dataset: str, predictions: list[Prediction]) -> Path:
        out_dir = self.predictions_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        return out_path
