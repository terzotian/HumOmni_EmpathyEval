"""Orchestrates Layer-1 feature extraction for one question."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from empathy_eval.data import Question
from empathy_eval.features.asr import transcribe
from empathy_eval.features.naturalness import naturalness_score
from empathy_eval.features.semantic import cosine_similarity, pairwise_text_similarity


@dataclass
class OptionFeatures:
    option_key: str
    audio_path: str
    transcript: str = ""
    semantic_sim_to_reference: float = 0.0
    naturalness_heuristic: float = 0.0
    naturalness_para_prob: float = -1.0
    naturalness_combined: float = 0.0
    prosody: dict[str, float] = field(default_factory=dict)


@dataclass
class QuestionFeatures:
    question_id: str
    dataset: str
    task: str
    group_id: str
    context: str
    utterance: str
    utterance_transcript: str
    reference_response: str
    option_text_similarity: float = 0.0
    options: dict[str, OptionFeatures] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> QuestionFeatures:
        raw = json.loads(path.read_text(encoding="utf-8"))
        options = {
            k: OptionFeatures(**v) for k, v in raw.pop("options", {}).items()
        }
        return cls(options=options, **raw)


class FeatureExtractor:
    def __init__(
        self,
        whisper_model: str = "base",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        para_checkpoint: str | Path | None = None,
        skip_asr: bool = False,
    ):
        self.whisper_model = whisper_model
        self.embedding_model = embedding_model
        self.para_checkpoint = Path(para_checkpoint) if para_checkpoint else None
        self.skip_asr = skip_asr

    def extract(self, question: Question, task: str) -> QuestionFeatures:
        utterance_transcript = question.utterance
        if not self.skip_asr and question.utterance_audio.exists():
            utterance_transcript = transcribe(
                question.utterance_audio, model_name=self.whisper_model
            )

        option_features: dict[str, OptionFeatures] = {}
        transcripts: list[str] = []

        for key, audio_path in sorted(question.options.items()):
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing audio: {audio_path}")

            transcript = ""
            if not self.skip_asr:
                transcript = transcribe(audio_path, model_name=self.whisper_model)

            nat = naturalness_score(
                audio_path,
                checkpoint_path=self.para_checkpoint,
            )
            prosody = {
                k.replace("prosody_", ""): v
                for k, v in nat.items()
                if k.startswith("prosody_")
            }
            sem_sim = cosine_similarity(
                transcript or question.reference_response,
                question.reference_response,
                self.embedding_model,
            )
            transcripts.append(transcript)

            option_features[key] = OptionFeatures(
                option_key=key,
                audio_path=str(audio_path),
                transcript=transcript,
                semantic_sim_to_reference=sem_sim,
                naturalness_heuristic=nat["naturalness_heuristic"],
                naturalness_para_prob=nat["naturalness_para_prob"],
                naturalness_combined=nat["naturalness_combined"],
                prosody=prosody,
            )

        text_sim = pairwise_text_similarity(
            [t for t in transcripts if t],
            self.embedding_model,
        ) if len([t for t in transcripts if t]) >= 2 else 1.0

        return QuestionFeatures(
            question_id=question.question_id,
            dataset=question.dataset,
            task=task,
            group_id=question.group_id,
            context=question.context,
            utterance=question.utterance,
            utterance_transcript=utterance_transcript,
            reference_response=question.reference_response,
            option_text_similarity=text_sim,
            options=option_features,
        )
