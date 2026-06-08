"""Prompt templates for local Qwen2.5-Omni empathy selection."""

from __future__ import annotations

from empathy_eval.data import Question


def build_selection_instruction(question: Question) -> str:
    option_lines = []
    for key in sorted(question.options):
        letter = key.replace("opt-", "")
        option_lines.append(f"Candidate {letter}: (audio attached as {key})")

    return (
        "You are selecting the most empathetic spoken response for a benchmark.\n"
        f"Task: listen to the user's utterance audio and pick the best response audio.\n\n"
        f"Context:\n{question.context}\n\n"
        f"User utterance (text hint):\n{question.utterance}\n\n"
        f"Reference response text (semantic anchor, not exact wording required):\n"
        f"{question.reference_response}\n\n"
        "Candidate response audios:\n"
        + "\n".join(option_lines)
        + "\n\n"
        "Rules:\n"
        "1. Prefer responses that fit the context and sound emotionally attuned.\n"
        "2. Avoid flat robotic delivery when a warmer human delivery exists.\n"
        "3. When wording is similar, prioritize vocal empathy and natural prosody.\n\n"
        "Reply with ONLY one letter: A, B, or C."
    )


def build_training_instruction(example: dict) -> str:
    return (
        "You are selecting the most empathetic spoken response for a benchmark.\n"
        f"Task type: {example['task']}\n\n"
        f"Context:\n{example['context']}\n\n"
        f"User utterance (text hint):\n{example['utterance_text']}\n\n"
        f"Reference response text:\n{example['reference_response']}\n\n"
        "Two candidate response audios are attached (opt-A and opt-B).\n"
        "Reply with ONLY one letter: A or B."
    )
