"""Text semantic similarity (sentence-transformers with TF-IDF fallback)."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np


def _tokenize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


@lru_cache(maxsize=1)
def _use_sentence_transformers(model_name: str) -> bool:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401

        _load_st_encoder(model_name)
        return True
    except Exception as exc:
        print(f"[semantic] sentence-transformers unavailable ({exc}); using TF-IDF fallback")
        return False


@lru_cache(maxsize=1)
def _load_st_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _tfidf_cosine(a: str, b: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [_tokenize(a), _tokenize(b)]
    if not docs[0] or not docs[1]:
        return 0.0
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    mat = vec.fit_transform(docs)
    return float((mat[0] @ mat[1].T).toarray()[0, 0])


def _st_cosine(a: str, b: str, model_name: str) -> float:
    encoder = _load_st_encoder(model_name)
    emb = encoder.encode([a, b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def cosine_similarity(a: str, b: str, model_name: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    if _use_sentence_transformers(model_name):
        return _st_cosine(a, b, model_name)
    return _tfidf_cosine(a, b)


def pairwise_text_similarity(texts: list[str], model_name: str) -> float:
    """Mean pairwise cosine similarity across texts (high = nearly identical wording)."""
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return 1.0

    if _use_sentence_transformers(model_name):
        encoder = _load_st_encoder(model_name)
        emb = encoder.encode(texts, normalize_embeddings=True)
        sims = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims.append(float(np.dot(emb[i], emb[j])))
        return float(np.mean(sims))

    sims = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(_tfidf_cosine(texts[i], texts[j]))
    return float(np.mean(sims)) if sims else 1.0
