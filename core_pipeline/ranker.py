"""
Cosine-similarity ranking between a job description and candidate resumes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict

from .embedder import Embedder


def _cosine(a: NDArray, b: NDArray) -> float:
    return float(np.dot(a, b))


def rank_candidates(
    job_description: str,
    resumes: List[Dict[str, str]],
    top_k: int = 10,
    embedder: Embedder | None = None,
) -> List[Dict[str, float]]:
    if embedder is None:
        embedder = Embedder()

    all_texts = [job_description] + [r["text"] for r in resumes]
    embs = embedder.encode(all_texts)
    jd_vec, res_vecs = embs[0], embs[1:]

    similarities = [_cosine(jd_vec, rv) for rv in res_vecs]
    ranked = sorted(
        [{"id": r["id"], "similarity": s} for r, s in zip(resumes, similarities)],
        key=lambda d: d["similarity"],
        reverse=True,
    )
    return ranked[: top_k or len(ranked)]
