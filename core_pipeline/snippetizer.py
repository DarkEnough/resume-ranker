

from __future__ import annotations
import re
from typing import List

import numpy as np
from sentence_transformers.util import cos_sim

from .embedder import Embedder


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def top_k_snippets(
    job_desc: str,
    resume_text: str,
    k: int = 3,
    embedder: Embedder | None = None,
) -> List[str]:
    """
    • Split résumé into sentences
    • Encode JD once, each sentence once
    • Return the K sentences with highest cosine similarity
    """
    embedder = embedder or Embedder()

    sentences: List[str] = [s.strip() for s in _SENT_SPLIT.split(resume_text) if s.strip()]
    if not sentences:
        return []

    emb = embedder.encode([job_desc] + sentences)
    jd_vec, sent_vecs = emb[0], emb[1:]
    sims = cos_sim(jd_vec, sent_vecs).flatten()

    top_idx = np.argsort(-sims)[:k]                    # highest first
    return [sentences[i] for i in top_idx]
