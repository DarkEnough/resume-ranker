from __future__ import annotations

import threading
from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


class Embedder:

    _lock = threading.Lock()

    def encode(self, texts: Iterable[str]) -> List[np.ndarray]:
        with self._lock:
            model = _load_model()
            return model.encode(
                list(texts),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )