import threading

import faiss
import numpy as np

from app import config


class VectorStore:
    """FAISS-backed cosine-similarity index (inner product over L2-normalized
    vectors), keyed by the same integer ids as the sqlite `chunks` table so
    results can be joined straight back to their text/citation metadata."""

    def __init__(self, dim: int = config.EMBED_DIM, path=config.FAISS_INDEX_PATH):
        self.dim = dim
        self.path = path
        self._lock = threading.Lock()
        if path.exists():
            self.index = faiss.read_index(str(path))
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms

    def add(self, ids: list[int], vectors: list[list[float]]):
        vecs = self._normalize(np.array(vectors, dtype="float32"))
        with self._lock:
            self.index.add_with_ids(vecs, np.array(ids, dtype="int64"))
            self._save()

    def search(self, vector: list[float], k: int) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        vec = self._normalize(np.array([vector], dtype="float32"))
        with self._lock:
            scores, ids = self.index.search(vec, min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def remove(self, ids: list[int]):
        if not ids:
            return
        with self._lock:
            self.index.remove_ids(np.array(ids, dtype="int64"))
            self._save()

    def _save(self):
        faiss.write_index(self.index, str(self.path))
