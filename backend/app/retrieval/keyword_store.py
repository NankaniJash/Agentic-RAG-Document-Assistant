import re

from rank_bm25 import BM25Okapi

from app.storage import db

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class KeywordStore:
    """BM25 keyword index. Rebuilt from sqlite on demand — cheap at the scale
    of a personal document assistant, and it guarantees the index can never
    drift out of sync with the source of truth (the `chunks` table)."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._faiss_ids: list[int] = []

    def refresh(self):
        rows = db.all_chunks()
        self._faiss_ids = [r["faiss_id"] for r in rows]
        corpus = [_tokenize(r["text"]) for r in rows]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._faiss_ids, scores), key=lambda x: x[1], reverse=True)
        return [(fid, score) for fid, score in ranked[:k] if score > 0]
