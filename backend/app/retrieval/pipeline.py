from app import config
from app.embeddings.base import Embedder
from app.llm.base import LLMClient
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.reranker import llm_rerank
from app.retrieval.vector_store import VectorStore
from app.storage import db


class Retriever:
    """The full retrieval pipeline: vector search + BM25 keyword search run in
    parallel, fused with Reciprocal Rank Fusion, then narrowed down further by
    an LLM-based listwise reranker. Each stage trades recall for precision."""

    def __init__(self, vector_store: VectorStore, keyword_store: KeywordStore, embedder: Embedder):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.embedder = embedder

    async def retrieve(self, llm: LLMClient, query: str, top_k: int = config.FINAL_CONTEXT_CHUNKS) -> list[dict]:
        query_vec = (await self.embedder.embed([query]))[0]
        vector_hits = self.vector_store.search(query_vec, config.RETRIEVAL_TOP_K)
        keyword_hits = self.keyword_store.search(query, config.RETRIEVAL_TOP_K)

        fused = reciprocal_rank_fusion([vector_hits, keyword_hits])
        fused_ids = [doc_id for doc_id, _ in fused[: config.RERANK_CANDIDATES]]

        rows = db.get_chunks_by_faiss_ids(fused_ids)
        candidates = [
            {
                "faiss_id": fid,
                "doc_id": rows[fid]["doc_id"],
                "filename": rows[fid]["filename"],
                "page": rows[fid]["page"],
                "text": rows[fid]["text"],
            }
            for fid in fused_ids
            if fid in rows
        ]

        return await llm_rerank(llm, query, candidates, keep_top=top_k)
