import uuid
from pathlib import Path

from app.embeddings.base import Embedder
from app.ingestion.chunker import chunk_pages
from app.ingestion.loaders import load_document
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store import VectorStore
from app.storage import db


async def ingest_file(
    path: Path,
    filename: str,
    embedder: Embedder,
    vector_store: VectorStore,
    keyword_store: KeywordStore,
) -> dict:
    pages = load_document(path)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No extractable text found in document")

    doc_id = str(uuid.uuid4())
    db.insert_document(doc_id, filename)

    start_id = db.next_faiss_id()
    faiss_ids = list(range(start_id, start_id + len(chunks)))

    vectors = await embedder.embed([c.text for c in chunks])

    db.insert_chunks(
        [
            (fid, doc_id, c.chunk_index, c.page, c.text)
            for fid, c in zip(faiss_ids, chunks)
        ]
    )
    vector_store.add(faiss_ids, vectors)
    keyword_store.refresh()

    return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}


def remove_document(doc_id: str, vector_store: VectorStore, keyword_store: KeywordStore):
    faiss_ids = db.delete_document(doc_id)
    vector_store.remove(faiss_ids)
    keyword_store.refresh()
