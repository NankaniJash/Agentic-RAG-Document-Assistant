import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    faiss_id INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_document(doc_id: str, filename: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO documents (id, filename, uploaded_at) VALUES (?, ?, ?)",
            (doc_id, filename, datetime.now(timezone.utc).isoformat()),
        )


def insert_chunks(rows: list[tuple[int, str, int, int, str]]):
    """rows: (faiss_id, doc_id, chunk_index, page, text)"""
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO chunks (faiss_id, doc_id, chunk_index, page, text) VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def list_documents() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()


def delete_document(doc_id: str) -> list[int]:
    """Deletes the document + its chunks, returns the freed faiss_ids so the
    caller can remove them from the vector/keyword indexes too."""
    with get_conn() as conn:
        faiss_ids = [r["faiss_id"] for r in conn.execute(
            "SELECT faiss_id FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()]
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return faiss_ids


def get_chunks_by_faiss_ids(faiss_ids: list[int]) -> dict[int, sqlite3.Row]:
    if not faiss_ids:
        return {}
    with get_conn() as conn:
        placeholders = ",".join("?" for _ in faiss_ids)
        rows = conn.execute(
            f"SELECT c.*, d.filename FROM chunks c JOIN documents d ON c.doc_id = d.id "
            f"WHERE c.faiss_id IN ({placeholders})",
            faiss_ids,
        ).fetchall()
        return {r["faiss_id"]: r for r in rows}


def all_chunks() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM chunks").fetchall()


def next_faiss_id() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT MAX(faiss_id) AS m FROM chunks").fetchone()
        # `row["m"] or -1` is wrong here: when the only existing chunk has
        # faiss_id 0, `0 or -1` evaluates to -1 (0 is falsy), producing 0
        # again instead of 1 and colliding with the existing row.
        m = row["m"]
        return 0 if m is None else m + 1
