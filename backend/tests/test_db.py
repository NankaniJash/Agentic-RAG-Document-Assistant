import importlib

import pytest

from app import config
from app.storage import db


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    importlib.reload(db)
    db.init_db()
    return db


def test_next_faiss_id_starts_at_zero_on_empty_table(fresh_db):
    assert fresh_db.next_faiss_id() == 0


def test_next_faiss_id_increments_past_zero(fresh_db):
    # Regression test: `row["m"] or -1` treats a legitimate MAX(faiss_id) of
    # 0 as "no rows" (0 is falsy), which used to make this return 0 again
    # instead of 1 and collide with the existing row on insert.
    fresh_db.insert_document("doc-1", "a.txt")
    fresh_db.insert_chunks([(0, "doc-1", 0, 1, "first chunk")])

    assert fresh_db.next_faiss_id() == 1


def test_next_faiss_id_after_multiple_chunks(fresh_db):
    fresh_db.insert_document("doc-1", "a.txt")
    fresh_db.insert_chunks([(0, "doc-1", 0, 1, "c0"), (1, "doc-1", 1, 1, "c1")])

    assert fresh_db.next_faiss_id() == 2
