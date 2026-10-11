from app.retrieval.vector_store import VectorStore


def test_search_returns_closest_vector_first(tmp_path):
    store = VectorStore(dim=4, path=tmp_path / "test.faiss")
    store.add([1, 2, 3], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    results = store.search([0.9, 0.1, 0, 0], k=1)

    assert results[0][0] == 1


def test_remove_excludes_id_from_future_searches(tmp_path):
    store = VectorStore(dim=4, path=tmp_path / "test.faiss")
    store.add([1, 2], [[1, 0, 0, 0], [0, 1, 0, 0]])

    store.remove([1])
    ids = [doc_id for doc_id, _ in store.search([1, 0, 0, 0], k=2)]

    assert 1 not in ids
    assert 2 in ids


def test_search_on_empty_index_returns_empty_list(tmp_path):
    store = VectorStore(dim=4, path=tmp_path / "test.faiss")

    assert store.search([1, 0, 0, 0], k=5) == []


def test_index_persists_and_reloads_from_disk(tmp_path):
    path = tmp_path / "test.faiss"
    store = VectorStore(dim=4, path=path)
    store.add([7], [[0, 0, 0, 1]])

    reloaded = VectorStore(dim=4, path=path)
    results = reloaded.search([0, 0, 0, 1], k=1)

    assert results[0][0] == 7
