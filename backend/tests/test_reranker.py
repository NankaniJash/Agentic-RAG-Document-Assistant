from conftest import FakeLLM

from app.retrieval.reranker import llm_rerank


async def test_reorders_candidates_by_model_output():
    candidates = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}, {"id": "c", "text": "C"}]
    llm = FakeLLM("[3,1,2]")

    result = await llm_rerank(llm, "query", candidates, keep_top=3)

    assert [c["id"] for c in result] == ["c", "a", "b"]


async def test_respects_keep_top():
    candidates = [{"id": i, "text": f"chunk {i}"} for i in range(5)]
    llm = FakeLLM("[1,2,3,4,5]")

    result = await llm_rerank(llm, "query", candidates, keep_top=2)

    assert len(result) == 2


async def test_falls_back_to_fused_order_on_unparseable_output():
    candidates = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
    llm = FakeLLM("I refuse to rank these, sorry.")

    result = await llm_rerank(llm, "query", candidates, keep_top=2)

    assert [c["id"] for c in result] == ["a", "b"]


async def test_appends_any_candidate_the_model_omitted():
    candidates = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}, {"id": "c", "text": "C"}]
    llm = FakeLLM("[2]")  # model only ranked one of three

    result = await llm_rerank(llm, "query", candidates, keep_top=3)

    assert [c["id"] for c in result] == ["b", "a", "c"]


async def test_empty_candidates_returns_empty_without_calling_model():
    llm = FakeLLM("[]")

    result = await llm_rerank(llm, "query", [], keep_top=5)

    assert result == []
    assert llm.calls == []
