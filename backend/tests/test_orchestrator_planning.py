from conftest import FakeLLM

from app.agents.orchestrator import AgentOrchestrator


async def test_plan_parses_json_array_and_caps_at_three():
    orchestrator = AgentOrchestrator(FakeLLM('["q1", "q2", "q3", "q4"]'), retriever=None)

    result = await orchestrator._plan("original query")

    assert result == ["q1", "q2", "q3"]


async def test_plan_falls_back_to_original_query_on_unparseable_output():
    orchestrator = AgentOrchestrator(FakeLLM("I don't understand the request."), retriever=None)

    result = await orchestrator._plan("original query")

    assert result == ["original query"]


async def test_plan_falls_back_when_model_returns_empty_array():
    orchestrator = AgentOrchestrator(FakeLLM("[]"), retriever=None)

    result = await orchestrator._plan("original query")

    assert result == ["original query"]


async def test_plan_includes_history_block_in_prompt_when_provided():
    llm = FakeLLM('["resolved question"]')
    orchestrator = AgentOrchestrator(llm, retriever=None)
    history_block = "Prior conversation:\nUser: What is RRF?\nAssistant: It's a fusion technique.\n\n"

    await orchestrator._plan("What about BM25 instead?", history_block)

    sent_content = llm.calls[0][1]["content"]
    assert "Prior conversation" in sent_content
    assert "What about BM25 instead?" in sent_content


async def test_plan_omits_history_block_when_absent():
    llm = FakeLLM('["q1"]')
    orchestrator = AgentOrchestrator(llm, retriever=None)

    await orchestrator._plan("a standalone question")

    sent_content = llm.calls[0][1]["content"]
    assert "Prior conversation" not in sent_content


def test_format_history_renders_prior_turns():
    history = [{"query": "What is RRF?", "answer": "It's a fusion technique."}]

    block = AgentOrchestrator._format_history(history)

    assert "What is RRF?" in block
    assert "It's a fusion technique." in block


def test_format_history_empty_list_returns_empty_string():
    assert AgentOrchestrator._format_history([]) == ""
