"""Evaluation harness for the retrieval + generation pipeline.

Usage:
    1. Upload the documents your eval cases reference (via the running app, or
       call app.ingestion.service.ingest_file directly).
    2. Fill in app/eval/dataset.json with cases:
       {"question": "...", "expected_source": "some_file.pdf"}
       `expected_source` is optional — omit it to only score faithfulness.
    3. Run: python -m app.eval.harness

Reports two independent signals, because a RAG system can fail at either stage:
    - retrieval hit-rate: did the expected document even make it into context?
    - faithfulness: is the generated answer actually supported by what was
      retrieved (LLM-as-judge), independent of whether retrieval was "correct"?
"""

import asyncio
import json
from pathlib import Path

from app.agents.orchestrator import AgentOrchestrator
from app.embeddings import get_embedder
from app.llm import get_llm_client
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.pipeline import Retriever
from app.retrieval.vector_store import VectorStore
from app.storage import db

DATASET_PATH = Path(__file__).parent / "dataset.json"
REPORT_PATH = Path(__file__).parent / "last_run_report.json"

JUDGE_SYSTEM = (
    "You are an evaluation judge. Given a question, the retrieved source excerpts, "
    "and a generated answer, score the answer 1-5 on FAITHFULNESS: is every claim "
    "supported by the excerpts, with no fabrication? Respond with ONLY a JSON "
    'object: {"score": <int 1-5>, "reason": "<one sentence>"}'
)


async def run_eval():
    db.init_db()
    dataset = json.loads(DATASET_PATH.read_text())
    if not dataset:
        print("No eval cases in app/eval/dataset.json — add some first. See module docstring.")
        return

    embedder = get_embedder()
    vector_store = VectorStore()
    keyword_store = KeywordStore()
    keyword_store.refresh()
    llm = get_llm_client()
    retriever = Retriever(vector_store, keyword_store, embedder)

    results = []
    for case in dataset:
        question = case["question"]
        expected_source = case.get("expected_source")

        retrieved = await retriever.retrieve(llm, question)
        hit = (expected_source in {r["filename"] for r in retrieved}) if expected_source else None

        orchestrator = AgentOrchestrator(llm, retriever)
        answer, evidence_used = "", []
        async for event in orchestrator.answer(question):
            if event["type"] == "token":
                answer += event["data"]
            elif event["type"] == "sources":
                evidence_used = event["data"]

        judge_input = (
            f"Question: {question}\n\nRetrieved excerpts:\n"
            + "\n".join(f"[{s['index']}] {s['snippet']}" for s in evidence_used)
            + f"\n\nGenerated answer:\n{answer}"
        )
        raw = await llm.generate(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": judge_input}],
            temperature=0.0,
        )
        try:
            judgment = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except Exception:
            judgment = {"score": None, "reason": "unparseable judge output"}

        results.append(
            {
                "question": question,
                "retrieval_hit": hit,
                "faithfulness_score": judgment.get("score"),
                "judge_reason": judgment.get("reason"),
                "answer": answer,
            }
        )
        print(f"Q: {question}\n  retrieval_hit={hit}  faithfulness={judgment.get('score')}/5  ({judgment.get('reason')})\n")

    hits = [r["retrieval_hit"] for r in results if r["retrieval_hit"] is not None]
    scores = [r["faithfulness_score"] for r in results if isinstance(r["faithfulness_score"], (int, float))]
    print("=== Summary ===")
    if hits:
        print(f"Retrieval hit-rate: {sum(hits)}/{len(hits)} ({100 * sum(hits) / len(hits):.0f}%)")
    if scores:
        print(f"Avg faithfulness: {sum(scores) / len(scores):.2f}/5")

    REPORT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(run_eval())
