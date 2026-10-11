import json
import re
from collections.abc import AsyncIterator

from app.agents.prompts import CRITIC_SYSTEM, FINAL_SYSTEM, PLANNER_SYSTEM, SYNTHESIS_SYSTEM
from app.llm.base import LLMClient
from app.retrieval.pipeline import Retriever

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class AgentOrchestrator:
    """Coordinates the agentic answer pipeline: plan -> retrieve -> draft ->
    critique -> stream final answer. Each step is a distinct LLM role rather
    than one monolithic prompt, so retrieval failures and unsupported claims
    get caught before the user ever sees them."""

    def __init__(self, llm: LLMClient, retriever: Retriever):
        self.llm = llm
        self.retriever = retriever

    async def answer(self, query: str, history: list[dict] | None = None) -> AsyncIterator[dict]:
        history = history or []
        history_block = self._format_history(history)

        yield {"type": "status", "data": "Planning research strategy..."}
        sub_questions = await self._plan(query, history_block)
        yield {"type": "trace", "stage": "plan", "data": {"sub_questions": sub_questions}}

        yield {"type": "status", "data": f"Retrieving evidence for {len(sub_questions)} sub-question(s)..."}
        seen_ids: set[int] = set()
        sources: list[dict] = []
        for sub_q in sub_questions:
            chunks = await self.retriever.retrieve(self.llm, sub_q)
            yield {
                "type": "trace",
                "stage": "retrieval",
                "data": {
                    "sub_question": sub_q,
                    "chunks": [
                        {"filename": c["filename"], "page": c["page"], "snippet": c["text"][:220]}
                        for c in chunks
                    ],
                },
            }
            for chunk in chunks:
                if chunk["faiss_id"] not in seen_ids:
                    seen_ids.add(chunk["faiss_id"])
                    sources.append(chunk)

        if not sources:
            yield {
                "type": "token",
                "data": "I couldn't find anything relevant in the uploaded documents to answer that.",
            }
            yield {"type": "sources", "data": []}
            yield {"type": "done"}
            return

        evidence_block = self._format_evidence(sources)

        yield {"type": "status", "data": "Drafting answer from sources..."}
        draft = await self.llm.generate(
            [
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user", "content": f"Question: {query}\n\nSource excerpts:\n{evidence_block}"},
            ],
            temperature=0.2,
        )
        yield {"type": "trace", "stage": "draft", "data": {"text": draft}}

        yield {"type": "status", "data": "Verifying claims against sources..."}
        critique = await self.llm.generate(
            [
                {"role": "system", "content": CRITIC_SYSTEM},
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nSource excerpts:\n{evidence_block}\n\nDraft answer:\n{draft}",
                },
            ],
            temperature=0.0,
        )
        yield {"type": "trace", "stage": "critique", "data": {"text": critique}}

        yield {"type": "status", "data": "Finalizing answer..."}
        final_messages = [
            {"role": "system", "content": FINAL_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"{history_block}Question: {query}\n\nSource excerpts:\n{evidence_block}\n\n"
                    f"Draft answer:\n{draft}\n\nCritic review:\n{critique}"
                ),
            },
        ]
        async for token in self.llm.stream(final_messages, temperature=0.2):
            yield {"type": "token", "data": token}

        yield {
            "type": "sources",
            "data": [
                {
                    "index": i + 1,
                    "filename": s["filename"],
                    "page": s["page"],
                    "snippet": s["text"][:280],
                }
                for i, s in enumerate(sources)
            ],
        }
        yield {"type": "done"}

    async def _plan(self, query: str, history_block: str = "") -> list[str]:
        try:
            raw = await self.llm.generate(
                [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": f"{history_block}New question: {query}"},
                ],
                temperature=0.0,
            )
            match = _JSON_ARRAY_RE.search(raw)
            sub_qs = json.loads(match.group(0)) if match else []
            sub_qs = [q for q in sub_qs if isinstance(q, str) and q.strip()]
            return sub_qs[:3] or [query]
        except Exception:
            return [query]

    @staticmethod
    def _format_evidence(sources: list[dict]) -> str:
        return "\n\n".join(
            f"[{i+1}] (from {s['filename']}, page {s['page']}): {s['text']}"
            for i, s in enumerate(sources)
        )

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        if not history:
            return ""
        turns = "\n\n".join(f"User: {h['query']}\nAssistant: {h['answer']}" for h in history)
        return f"Prior conversation:\n{turns}\n\n"
