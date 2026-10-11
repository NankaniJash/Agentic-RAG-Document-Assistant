import json
import re

from app.llm.base import LLMClient

_RERANK_SYSTEM = (
    "You are a relevance-ranking engine. You will be given a user query and a "
    "numbered list of text passages. Rank the passage numbers from MOST to LEAST "
    "relevant to answering the query. Respond with ONLY a JSON array of integers, "
    "e.g. [3,1,4,2]. Include every passage number exactly once. No other text."
)

_JSON_ARRAY_RE = re.compile(r"\[[\d,\s]+\]")


async def llm_rerank(llm: LLMClient, query: str, candidates: list[dict], keep_top: int) -> list[dict]:
    """Listwise LLM reranking (RankGPT-style): the LLM sees all candidates at
    once and reasons about relative relevance, which catches cases plain
    similarity scores miss (e.g. a lexically-similar but off-topic passage).
    Falls back to the pre-rerank fused order if the model output can't be parsed."""
    if not candidates:
        return []

    listing = "\n\n".join(
        f"[{i+1}] {c['text'][:600]}" for i, c in enumerate(candidates)
    )
    messages = [
        {"role": "system", "content": _RERANK_SYSTEM},
        {"role": "user", "content": f"Query: {query}\n\nPassages:\n{listing}"},
    ]

    try:
        raw = await llm.generate(messages, temperature=0.0)
        match = _JSON_ARRAY_RE.search(raw)
        order = json.loads(match.group(0)) if match else []
        seen = set()
        reranked = []
        for num in order:
            i = num - 1
            if 0 <= i < len(candidates) and i not in seen:
                reranked.append(candidates[i])
                seen.add(i)
        for i, c in enumerate(candidates):
            if i not in seen:
                reranked.append(c)
        return reranked[:keep_top]
    except Exception:
        return candidates[:keep_top]
