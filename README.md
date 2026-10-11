Agentic RAG Document Assistant



Upload documents, then ask questions and get answers grounded in your own
sources with citations — powered by a multi-stage retrieval pipeline and a
multi-agent reasoning loop rather than a single prompt to an LLM.

## Why this isn't "just a RAG chatbot"

Most portfolio RAG projects wire an embedding model to a vector DB and call
it done. Synapse goes further at each stage:

- **Hybrid retrieval** — every query runs through both BM25 keyword search and
  dense vector search in parallel, merged with **Reciprocal Rank Fusion**
  (`app/retrieval/hybrid.py`). Vector search alone misses exact-term matches
  (IDs, names, jargon); keyword search alone misses paraphrases. RRF gets both
  without needing to normalize incomparable similarity scores.
- **LLM-based listwise reranking** — the fused candidates are re-ordered by
  showing them all to the LLM at once and asking it to rank them
  (RankGPT-style, `app/retrieval/reranker.py`), which catches relevance
  nuances plain similarity scores miss.
- **Multi-agent reasoning pipeline** (`app/agents/orchestrator.py`), not one
  prompt: a **planner** decomposes the question into sub-questions, a
  **synthesizer** drafts an answer strictly from retrieved evidence, a
  **critic** checks every claim against the sources, and a final pass
  produces the polished, corrected, cited answer — streamed to the UI token
  by token.
- **Automated evaluation harness** (`app/eval/harness.py`) — measures
  retrieval hit-rate and LLM-judged answer faithfulness, so quality claims
  are backed by numbers instead of vibes.
- **Fully pluggable model backend** — local (Ollama, free, no API key) for
  development, hosted (Groq + Hugging Face) for the live deployed demo, via a
  single `LLMClient` / `Embedder` interface (`app/llm/`, `app/embeddings/`).
- **Full reasoning trace in the UI** — every answer has a collapsible panel
  showing the planner's sub-questions, the evidence retrieved for each one,
  the pre-critique draft, and the critic's actual findings, not just a final
  source list.
- **Multi-turn conversation memory** — follow-up questions ("why is that
  better?") resolve against prior turns via a per-session history
  (`app/api/routes.py`), while the answer itself still has to come from
  retrieved evidence, not the model's memory of the chat.
- **Rate-limited API** (`app/rate_limit.py`) — protects the hosted demo's free
  Groq/HF quota from abuse.
- **Test suite + CI** (`backend/tests/`, `.github/workflows/ci.yml`) — unit
  tests for chunking, RRF fusion, reranking, planning fallbacks, and the
  vector/keyword stores, run on every push.

## Architecture

```
Upload → loaders.py → chunker.py → embeddings → FAISS + BM25 index (sqlite metadata)
                                                        │
User query → planner (LLM) → sub-questions ────────────┤
                                                        ▼
                              hybrid retrieval (RRF) → LLM rerank → evidence
                                                        │
                              synthesizer (LLM) → draft answer
                                                        │
                              critic (LLM) → faithfulness check
                                                        │
                              final pass (LLM, streamed) → cited answer + sources
```

## Running locally

Requires [Ollama](https://ollama.com) installed and running.

```bash
ollama pull llama3.2
ollama pull nomic-embed-text

cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp ../.env.example .env       # defaults already point at Ollama

uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Running the tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest -v
```

## Running the evaluation harness

```bash
# 1. Upload the doc(s) your eval cases reference via the running app's UI
# 2. Fill in app/eval/dataset.json:
#    [{"question": "...", "expected_source": "some_file.pdf"}]
cd backend
python -m app.eval.harness
```

Reports retrieval hit-rate and LLM-judged faithfulness, and writes a full
per-case report to `app/eval/last_run_report.json`.

## Deploying the live demo

Free hosting platforms (Render, Railway free tiers, etc.) can't run a local
Ollama daemon, so the deployed version swaps in hosted APIs — no code
changes, just environment variables:

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile

EMBED_PROVIDER=hf
HF_API_TOKEN=...
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_DIM=384
```

This repo includes a `render.yaml` Blueprint: on [Render](https://render.com),
choose **New + → Blueprint**, point it at this repo, and Render will pick up
the service config automatically. You'll be prompted to paste in `GROQ_API_KEY`
and `HF_API_TOKEN` as secrets in Render's dashboard (get free keys at
[console.groq.com](https://console.groq.com) and
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) —
never commit these to the repo.

**Note:** Render's free tier has ephemeral disk, so uploaded documents and the
FAISS/sqlite index reset on every redeploy or restart. Fine for a live demo
(re-upload a sample doc after each restart); for persistence, upgrade to a
paid disk or point `DATA_DIR` at external storage.

## Stack

FastAPI · FAISS · rank-bm25 · Ollama (llama3.2, nomic-embed-text) · Groq
(hosted deployment) · vanilla JS/CSS frontend with streaming SSE
