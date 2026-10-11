import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "synapse.db"
FAISS_INDEX_PATH = DATA_DIR / "vectors.faiss"
BM25_STORE_PATH = DATA_DIR / "bm25_store.pkl"

# "ollama" for local dev, "groq" for the hosted demo deployment.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # nomic-embed-text output size

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1600"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))  # per retriever, pre-fusion
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "12"))  # post-fusion, pre-rerank
FINAL_CONTEXT_CHUNKS = int(os.getenv("FINAL_CONTEXT_CHUNKS", "5"))  # post-rerank, used as evidence

RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "20/hour")
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "10/hour")

SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "6"))  # conversation turns kept per session
