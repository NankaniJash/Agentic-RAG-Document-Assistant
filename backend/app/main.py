from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import router
from app.embeddings import get_embedder
from app.rate_limit import limiter
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store import VectorStore
from app.storage import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    app.state.embedder = get_embedder()
    app.state.vector_store = VectorStore()
    app.state.keyword_store = KeywordStore()
    app.state.keyword_store.refresh()
    app.state.sessions = {}
    yield


app = FastAPI(title="Synapse — Agentic Document Assistant", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
