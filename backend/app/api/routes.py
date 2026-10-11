import json
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app import config
from app.agents.orchestrator import AgentOrchestrator
from app.api.schemas import ChatRequest, DocumentOut, UploadResponse
from app.ingestion.service import ingest_file, remove_document
from app.llm import get_llm_client
from app.rate_limit import limiter
from app.retrieval.pipeline import Retriever
from app.storage import db

router = APIRouter()

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}


@router.post("/documents/upload", response_model=UploadResponse)
@limiter.limit(config.RATE_LIMIT_UPLOAD)
async def upload_document(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "Only .pdf, .txt, and .md files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result = await ingest_file(
            tmp_path,
            file.filename,
            request.app.state.embedder,
            request.app.state.vector_store,
            request.app.state.keyword_store,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    return result


@router.get("/documents", response_model=list[DocumentOut])
async def get_documents():
    rows = db.list_documents()
    return [{"id": r["id"], "filename": r["filename"], "uploaded_at": r["uploaded_at"]} for r in rows]


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    remove_document(doc_id, request.app.state.vector_store, request.app.state.keyword_store)
    return {"status": "deleted"}


@router.post("/chat")
@limiter.limit(config.RATE_LIMIT_CHAT)
async def chat(req: ChatRequest, request: Request):
    llm = get_llm_client()
    retriever = Retriever(
        request.app.state.vector_store, request.app.state.keyword_store, request.app.state.embedder
    )
    orchestrator = AgentOrchestrator(llm, retriever)

    sessions = request.app.state.sessions
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.get(session_id, [])

    async def event_stream():
        answer_text = ""
        yield f"data: {json.dumps({'type': 'session', 'data': session_id})}\n\n"
        async for event in orchestrator.answer(req.query, history):
            if event["type"] == "token":
                answer_text += event["data"]
            yield f"data: {json.dumps(event)}\n\n"

        updated = history + [{"query": req.query, "answer": answer_text}]
        sessions[session_id] = updated[-config.SESSION_MAX_TURNS:]

    return StreamingResponse(event_stream(), media_type="text/event-stream")
