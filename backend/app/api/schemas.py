from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int


class DocumentOut(BaseModel):
    id: str
    filename: str
    uploaded_at: str
