from dataclasses import dataclass

from app import config
from app.ingestion.loaders import PageText


@dataclass
class Chunk:
    text: str
    page: int
    chunk_index: int


def chunk_pages(
    pages: list[PageText],
    size: int = config.CHUNK_SIZE_CHARS,
    overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """Paragraph-aware sliding-window chunking: greedily pack paragraphs into
    ~`size`-char windows, carrying the tail of each window into the next as
    overlap so answers near a chunk boundary don't lose context."""
    chunks: list[Chunk] = []
    idx = 0
    for page in pages:
        paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
        buf = ""
        for para in paragraphs:
            candidate = f"{buf}\n\n{para}" if buf else para
            if len(candidate) > size and buf:
                chunks.append(Chunk(text=buf, page=page.page, chunk_index=idx))
                idx += 1
                buf = buf[-overlap:] + "\n\n" + para if overlap else para
            else:
                buf = candidate
        if buf.strip():
            chunks.append(Chunk(text=buf, page=page.page, chunk_index=idx))
            idx += 1
    return chunks
