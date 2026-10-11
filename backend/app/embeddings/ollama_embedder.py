import httpx

from app import config
from app.embeddings.base import Embedder


class OllamaEmbedder(Embedder):
    def __init__(self, host: str = config.OLLAMA_HOST, model: str = config.OLLAMA_EMBED_MODEL):
        self.host = host
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
