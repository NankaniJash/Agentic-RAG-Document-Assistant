import os

import httpx

from app.embeddings.base import Embedder

HF_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")


class HFInferenceEmbedder(Embedder):
    """Hosted embedding fallback for deployments where Ollama isn't reachable
    (e.g. the free-tier demo host). Uses Hugging Face's free Inference API."""

    def __init__(self, model: str = HF_MODEL, api_token: str = HF_API_TOKEN):
        if not api_token:
            raise RuntimeError("HF_API_TOKEN is not set — required when EMBED_PROVIDER=hf")
        self.model = model
        self.api_token = api_token

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # HF's old api-inference.huggingface.co host was retired in favor of the
        # unified Inference Providers router; feature-extraction goes through the
        # "hf-inference" provider under /models/{id}/pipeline/feature-extraction.
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"https://router.huggingface.co/hf-inference/models/{self.model}/pipeline/feature-extraction",
                headers={"Authorization": f"Bearer {self.api_token}"},
                json={"inputs": texts, "options": {"wait_for_model": True}},
            )
            resp.raise_for_status()
            return resp.json()
