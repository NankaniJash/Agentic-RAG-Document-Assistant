import os

from app.embeddings.base import Embedder

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")


def get_embedder() -> Embedder:
    if EMBED_PROVIDER == "hf":
        from app.embeddings.hf_embedder import HFInferenceEmbedder

        return HFInferenceEmbedder()
    from app.embeddings.ollama_embedder import OllamaEmbedder

    return OllamaEmbedder()
