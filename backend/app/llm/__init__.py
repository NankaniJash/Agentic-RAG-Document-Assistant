from app import config
from app.llm.base import LLMClient


def get_llm_client() -> LLMClient:
    if config.LLM_PROVIDER == "groq":
        from app.llm.groq_client import GroqClient

        return GroqClient()
    from app.llm.ollama_client import OllamaClient

    return OllamaClient()
