from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMClient(ABC):
    """Common interface so the rest of the app doesn't care whether it's
    talking to a local Ollama model or a hosted Groq model."""

    @abstractmethod
    async def generate(self, messages: list[dict], temperature: float = 0.2) -> str:
        """Return the full completion in one shot (used for planning/critique/rerank steps)."""

    @abstractmethod
    async def stream(self, messages: list[dict], temperature: float = 0.2) -> AsyncIterator[str]:
        """Yield the completion token-by-token (used for the final answer to the user)."""
