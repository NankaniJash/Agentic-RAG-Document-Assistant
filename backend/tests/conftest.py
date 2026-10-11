import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeLLM:
    """Scripted LLMClient stand-in so retrieval/agent tests never hit a real
    model — deterministic, no Ollama/Groq required to run the suite."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list[list[dict]] = []

    async def generate(self, messages: list[dict], temperature: float = 0.2) -> str:
        self.calls.append(messages)
        return self.response

    async def stream(self, messages: list[dict], temperature: float = 0.2):
        self.calls.append(messages)
        yield self.response
