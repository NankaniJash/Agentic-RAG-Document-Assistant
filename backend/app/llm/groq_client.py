import json
from collections.abc import AsyncIterator

import httpx

from app import config
from app.llm.base import LLMClient


class GroqClient(LLMClient):
    """OpenAI-compatible client for Groq's hosted inference API.
    Used for the live deployed demo, where a local Ollama daemon isn't available."""

    def __init__(self, api_key: str = config.GROQ_API_KEY, model: str = config.GROQ_MODEL):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set — required when LLM_PROVIDER=groq")
        self.api_key = api_key
        self.model = model

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def generate(self, messages: list[dict], temperature: float = 0.2) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{config.GROQ_BASE_URL}/chat/completions",
                headers=self._headers(),
                json={"model": self.model, "messages": messages, "temperature": temperature},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def stream(self, messages: list[dict], temperature: float = 0.2) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{config.GROQ_BASE_URL}/chat/completions",
                headers=self._headers(),
                json={"model": self.model, "messages": messages, "temperature": temperature, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload.strip() == "[DONE]":
                        break
                    delta = json.loads(payload)["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        yield content
