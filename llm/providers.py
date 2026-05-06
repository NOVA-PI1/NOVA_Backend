from abc import ABC, abstractmethod
from typing import Any

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from config import Settings
from schemas import LLMRequest, LLMResponse


class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


class FakeLLMProvider(LLMProvider):
    name = "fake"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        user_text = next((message.content for message in reversed(request.messages) if message.role == "user"), "")
        return LLMResponse(
            text=f"[Respuesta simulada Nova] {user_text[:700]}",
            model=request.model or "nova-fake",
            provider=self.name,
            tokens_used=len(user_text.split()),
        )


class OpenAICompatibleProvider(LLMProvider):
    name = "openai-compatible"

    def __init__(self, settings: Settings, *, provider_name: str | None = None) -> None:
        api_key = settings.openai_api_key or "not-required"
        self.settings = settings
        self.name = provider_name or self.name
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.settings.llm_model
        response = await self.client.chat.completions.create(
            model=model,
            messages=[message.model_dump() for message in request.messages],
            max_tokens=request.max_tokens or self.settings.llm_max_tokens,
            temperature=request.temperature,
        )
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) if usage else 0
        return LLMResponse(text=text, model=model, provider=self.name, tokens_used=tokens)


class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, provider_name="openai")


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.settings.llm_model
        system = "\n\n".join(message.content for message in request.messages if message.role == "system")
        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
            if message.role in {"user", "assistant"}
        ]
        response = await self.client.messages.create(
            model=model,
            max_tokens=request.max_tokens or self.settings.llm_max_tokens,
            temperature=request.temperature,
            system=system or None,
            messages=messages,
        )
        text = "".join(getattr(block, "text", "") for block in response.content)
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return LLMResponse(text=text, model=model, provider=self.name, tokens_used=tokens)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for the gemini provider")

        model = request.model or self.settings.llm_model
        prompt = "\n\n".join(f"{message.role}: {message.content}" for message in request.messages)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            f"?key={self.settings.gemini_api_key}"
        )
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": request.max_tokens or self.settings.llm_max_tokens,
                "temperature": request.temperature,
            },
        }
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "".join(part.get("text", "") for part in parts)
        token_data = data.get("usageMetadata", {})
        tokens = token_data.get("totalTokenCount", 0)
        return LLMResponse(text=text, model=model, provider=self.name, tokens_used=tokens, raw={"usage": token_data})
