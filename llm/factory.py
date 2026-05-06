from config import Settings
from llm.providers import (
    AnthropicProvider,
    FakeLLMProvider,
    GeminiProvider,
    LLMProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai-compatible":
        return OpenAICompatibleProvider(settings)
    if settings.llm_provider == "openai":
        return OpenAIProvider(settings)
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(settings)
    if settings.llm_provider == "gemini":
        return GeminiProvider(settings)
    return FakeLLMProvider()
