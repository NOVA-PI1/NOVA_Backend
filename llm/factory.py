from config import Settings
from llm.providers import (
    AnthropicProvider,
    FakeLLMProvider,
    GeminiProvider,
    GroqProvider,
    LLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenRouterProvider,
    TogetherProvider,
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
    if settings.llm_provider == "groq":
        return GroqProvider(settings)
    if settings.llm_provider == "openrouter":
        return OpenRouterProvider(settings)
    if settings.llm_provider == "together":
        return TogetherProvider(settings)
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings)
    return FakeLLMProvider()
