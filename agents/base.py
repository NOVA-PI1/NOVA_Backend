from abc import ABC, abstractmethod
from typing import Protocol

from schemas import AgentName, AgentResult, LLMMessage, LLMRequest, SessionState


class LLMProvider(Protocol):
    async def generate(self, request: LLMRequest):
        ...


class BaseAgent(ABC):
    name: AgentName

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    @abstractmethod
    async def run(self, state: SessionState) -> AgentResult:
        raise NotImplementedError

    async def ask_llm(self, system: str, user: str, *, temperature: float = 0.2) -> tuple[str, int, str | None]:
        try:
            response = await self.llm.generate(
                LLMRequest(
                    messages=[
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    temperature=temperature,
                )
            )
            return response.text.strip(), response.tokens_used, None
        except Exception as exc:
            return "", 0, str(exc)
