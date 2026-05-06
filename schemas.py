from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


AgentName = Literal["etico", "dialectico", "editorial", "multimodal", "orchestrator"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    model: str | None = None
    max_tokens: int | None = None
    temperature: float = 0.2


class LLMResponse(BaseModel):
    text: str
    model: str
    provider: str
    tokens_used: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


class SessionRequest(BaseModel):
    texto: str = Field(min_length=1)
    session_id: str | None = None
    perfil: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    images: list[str] = Field(default_factory=list)


class KnowledgeHit(BaseModel):
    text: str
    score: float | None = None
    source: str = "bcl"


class AgentResult(BaseModel):
    agent: AgentName
    output: str
    warnings: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    tokens_used: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class BusEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    input_text: str
    perfil: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    knowledge_hits: list[KnowledgeHit] = Field(default_factory=list)
    agent_results: list[AgentResult] = Field(default_factory=list)
    status: Literal["created", "running", "completed", "failed"] = "created"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SessionResponse(BaseModel):
    session_id: str
    status: str
    editorial: AgentResult | None = None
    etico: AgentResult | None = None
    dialectico: AgentResult | None = None
    multimodal: AgentResult | None = None
    knowledge_hits: list[KnowledgeHit] = Field(default_factory=list)
    trace: list[AgentResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanvasEditRequest(BaseModel):
    session_id: str | None = None
    texto: str | None = None
    canvas: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
