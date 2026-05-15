from typing import Iterable

from agents.dialectico import DialecticalAgent
from agents.editorial import EditorialAgent
from agents.etico import EthicalAgent
from agents.multimodal import MultimodalAgent
from bcl.loader import KnowledgeBaseService
from schemas import (
    AgentResult,
    BusEvent,
    CanvasEditRequest,
    SessionRequest,
    SessionResponse,
    SessionState,
    utc_now,
)
from services import InMemoryMessageBus, SQLiteSessionStore


class NovaOrchestrator:
    def __init__(
        self,
        *,
        store: SQLiteSessionStore,
        bus: InMemoryMessageBus,
        knowledge_base: KnowledgeBaseService,
        agents: Iterable[object],
    ) -> None:
        self.store = store
        self.bus = bus
        self.knowledge_base = knowledge_base
        self.agents = list(agents)

    async def run_session(self, request: SessionRequest) -> SessionResponse:
        state_data = {
            "input_text": request.texto,
            "perfil": request.perfil,
            "metadata": {**request.metadata, "images": request.images},
            "status": "running",
        }
        if request.session_id:
            state_data["session_id"] = request.session_id
        state = SessionState(**state_data)

        state.knowledge_hits = self.knowledge_base.search(request.texto)
        self.store.save_session(state)
        self.store.save_message(state.session_id, "user", request.texto)
        await self._publish(state.session_id, "session.started", {"status": state.status})

        for agent in self.agents:
            result = await agent.run(state)
            state.agent_results.append(result)
            state.updated_at = utc_now()
            self.store.save_agent_result(state.session_id, result)
            self.store.save_session(state)
            await self._publish(
                state.session_id,
                f"agent.{result.agent}.completed",
                result.model_dump(mode="json"),
            )

        state.status = "completed"
        state.updated_at = utc_now()
        self.store.save_session(state)
        await self._publish(state.session_id, "session.completed", {"status": state.status})
        return self._response_from_state(state)

    def get_session(self, session_id: str) -> SessionResponse | None:
        state = self.store.get_session(session_id)
        if state is None:
            return None
        return self._response_from_state(state)

    async def handle_canvas_edit(self, request: CanvasEditRequest) -> BusEvent:
        session_id = request.session_id or "canvas"
        payload = {
            "texto": request.texto,
            "canvas": request.canvas,
            "metadata": request.metadata,
        }
        event = BusEvent(session_id=session_id, type="canvas.edited", payload=payload)
        await self.bus.publish(event)
        self.store.save_event(event)
        if request.texto:
            self.store.save_message(session_id, "canvas", request.texto)
        return event

    async def _publish(self, session_id: str, event_type: str, payload: dict) -> None:
        event = BusEvent(session_id=session_id, type=event_type, payload=payload)
        await self.bus.publish(event)
        self.store.save_event(event)

    def _response_from_state(self, state: SessionState) -> SessionResponse:
        by_agent: dict[str, AgentResult] = {result.agent: result for result in state.agent_results}
        return SessionResponse(
            session_id=state.session_id,
            status=state.status,
            editorial=by_agent.get("editorial"),
            etico=by_agent.get("etico"),
            dialectico=by_agent.get("dialectico"),
            multimodal=by_agent.get("multimodal"),
            knowledge_hits=state.knowledge_hits,
            trace=state.agent_results,
            metadata=state.metadata,
        )


def create_orchestrator(store: SQLiteSessionStore, bus: InMemoryMessageBus, knowledge_base, llm) -> NovaOrchestrator:
    return NovaOrchestrator(
        store=store,
        bus=bus,
        knowledge_base=knowledge_base,
        agents=[
            EditorialAgent(llm),
            EthicalAgent(llm),
            DialecticalAgent(llm),
            MultimodalAgent(llm),
        ],
    )
