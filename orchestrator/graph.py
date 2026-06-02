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
    DraftCreateRequest,
    DraftRevision,
    DriveDocumentRequest,
    QuestionsRequest,
    SessionRequest,
    SessionResponse,
    SessionSummary,
    SessionState,
    utc_now,
)
from services import InMemoryMessageBus
from services.drive import GoogleDriveService
from services.web_search import WebSearchService


class NovaOrchestrator:
    def __init__(
        self,
        *,
        store,
        bus: InMemoryMessageBus,
        knowledge_base: KnowledgeBaseService,
        web_search: WebSearchService,
        drive: GoogleDriveService,
        agents: Iterable[object],
    ) -> None:
        self.store = store
        self.bus = bus
        self.knowledge_base = knowledge_base
        self.web_search = web_search
        self.drive = drive
        self.agents = list(agents)

    async def run_session(self, request: SessionRequest) -> SessionResponse:
        existing_state = self.store.get_session(request.session_id) if request.session_id else None
        target_draft = self._resolve_target_draft(request.session_id, request.target_draft_id) if request.session_id else None
        target_text = target_draft.content if target_draft else self._active_text(existing_state) if existing_state else request.texto
        active_draft_text = self._active_text(existing_state) if existing_state else ""

        state_data = {
            "input_text": request.texto,
            "perfil": {
                **(existing_state.perfil if existing_state else {}),
                **request.perfil,
            },
            "metadata": {
                **(existing_state.metadata if existing_state else {}),
                **request.metadata,
                "images": request.images,
                "operation": request.operation,
                "output_format": request.output_format,
                "target_draft_id": request.target_draft_id,
                "target_text": target_text,
                "active_draft_text": active_draft_text,
                "use_web_context": request.use_web_context,
            },
            "status": "running",
        }
        if request.session_id:
            state_data["session_id"] = request.session_id
        state = SessionState(**state_data)

        state.knowledge_hits = self.knowledge_base.search(request.texto)
        state.web_hits = await self.web_search.search(request.texto) if request.use_web_context else []
        self.store.save_session(state)
        self.store.save_message(state.session_id, "user", request.texto)
        await self._publish(state.session_id, "session.started", {"status": state.status})

        for agent in self._agents_for_operation(request.operation):
            result = await agent.run(state)
            state.agent_results.append(result)
            state.updated_at = utc_now()
            self.store.save_agent_result(state.session_id, result)
            self.store.save_session(state)
            await self._publish(state.session_id, f"agent.{result.agent}.completed", result.model_dump(mode="json"))

        state.status = "completed"
        state.updated_at = utc_now()
        self.store.save_session(state)
        self._persist_outputs_as_drafts(state, request, target_draft)
        await self._publish(state.session_id, "session.completed", {"status": state.status})
        return self._response_from_state(state)

    def get_session(self, session_id: str) -> SessionResponse | None:
        state = self.store.get_session(session_id)
        if state is None:
            return None
        return self._response_from_state(state)

    def list_sessions(self, user_id: str | None = None, limit: int = 50) -> list[SessionSummary]:
        return self.store.list_sessions(user_id=user_id, limit=limit)

    def list_drafts(self, session_id: str) -> list[DraftRevision]:
        return self.store.list_draft_revisions(session_id)

    def create_draft(self, session_id: str, request: DraftCreateRequest) -> DraftRevision:
        state = self.store.get_session(session_id)
        if state is None:
            raise ValueError("Session not found")
        draft = self.store.save_draft_revision(
            DraftRevision(
                session_id=session_id,
                version=0,
                content=request.content,
                source=request.source,
                instruction=request.instruction,
                agent=request.agent,
                metadata=request.metadata,
            )
        )
        state.metadata = {**state.metadata, "current_draft_id": draft.id}
        state.updated_at = utc_now()
        self.store.save_session(state)
        return draft

    async def suggest_questions(self, session_id: str, request: QuestionsRequest) -> list[str]:
        state = self.store.get_session(session_id)
        if state is None:
            raise ValueError("Session not found")
        draft = self._resolve_target_draft(session_id, request.draft_id)
        target_text = request.text or (draft.content if draft else self._active_text(state))
        state.metadata = {**state.metadata, "target_text": target_text, "question_count": request.count}
        agent = next((agent for agent in self.agents if getattr(agent, "name", None) == "dialectico"), None)
        if agent is None:
            return self._fallback_questions(target_text, request.count)
        result = await agent.run(state)
        self.store.save_agent_result(session_id, result)
        questions = result.questions or self._extract_questions(result.output) or self._fallback_questions(target_text, request.count)
        state.metadata = {**state.metadata, "suggested_questions": questions[: request.count]}
        state.updated_at = utc_now()
        self.store.save_session(state)
        return questions[: request.count]

    async def handle_canvas_edit(self, request: CanvasEditRequest) -> BusEvent:
        session_id = request.session_id or "canvas"
        saved_draft = None
        if request.texto and request.session_id:
            saved_draft = self.create_draft(
                request.session_id,
                DraftCreateRequest(
                    content=request.texto,
                    source="canvas_edit",
                    instruction=request.metadata.get("instruction"),
                    agent="editorial",
                    metadata={"canvas": request.canvas, **request.metadata},
                ),
            )
            self.store.save_message(session_id, "canvas_revision", request.texto)

        payload = {
            "texto": request.texto,
            "canvas": request.canvas,
            "metadata": request.metadata,
            "draft_saved": saved_draft is not None,
            "draft": saved_draft.model_dump(mode="json") if saved_draft else None,
        }
        event = BusEvent(session_id=session_id, type="canvas.edited", payload=payload)
        await self.bus.publish(event)
        self.store.save_event(event)
        return event

    async def apply_drive_action(
        self,
        session_id: str,
        request: DriveDocumentRequest,
        *,
        access_token: str | None,
    ) -> dict | None:
        state = self.store.get_session(session_id)
        if state is None:
            raise ValueError("Session not found")
        draft = self._resolve_target_draft(session_id, request.draft_id)
        content = request.content or (draft.content if draft else self._active_text(state))
        document = await self.drive.apply_document_action(
            action=request.action,
            access_token=access_token,
            session_id=session_id,
            title=request.title or self._title_for_state(state),
            content=content,
            existing_document=state.metadata.get("drive_document"),
        )
        metadata = dict(state.metadata)
        if document:
            metadata["drive_document"] = document
        else:
            metadata.pop("drive_document", None)
        state.metadata = metadata
        state.updated_at = utc_now()
        self.store.save_session(state)
        return document

    async def _publish(self, session_id: str, event_type: str, payload: dict) -> None:
        event = BusEvent(session_id=session_id, type=event_type, payload=payload)
        await self.bus.publish(event)
        self.store.save_event(event)

    def _response_from_state(self, state: SessionState) -> SessionResponse:
        by_agent: dict[str, AgentResult] = {result.agent: result for result in state.agent_results}
        drafts = self.store.list_draft_revisions(state.session_id)
        return SessionResponse(
            session_id=state.session_id,
            input_text=state.input_text,
            status=state.status,
            editorial=by_agent.get("editorial"),
            etico=by_agent.get("etico"),
            dialectico=by_agent.get("dialectico"),
            multimodal=by_agent.get("multimodal"),
            knowledge_hits=state.knowledge_hits,
            web_hits=state.web_hits,
            trace=state.agent_results,
            metadata=state.metadata,
            drafts=drafts,
            current_draft=self._current_draft(state, drafts),
            suggested_questions=self._suggested_questions(state, by_agent),
            social_outputs=self._social_outputs(state),
            drive_document=state.metadata.get("drive_document"),
        )

    def _persist_outputs_as_drafts(
        self,
        state: SessionState,
        request: SessionRequest,
        target_draft: DraftRevision | None,
    ) -> None:
        editorial = next((result for result in reversed(state.agent_results) if result.agent == "editorial"), None)
        multimodal = next((result for result in reversed(state.agent_results) if result.agent == "multimodal"), None)
        if request.operation in {"generate", "revise"} and editorial:
            draft = self.store.save_draft_revision(
                DraftRevision(
                    session_id=state.session_id,
                    version=0,
                    content=editorial.output,
                    source="revision" if target_draft or request.operation == "revise" else "generation",
                    instruction=request.texto,
                    agent="editorial",
                    metadata={"operation": request.operation, "target_draft_id": target_draft.id if target_draft else None},
                )
            )
            state.metadata = {**state.metadata, "current_draft_id": draft.id}
            self.store.save_session(state)
        if request.operation == "format" and multimodal:
            outputs = dict(state.metadata.get("social_outputs", {}))
            outputs[request.output_format] = multimodal.output
            state.metadata = {**state.metadata, "social_outputs": outputs}
            self.store.save_session(state)

    def _resolve_target_draft(self, session_id: str | None, draft_id: int | None) -> DraftRevision | None:
        if not session_id:
            return None
        if draft_id:
            return self.store.get_draft_revision(session_id, draft_id)
        state = self.store.get_session(session_id)
        current_id = state.metadata.get("current_draft_id") if state else None
        if current_id:
            try:
                draft = self.store.get_draft_revision(session_id, int(current_id))
            except (TypeError, ValueError):
                draft = None
            if draft:
                return draft
        drafts = self.store.list_draft_revisions(session_id)
        return drafts[-1] if drafts else None

    def _agents_for_operation(self, operation: str) -> list[object]:
        if operation == "question":
            return [agent for agent in self.agents if getattr(agent, "name", None) == "dialectico"]
        if operation == "format":
            return [agent for agent in self.agents if getattr(agent, "name", None) == "multimodal"]
        return self.agents

    def _active_text(self, state: SessionState | None) -> str:
        if state is None:
            return ""
        drafts = self.store.list_draft_revisions(state.session_id)
        if drafts:
            return drafts[-1].content
        metadata_target = str(state.metadata.get("target_text") or "").strip()
        if metadata_target:
            return metadata_target
        return state.input_text

    def _current_draft(self, state: SessionState, drafts: list[DraftRevision]) -> DraftRevision | None:
        current_id = state.metadata.get("current_draft_id")
        if current_id:
            match = next((draft for draft in drafts if draft.id == current_id), None)
            if match:
                return match
        return drafts[-1] if drafts else None

    def _suggested_questions(self, state: SessionState, by_agent: dict[str, AgentResult]) -> list[str]:
        metadata_questions = state.metadata.get("suggested_questions")
        if isinstance(metadata_questions, list):
            return [str(question) for question in metadata_questions]
        dialectical = by_agent.get("dialectico")
        if dialectical and dialectical.questions:
            return dialectical.questions
        if dialectical:
            return self._extract_questions(dialectical.output)
        return self._fallback_questions(self._active_text(state), 6)

    def _social_outputs(self, state: SessionState) -> dict[str, str]:
        outputs = state.metadata.get("social_outputs")
        return outputs if isinstance(outputs, dict) else {}

    def _extract_questions(self, text: str) -> list[str]:
        questions = []
        for line in text.splitlines():
            clean = line.strip(" -0123456789.)")
            if "?" in clean or "¿" in clean:
                questions.append(clean)
        return questions[:8]

    def _fallback_questions(self, text: str, count: int) -> list[str]:
        excerpt = text[:120].strip() or "este texto"
        base = [
            f"¿Qué afirmación central de '{excerpt}' necesita más evidencia?",
            "¿Qué voces directamente afectadas todavía no aparecen?",
            "¿Qué supuesto de contexto podría estar guiando la narrativa?",
            "¿Qué dato cambiaría la lectura principal del texto?",
            "¿Qué consecuencia social merece una línea adicional?",
            "¿Qué parte debería verificarse antes de publicar?",
        ]
        return base[:count]

    def _title_for_state(self, state: SessionState) -> str:
        title = state.metadata.get("title") or state.input_text or state.session_id
        return " ".join(str(title).split())[:90]


def create_orchestrator(store, bus: InMemoryMessageBus, knowledge_base, llm, web_search=None, drive=None) -> NovaOrchestrator:
    return NovaOrchestrator(
        store=store,
        bus=bus,
        knowledge_base=knowledge_base,
        web_search=web_search or WebSearchService(knowledge_base.settings),
        drive=drive or GoogleDriveService(),
        agents=[
            EditorialAgent(llm),
            EthicalAgent(llm),
            DialecticalAgent(llm),
            MultimodalAgent(llm),
        ],
    )
