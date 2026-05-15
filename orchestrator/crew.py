# pyrefly: ignore [missing-import]
import json
from typing import Any
from crewai import Crew, Process
from agents.editorial import create_editorial_agent
from agents.etico import create_ethical_agent
from agents.dialectico import create_dialectical_agent
from agents.multimodal import create_multimodal_agent
from orchestrator.tasks import create_nova_tasks
from tools.bcl_tool import BCLTool
from schemas import (
    AgentResult,
    BusEvent,
    CanvasEditRequest,
    SessionRequest,
    SessionResponse,
    SessionState,
    utc_now,
)

class NovaCrewOrchestrator:
    def __init__(self, llm_instance, kb_service, bus, store):
        self.llm = llm_instance
        self.kb_service = kb_service
        self.bus = bus
        self.store = store
        
        self.bcl_tool = BCLTool(kb_service=self.kb_service)
        
        self.editorial = create_editorial_agent(self.llm, self.bcl_tool)
        self.etico = create_ethical_agent(self.llm)
        self.dialectico = create_dialectical_agent(self.llm)
        self.multimodal = create_multimodal_agent(self.llm)
        
        self.tasks = create_nova_tasks(
            self.editorial, 
            self.etico, 
            self.multimodal, 
            self.dialectico
        )
        
        self.crew = Crew(
            agents=[self.editorial, self.etico, self.multimodal, self.dialectico],
            tasks=self.tasks,
            process=Process.sequential,
            step_callback=self.stream_agent_thought,
            verbose=True
        )

    async def stream_agent_thought(self, step_output: Any):
        """
        Callback para streaming de 'Caja Blanca'.
        """
        event_data = {
            "agent": getattr(step_output, 'agent', 'Unknown'),
            "thought": getattr(step_output, 'thought', ''),
            "tool": getattr(step_output, 'tool', 'None'),
            "tool_input": getattr(step_output, 'tool_input', ''),
        }
        
        event = BusEvent(
            session_id="current_session", # Debería mapearse al ID real
            type="agent.thought",
            payload=event_data
        )
        await self.bus.publish(event)

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
        
        self.store.save_session(state)
        self.store.save_message(state.session_id, "user", request.texto)

        if self.llm is None:
            return self._run_fake_session(state, request)
        
        # Ejecución de la Crew
        print(f"--- Iniciando CrewAI para: {request.texto} ---")
        result = self.crew.kickoff(inputs={'tema': request.texto})
        
        # Mapeamos el resultado final (esto es simplificado, CrewAI entrega el output de la última tarea)
        state.status = "completed"
        state.updated_at = utc_now()
        
        # Guardamos un resultado genérico de la ejecución en la traza
        final_result = AgentResult(
            agent="editorial",
            output=str(result),
            tokens_used=0 # CrewAI no lo entrega directo en kickoff fácilmente sin extraer de la salida
        )
        state.agent_results.append(final_result)
        self.store.save_agent_result(state.session_id, final_result)
        self.store.save_session(state)
        
        return self._response_from_state(state)

    def _run_fake_session(self, state: SessionState, request: SessionRequest) -> SessionResponse:
        draft = (
            f"## Borrador NOVA\n\n"
            f"**Tema:** {request.texto}\n\n"
            "Esta es una respuesta local de verificación generada con LLM_PROVIDER=fake. "
            "Confirma que FastAPI, la persistencia SQLite y la integración con el frontend "
            "están respondiendo sin requerir credenciales de un proveedor LLM."
        )
        results = [
            AgentResult(agent="editorial", output=draft),
            AgentResult(
                agent="etico",
                output="Revisión ética simulada: no se detectan riesgos en esta prueba técnica.",
            ),
            AgentResult(
                agent="multimodal",
                output="Sugerencia simulada: usar una visualización simple del flujo frontend-backend.",
            ),
            AgentResult(
                agent="dialectico",
                output="Pregunta crítica simulada: ¿la interfaz muestra claramente errores del backend?",
            ),
        ]

        state.agent_results.extend(results)
        state.status = "completed"
        state.updated_at = utc_now()
        state.metadata = {**state.metadata, "mode": "fake"}

        for result in results:
            self.store.save_agent_result(state.session_id, result)
        self.store.save_message(state.session_id, "assistant", draft)
        self.store.save_session(state)

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
        return event

    def _response_from_state(self, state: SessionState) -> SessionResponse:
        by_agent: dict[str, AgentResult] = {result.agent: result for result in state.agent_results}
        return SessionResponse(
            session_id=state.session_id,
            status=state.status,
            editorial=by_agent.get("editorial"),
            etico=by_agent.get("etico"),
            dialectico=by_agent.get("dialectico"),
            multimodal=by_agent.get("multimodal"),
            knowledge_hits=[],
            trace=state.agent_results,
            metadata=state.metadata,
        )
