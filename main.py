from fastapi import FastAPI, HTTPException
import socketio

from bcl.loader import KnowledgeBaseService
from config import get_settings
from llm import create_llm_provider
from orchestrator.graph import create_orchestrator
from schemas import CanvasEditRequest, SessionRequest, SessionResponse
from services import InMemoryMessageBus, SQLiteSessionStore


settings = get_settings()
store = SQLiteSessionStore(settings.database_url)
bus = InMemoryMessageBus()
knowledge_base = KnowledgeBaseService(settings)
llm = create_llm_provider(settings)
orchestrator = create_orchestrator(store, bus, knowledge_base, llm)

app = FastAPI(title=settings.app_name)
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=settings.cors_allowed_origins)
socket_app = socketio.ASGIApp(sio, app)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }


@app.post("/session", response_model=SessionResponse)
async def nueva_sesion(data: SessionRequest) -> SessionResponse:
    return await orchestrator.run_session(data)


@app.get("/session/{session_id}", response_model=SessionResponse)
async def obtener_sesion(session_id: str) -> SessionResponse:
    result = orchestrator.get_session(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@sio.on("editar_canvas")
async def on_editar(sid, data):
    event = await orchestrator.handle_canvas_edit(CanvasEditRequest(**data))
    await sio.emit(
        "actualizar_chat",
        {
            "msg": "Canvas actualizado",
            "event": event.model_dump(mode="json"),
        },
        to=sid,
    )
