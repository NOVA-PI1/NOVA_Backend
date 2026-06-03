from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import socketio

from auth import (
    configured_providers,
    create_google_authorize_redirect,
    create_jwt,
    exchange_google_code,
    get_current_user_from_request,
    read_oauth_state,
    redirect_with_auth_error,
)
from bcl.loader import KnowledgeBaseService
from config import get_settings
from llm import create_llm_provider
from orchestrator.graph import create_orchestrator
from schemas import (
    CanvasEditRequest,
    DraftCreateRequest,
    DraftRevision,
    DriveDocumentRequest,
    QuestionsRequest,
    SessionRequest,
    SessionResponse,
    BusEvent,
    SessionSummary,
)
from services import InMemoryMessageBus, create_session_store
from services.drive import GoogleDriveService
from services.web_search import WebSearchService


settings = get_settings()
store = create_session_store(settings.database_url)
bus = InMemoryMessageBus()
knowledge_base = KnowledgeBaseService(settings)
llm_provider = create_llm_provider(settings)
web_search = WebSearchService(settings)
drive = GoogleDriveService()
orchestrator = create_orchestrator(store, bus, knowledge_base, llm_provider, web_search, drive)

app = FastAPI(title=settings.app_name)


def parse_cors_origins(raw_origins: str) -> list[str] | str:
    if raw_origins.strip() == "*":
        return "*"
    origins: list[str] = []
    for origin in raw_origins.split(","):
        origin = origin.strip()
        if not origin:
            continue
        origins.append(origin if "://" in origin else f"https://{origin}")
    return origins or "*"


cors_origins = parse_cors_origins(settings.cors_allowed_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if cors_origins == "*" else cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=cors_origins)
socket_app = socketio.ASGIApp(sio, app)

# Puente: Todo lo que llegue al bus se emite por Socket.IO al frontend
async def bridge_bus_to_sio(event: BusEvent):
    await sio.emit("agent_event", event.model_dump(mode="json"))

bus.subscribe(bridge_bus_to_sio)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }


@app.get("/auth/providers")
async def auth_providers() -> dict:
    return {
        "auth_required": settings.auth_required,
        "providers": configured_providers(settings),
    }


@app.get("/auth/google/authorize")
async def google_authorize(request: Request, frontend_redirect_url: str | None = None) -> RedirectResponse:
    redirect_url = frontend_redirect_url or settings.frontend_base_url
    return create_google_authorize_redirect(request, settings, redirect_url)


@app.get("/auth/google/callback", name="google_callback")
async def google_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    frontend_redirect_url = settings.frontend_base_url
    if state:
        try:
            frontend_redirect_url = read_oauth_state(state, settings)
        except HTTPException:
            pass

    if error:
        return redirect_with_auth_error(frontend_redirect_url, error)
    if not code:
        return redirect_with_auth_error(frontend_redirect_url, "missing_code")

    try:
        user = await exchange_google_code(code, request, settings)
        token = create_jwt(
            {"sub": user["user_id"], "provider": "google", "user": user},
            settings.secret_key,
            settings.jwt_expiration_minutes,
        )
    except HTTPException as auth_error:
        return redirect_with_auth_error(frontend_redirect_url, str(auth_error.detail))

    query = urlencode({"auth_token": token})
    separator = "&" if "?" in frontend_redirect_url else "?"
    return RedirectResponse(f"{frontend_redirect_url}{separator}{query}", status_code=302)


@app.get("/auth/me")
async def auth_me(request: Request) -> dict:
    user = get_current_user_from_request(request, settings)
    return {"authenticated": True, "user": user}


def require_user_when_enabled(request: Request) -> dict | None:
    if not settings.auth_required:
        return None
    return get_current_user_from_request(request, settings)


def ensure_session_access(session_id: str, request: Request) -> tuple[SessionResponse, dict | None]:
    user = require_user_when_enabled(request)
    result = orchestrator.get_session(session_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No encontré la sesión '{session_id}'. Carga otra conversación o inicia una nueva.",
        )
    if user and result.metadata.get("user_id") not in {None, user["user_id"]}:
        raise HTTPException(
            status_code=403,
            detail="Esta sesión pertenece a otro usuario autenticado.",
        )
    return result, user


@app.post("/session", response_model=SessionResponse)
async def nueva_sesion(data: SessionRequest, request: Request) -> SessionResponse:
    user = require_user_when_enabled(request)
    if data.session_id:
        existing = orchestrator.get_session(data.session_id)
        if user and existing and existing.metadata.get("user_id") not in {None, user["user_id"]}:
            raise HTTPException(
                status_code=403,
                detail="No puedes continuar una sesión que pertenece a otro usuario.",
            )
    if user:
        data.perfil.update(
            {
                "user_id": user["user_id"],
                "user_name": user.get("name"),
                "user_email": user.get("email"),
                "auth_provider": user.get("provider"),
            }
        )
        data.metadata["user_id"] = user["user_id"]
    try:
        return await orchestrator.run_session(data)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error) or "No pude ejecutar la sesión con el borrador indicado.") from None


@app.get("/sessions", response_model=list[SessionSummary])
async def listar_sesiones(
    request: Request,
    user_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[SessionSummary]:
    user = require_user_when_enabled(request)
    if user:
        user_id = user["user_id"]
    return orchestrator.list_sessions(user_id=user_id, limit=limit)


@app.get("/session/{session_id}", response_model=SessionResponse)
async def obtener_sesion(session_id: str, request: Request) -> SessionResponse:
    result, _ = ensure_session_access(session_id, request)
    return result


@app.get("/session/{session_id}/drafts", response_model=list[DraftRevision])
async def listar_borradores(session_id: str, request: Request) -> list[DraftRevision]:
    ensure_session_access(session_id, request)
    return orchestrator.list_drafts(session_id)


@app.post("/session/{session_id}/drafts", response_model=DraftRevision)
async def crear_borrador(session_id: str, data: DraftCreateRequest, request: Request) -> DraftRevision:
    ensure_session_access(session_id, request)
    try:
        return orchestrator.create_draft(session_id, data)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error) or "No pude guardar el borrador en esta sesión.") from None


@app.post("/session/{session_id}/questions", response_model=list[str])
async def sugerir_preguntas(session_id: str, data: QuestionsRequest, request: Request) -> list[str]:
    ensure_session_access(session_id, request)
    try:
        return await orchestrator.suggest_questions(session_id, data)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error) or "No pude generar preguntas para esta sesión.") from None


@app.post("/session/{session_id}/drive")
async def sincronizar_drive(session_id: str, data: DriveDocumentRequest, request: Request) -> dict:
    _, user = ensure_session_access(session_id, request)
    access_token = user.get("google_access_token") if user else None
    try:
        document = await orchestrator.apply_drive_action(session_id, data, access_token=access_token)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error) or "No pude sincronizar Drive para esta sesión.") from None
    return {"drive_document": document}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:socket_app", host="0.0.0.0", port=8000, reload=True)
