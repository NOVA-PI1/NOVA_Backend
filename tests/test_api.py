import unittest
import os
from unittest.mock import AsyncMock

os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import main
from schemas import AgentResult, BusEvent, DraftRevision, SessionResponse, SessionSummary, utc_now


class FakeOrchestrator:
    def __init__(self):
        async def run_session_side_effect(request):
            if request.target_draft_id == 404:
                raise ValueError("Draft 404 not found for session session-1")
            return SessionResponse(
                session_id="session-1",
                status="completed",
                editorial=AgentResult(agent="editorial", output="Texto final"),
                trace=[AgentResult(agent="editorial", output="Texto final")],
                metadata={
                    "display_status": "Listo para editar",
                    "completed_agents": 1,
                    "active_word_count": 2,
                    "markdown_enabled": True,
                },
            )

        self.run_session = AsyncMock(
            side_effect=run_session_side_effect
        )

    def get_session(self, session_id: str):
        if session_id == "missing":
            return None
        return SessionResponse(session_id=session_id, status="completed")

    def list_sessions(self, user_id=None, limit=50):
        return [
            SessionSummary(
                session_id="session-1",
                title="Texto final",
                status="completed",
                created_at=utc_now(),
                updated_at=utc_now(),
                user_id=user_id,
            )
        ]

    def list_drafts(self, session_id):
        return [DraftRevision(session_id=session_id, version=1, content="Texto final")]

    def create_draft(self, session_id, request):
        return DraftRevision(session_id=session_id, version=2, content=request.content, source=request.source)

    async def suggest_questions(self, session_id, request):
        return ["¿Qué evidencia falta?"]

    async def apply_drive_action(self, session_id, request, *, access_token=None):
        if request.action == "delete":
            return None
        return {
            "document_id": "doc-1",
            "url": "https://docs.google.com/document/d/doc-1/edit",
            "last_synced_at": utc_now().isoformat(),
            "shared": True,
        }

    async def handle_canvas_edit(self, request):
        return BusEvent(session_id=request.session_id or "canvas", type="canvas.edited")


class FakeSIO:
    def __init__(self):
        self.emitted = []

    async def emit(self, event, data, to=None):
        self.emitted.append((event, data, to))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.original_orchestrator = main.orchestrator
        self.original_sio = main.sio
        main.orchestrator = FakeOrchestrator()

    def tearDown(self):
        main.orchestrator = self.original_orchestrator
        main.sio = self.original_sio

    def test_app_exposes_session_routes(self):
        paths = {route.path for route in main.app.routes}

        self.assertIn("/session", paths)
        self.assertIn("/sessions", paths)
        self.assertIn("/session/{session_id}", paths)


class ApiHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_orchestrator = main.orchestrator
        main.orchestrator = FakeOrchestrator()

    def tearDown(self):
        main.orchestrator = self.original_orchestrator

    async def test_post_session_handler(self):
        response = await main.nueva_sesion(main.SessionRequest(texto="Hola Nova"), None)

        self.assertEqual(response.session_id, "session-1")
        self.assertTrue(response.metadata["markdown_enabled"])

    async def test_post_session_invalid_draft_returns_clear_not_found(self):
        with self.assertRaises(main.HTTPException) as error:
            await main.nueva_sesion(main.SessionRequest(texto="Hola Nova", session_id="session-1", target_draft_id=404), None)

        self.assertEqual(error.exception.status_code, 404)
        self.assertIn("Draft 404", error.exception.detail)

    async def test_get_session_not_found_handler(self):
        with self.assertRaises(main.HTTPException) as error:
            await main.obtener_sesion("missing", None)

        self.assertEqual(error.exception.status_code, 404)
        self.assertIn("No encontré la sesión", error.exception.detail)

    async def test_list_sessions_handler(self):
        response = await main.listar_sesiones(None, user_id="user-1", limit=10)

        self.assertEqual(response[0].session_id, "session-1")
        self.assertEqual(response[0].user_id, "user-1")

    async def test_draft_and_question_handlers(self):
        draft = await main.crear_borrador("session-1", main.DraftCreateRequest(content="Nuevo texto"), None)
        questions = await main.sugerir_preguntas("session-1", main.QuestionsRequest(text="Nuevo texto"), None)

        self.assertEqual(draft.version, 2)
        self.assertEqual(questions, ["¿Qué evidencia falta?"])

    async def test_drive_handler(self):
        response = await main.sincronizar_drive("session-1", main.DriveDocumentRequest(action="create"), None)

        self.assertEqual(response["drive_document"]["document_id"], "doc-1")


class SocketTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_orchestrator = main.orchestrator
        self.original_sio = main.sio
        main.orchestrator = FakeOrchestrator()

    def tearDown(self):
        main.orchestrator = self.original_orchestrator
        main.sio = self.original_sio

    async def test_socket_canvas_edit_emits_update(self):
        fake_sio = FakeSIO()
        main.sio = fake_sio

        await main.on_editar("sid-1", {"session_id": "session-1", "texto": "Cambio"})

        self.assertEqual(fake_sio.emitted[0][0], "actualizar_chat")
        self.assertEqual(fake_sio.emitted[0][2], "sid-1")


if __name__ == "__main__":
    unittest.main()
