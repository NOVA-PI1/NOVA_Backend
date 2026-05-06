import unittest
from unittest.mock import AsyncMock

import main
from schemas import AgentResult, BusEvent, SessionResponse


class FakeOrchestrator:
    def __init__(self):
        self.run_session = AsyncMock(
            return_value=SessionResponse(
                session_id="session-1",
                status="completed",
                editorial=AgentResult(agent="editorial", output="Texto final"),
                trace=[AgentResult(agent="editorial", output="Texto final")],
            )
        )

    def get_session(self, session_id: str):
        if session_id == "missing":
            return None
        return SessionResponse(session_id=session_id, status="completed")

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
        self.assertIn("/session/{session_id}", paths)


class ApiHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_orchestrator = main.orchestrator
        main.orchestrator = FakeOrchestrator()

    def tearDown(self):
        main.orchestrator = self.original_orchestrator

    async def test_post_session_handler(self):
        response = await main.nueva_sesion(main.SessionRequest(texto="Hola Nova"))

        self.assertEqual(response.session_id, "session-1")

    async def test_get_session_not_found_handler(self):
        with self.assertRaises(main.HTTPException) as error:
            await main.obtener_sesion("missing")

        self.assertEqual(error.exception.status_code, 404)


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
