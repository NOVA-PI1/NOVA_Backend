import tempfile
import unittest

from bcl.loader import KnowledgeBaseService
from config import Settings
from llm.providers import FakeLLMProvider
from orchestrator.graph import create_orchestrator
from schemas import SessionRequest
from services import InMemoryMessageBus, SQLiteSessionStore


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_full_backend_flow_with_fake_provider(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                llm_provider="fake",
                database_url=f"sqlite:///{temp_dir}/nova.db",
                chroma_persist_path=f"{temp_dir}/chroma",
            )
            store = SQLiteSessionStore(settings.database_url)
            bus = InMemoryMessageBus()
            knowledge_base = KnowledgeBaseService(settings)
            orchestrator = create_orchestrator(store, bus, knowledge_base, FakeLLMProvider())

            response = await orchestrator.run_session(SessionRequest(texto="Genera una nota breve."))
            restored = orchestrator.get_session(response.session_id)

            self.assertEqual(response.status, "completed")
            self.assertIsNotNone(response.editorial)
            self.assertIsNotNone(restored)
            self.assertEqual(len(restored.trace), 4)


if __name__ == "__main__":
    unittest.main()
