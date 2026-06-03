import tempfile
import unittest

from bcl.loader import KnowledgeBaseService
from config import Settings
from llm.providers import FakeLLMProvider
from orchestrator.graph import create_orchestrator
from schemas import DraftCreateRequest, DriveDocumentRequest, SessionRequest
from services import InMemoryMessageBus, SQLiteSessionStore


class MockDriveService:
    async def apply_document_action(
        self,
        *,
        action,
        access_token,
        session_id,
        title,
        content,
        existing_document=None,
    ):
        if action == "delete":
            return None
        document_id = (existing_document or {}).get("document_id", "doc-1")
        return {
            "document_id": document_id,
            "url": f"https://docs.google.com/document/d/{document_id}/edit",
            "last_synced_at": "2026-06-02T00:00:00+00:00",
            "shared": True,
        }


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

    async def test_drafts_questions_format_and_drive_flow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                llm_provider="fake",
                database_url=f"sqlite:///{temp_dir}/nova.db",
                chroma_persist_path=f"{temp_dir}/chroma",
                web_search_enabled=True,
                web_search_provider="fake",
            )
            store = SQLiteSessionStore(settings.database_url)
            bus = InMemoryMessageBus()
            knowledge_base = KnowledgeBaseService(settings)
            orchestrator = create_orchestrator(
                store,
                bus,
                knowledge_base,
                FakeLLMProvider(),
                drive=MockDriveService(),
            )

            first = await orchestrator.run_session(SessionRequest(texto="Genera una nota breve."))
            manual = orchestrator.create_draft(
                first.session_id,
                DraftCreateRequest(content="Texto manual del canvas", source="canvas"),
            )
            revised = await orchestrator.run_session(
                SessionRequest(
                    session_id=first.session_id,
                    texto="Hazlo más directo.",
                    operation="revise",
                    target_draft_id=manual.id,
                    use_web_context=True,
                )
            )
            questioned = await orchestrator.run_session(
                SessionRequest(session_id=first.session_id, texto="¿Qué falta?", operation="question", target_draft_id=manual.id)
            )
            formatted = await orchestrator.run_session(
                SessionRequest(
                    session_id=first.session_id,
                    texto="Adapta para redes.",
                    operation="format",
                    output_format="twitter_thread",
                    target_draft_id=manual.id,
                )
            )
            document = await orchestrator.apply_drive_action(
                first.session_id,
                DriveDocumentRequest(action="create", draft_id=manual.id),
                access_token="token",
            )
            deleted = await orchestrator.apply_drive_action(
                first.session_id,
                DriveDocumentRequest(action="delete"),
                access_token="token",
            )

            drafts = revised.drafts
            versions = [d.version for d in drafts]
            contents = [d.content for d in drafts]

            self.assertGreaterEqual(len(revised.drafts), 3)
            self.assertEqual(questioned.dialectico.metadata["target_text"], "Texto manual del canvas")
            self.assertIn("twitter_thread", formatted.social_outputs)
            self.assertEqual(revised.web_hits[0].source, "fake")
            self.assertEqual(document["document_id"], "doc-1")
            self.assertIsNone(deleted)
            self.assertEqual(versions, sorted(versions))
            self.assertEqual(len(versions), len(set(versions)))
            self.assertIn("Texto manual del canvas", contents)

    async def test_delete_session_removes_saved_history(self):
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
            deleted = orchestrator.delete_session(response.session_id)

            self.assertTrue(deleted)
            self.assertIsNone(orchestrator.get_session(response.session_id))
            self.assertEqual(orchestrator.list_sessions(), [])


if __name__ == "__main__":
    unittest.main()
