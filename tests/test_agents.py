import unittest

from agents.dialectico import DialecticalAgent
from agents.editorial import EditorialAgent
from agents.etico import EthicalAgent
from agents.multimodal import MultimodalAgent
from schemas import LLMRequest, SessionState


class FailingLLM:
    name = "failing"

    async def generate(self, request: LLMRequest):
        raise RuntimeError("model unavailable")


class AgentStabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = SessionState(input_text="Texto con una afirmacion que necesita verificacion.")
        self.llm = FailingLLM()

    async def test_ethical_agent_returns_stable_result_on_llm_failure(self):
        result = await EthicalAgent(self.llm).run(self.state)

        self.assertEqual(result.agent, "etico")
        self.assertTrue(result.warnings)
        self.assertEqual(result.error, "model unavailable")

    async def test_dialectical_agent_returns_stable_result_on_llm_failure(self):
        result = await DialecticalAgent(self.llm).run(self.state)

        self.assertEqual(result.agent, "dialectico")
        self.assertTrue(result.questions)
        self.assertEqual(result.error, "model unavailable")

    async def test_editorial_agent_returns_stable_result_on_llm_failure(self):
        result = await EditorialAgent(self.llm).run(self.state)

        self.assertEqual(result.agent, "editorial")
        self.assertIn("Titulo:", result.output)
        self.assertEqual(result.error, "model unavailable")

    async def test_multimodal_agent_returns_stub_without_images(self):
        result = await MultimodalAgent(self.llm).run(self.state)

        self.assertEqual(result.agent, "multimodal")
        self.assertEqual(result.metadata["mode"], "stub")


if __name__ == "__main__":
    unittest.main()
