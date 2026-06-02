import unittest

from agents.dialectico import DialecticalAgent
from agents.editorial import EditorialAgent
from agents.etico import EthicalAgent
from agents.multimodal import MultimodalAgent
from schemas import AgentResult, LLMRequest, SessionState


class FailingLLM:
    name = "failing"

    async def generate(self, request: LLMRequest):
        raise RuntimeError("model unavailable")


class CapturingLLM:
    name = "capturing"

    def __init__(self):
        self.last_request = None

    async def generate(self, request: LLMRequest):
        from schemas import LLMResponse

        self.last_request = request
        return LLMResponse(text="¿Qué evidencia falta?\n- Falta una voz directa.", model="test", provider="capturing")


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

    async def test_dialectical_agent_uses_target_text_over_editorial_output(self):
        llm = CapturingLLM()
        state = SessionState(input_text="Texto original", metadata={"target_text": "Borrador activo"})
        state.agent_results.append(AgentResult(agent="editorial", output="Narrativa inventada"))

        result = await DialecticalAgent(llm).run(state)
        user_message = llm.last_request.messages[-1].content

        self.assertEqual(result.metadata["target_text"], "Borrador activo")
        self.assertIn("Borrador activo", user_message)
        self.assertNotIn("Narrativa inventada", user_message)

    async def test_dialectical_agent_falls_back_to_input_text_not_editorial(self):
        llm = CapturingLLM()
        state = SessionState(input_text="Texto original", metadata={"target_text": ""})
        state.agent_results.append(AgentResult(agent="editorial", output="Narrativa inventada"))

        result = await DialecticalAgent(llm).run(state)
        user_message = llm.last_request.messages[-1].content

        self.assertEqual(result.metadata["target_text"], "Texto original")
        self.assertIn("Texto original", user_message)
        self.assertNotIn("Narrativa inventada", user_message)

if __name__ == "__main__":
    unittest.main()
