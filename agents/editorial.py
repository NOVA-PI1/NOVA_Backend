from agents.base import BaseAgent
from schemas import AgentResult, SessionState


class EditorialAgent(BaseAgent):
    name = "editorial"

    async def run(self, state: SessionState) -> AgentResult:
        ethical = next((result for result in state.agent_results if result.agent == "etico"), None)
        dialectical = next((result for result in state.agent_results if result.agent == "dialectico"), None)
        warnings = "\n".join(ethical.warnings if ethical else [])
        questions = "\n".join(dialectical.questions if dialectical else [])
        context = "\n".join(hit.text for hit in state.knowledge_hits) or "Sin contexto BCL local."

        system = (
            "Eres un agente editorial periodistico latinoamericano. Corrige y redacta con claridad, "
            "lenguaje inclusivo cuando corresponda, y senala informacion no verificada. "
            "Formato sugerido: titulo, entradilla y cuerpo."
        )
        user = (
            f"Texto base:\n{state.input_text}\n\n"
            f"Contexto BCL:\n{context}\n\n"
            f"Alertas eticas:\n{warnings or 'Sin alertas previas.'}\n\n"
            f"Preguntas dialecticas:\n{questions or 'Sin preguntas previas.'}"
        )
        output, tokens, error = await self.ask_llm(system, user, temperature=0.3)
        return AgentResult(
            agent=self.name,
            output=output or _fallback_editorial(state.input_text),
            warnings=[] if output else ["Salida editorial generada por fallback local."],
            tokens_used=tokens,
            metadata={"used_ethical_context": ethical is not None, "used_dialectical_context": dialectical is not None},
            error=error,
        )


def _fallback_editorial(text: str) -> str:
    return (
        "Titulo: Borrador editorial pendiente de revision\n\n"
        "Entradilla: Se genero una version local porque el proveedor LLM no respondio.\n\n"
        f"Cuerpo: {text}"
    )
