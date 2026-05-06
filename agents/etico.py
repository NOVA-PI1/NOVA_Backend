from agents.base import BaseAgent
from schemas import AgentResult, SessionState


class EthicalAgent(BaseAgent):
    name = "etico"

    async def run(self, state: SessionState) -> AgentResult:
        context = "\n".join(hit.text for hit in state.knowledge_hits) or "No hay contexto BCL local relevante."
        system = (
            "Eres el agente etico de Nova. Evalua vigilancia epistemica, posibles sesgos, "
            "necesidad de fuentes y afirmaciones que deben tratarse con cautela. "
            "Responde en espanol con bullets breves."
        )
        user = f"Texto:\n{state.input_text}\n\nContexto BCL:\n{context}"
        output, tokens, error = await self.ask_llm(system, user)
        warnings = _extract_bullets(output) if output else _fallback_warnings(state)
        if not state.knowledge_hits:
            warnings.append("No se encontro respaldo en la BCL local; marcar fuentes externas o verificar.")
        return AgentResult(
            agent=self.name,
            output=output or "Evaluacion etica generada por reglas locales.",
            warnings=warnings,
            tokens_used=tokens,
            metadata={"knowledge_hits": len(state.knowledge_hits)},
            error=error,
        )


def _extract_bullets(text: str) -> list[str]:
    bullets = []
    for line in text.splitlines():
        clean = line.strip().lstrip("-*0123456789. ").strip()
        if clean:
            bullets.append(clean)
    return bullets[:6]


def _fallback_warnings(state: SessionState) -> list[str]:
    warnings = ["Verificar datos, cifras y nombres propios antes de publicar."]
    lowered = state.input_text.lower()
    if any(word in lowered for word in ["siempre", "nunca", "todos", "nadie"]):
        warnings.append("Revisar generalizaciones absolutas.")
    return warnings
