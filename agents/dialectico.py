from agents.base import BaseAgent
from schemas import AgentResult, SessionState


class DialecticalAgent(BaseAgent):
    name = "dialectico"

    async def run(self, state: SessionState) -> AgentResult:
        system = (
            "Eres el agente dialectico de Nova. Genera preguntas criticas, dilemas "
            "eticos/editoriales y puntos ciegos del texto. Responde en espanol."
        )
        user = f"Texto de trabajo:\n{state.input_text}"
        output, tokens, error = await self.ask_llm(system, user, temperature=0.4)
        questions = _extract_questions(output) if output else _fallback_questions()
        return AgentResult(
            agent=self.name,
            output=output or "Preguntas dialecticas generadas por reglas locales.",
            questions=questions,
            tokens_used=tokens,
            error=error,
        )


def _extract_questions(text: str) -> list[str]:
    questions = []
    for line in text.splitlines():
        clean = line.strip().lstrip("-*0123456789. ").strip()
        if clean:
            questions.append(clean)
    return questions[:6]


def _fallback_questions() -> list[str]:
    return [
        "Que evidencia sostiene la afirmacion principal?",
        "Que voces o perspectivas faltan?",
        "Que podria malinterpretar la audiencia?",
    ]
