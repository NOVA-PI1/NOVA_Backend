from typing import Any

from agents.base import BaseAgent
from schemas import AgentResult, SessionState


class DialecticalAgent(BaseAgent):
    name = "dialectico"

    async def run(self, state: SessionState) -> AgentResult:
        latest_article = next(
            (result.output for result in reversed(state.agent_results) if result.agent == "editorial"),
            state.input_text,
        )
        ethical_review = next(
            (result.output for result in reversed(state.agent_results) if result.agent == "etico"),
            "Aun no hay revisión ética.",
        )
        system = (
            "Eres el Provocador Crítico de NOVA. No reescribes el artículo: tensionas sus "
            "supuestos. Buscas contradicciones, ángulos ciegos, voces no escuchadas, "
            "preguntas incómodas y consecuencias sociales no exploradas."
        )
        user = (
            f"Artículo:\n{latest_article}\n\n"
            f"Revisión ética disponible:\n{ethical_review}\n\n"
            "Formula entre 3 y 5 preguntas críticas y un breve mapa de tensiones. "
            "Sé incisivo, pero útil para el periodista."
        )
        output, tokens, error = await self.ask_llm(system, user, temperature=0.45)
        questions = [
            "Que supuestos del texto no pudieron ser tensionados porque el modelo no respondio?"
        ] if error else []
        return AgentResult(
            agent=self.name,
            output=output or "No se pudo generar la reflexión dialéctica.",
            questions=questions,
            tokens_used=tokens,
            error=error,
            metadata={"role": "dialectical_review"},
        )

def create_dialectical_agent(llm_instance) -> Any:
    """
    Crea el Agente Dialéctico para NOVA v2.0 utilizando CrewAI.
    """
    from crewai import Agent

    return Agent(
        role="Provocador Crítico",
        goal=(
            "Generar preguntas críticas y debates profundos al final del proceso de "
            "redacción para fomentar la reflexión del periodista humano."
        ),
        backstory=(
            "Eres un filósofo analítico. No corriges texto; tu trabajo es leer la noticia "
            "final y proponer dos o tres preguntas incómodas o ángulos ciegos que el "
            "periodista podría haber omitido. Buscas expandir la consciencia sobre el "
            "impacto social de lo que se está comunicando."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        memory=True
    )
