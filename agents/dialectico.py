from typing import Any

from agents.base import BaseAgent
from schemas import AgentResult, SessionState


class DialecticalAgent(BaseAgent):
    name = "dialectico"

    async def run(self, state: SessionState) -> AgentResult:
        target_text = str(state.metadata.get("target_text") or "").strip()
        if not target_text:
            target_text = state.input_text
        ethical_review = next(
            (result.output for result in reversed(state.agent_results) if result.agent == "etico"),
            "Aun no hay revisión ética disponible.",
        )
        web_sources = "\n\n".join(
            "\n".join(
                part
                for part in [
                    f"- {hit.title}",
                    f"  URL: {hit.url}",
                    f"  Fecha: {hit.published_at}" if hit.published_at else "",
                    f"  Snippet: {hit.snippet}" if hit.snippet else "",
                ]
                if part
            )
            for hit in state.web_hits[:5]
        ) or "No se solicitó contexto web o no hubo resultados disponibles."
        system = (
            "Eres el Provocador Crítico de NOVA, una compañera de reflexión periodística. "
            "Tu trabajo es CUESTIONAR EL ARTÍCULO QUE SE TE DA, no crear uno nuevo ni analizar tu propio proceso. "
            "Buscas: contradicciones internas, ángulos ciegos, voces ausentes, supuestos no demostrados "
            "y consecuencias sociales no exploradas. NUNCA reescribas el artículo."
        )
        user = (
            f"Texto exacto a cuestionar:\n{target_text}\n\n"
            f"Revisión ética disponible:\n{ethical_review}\n\n"
            f"Contexto web/citas, separado del texto objetivo:\n{web_sources}\n\n"
            "Formula entre 4 y 6 preguntas críticas específicas sobre el texto de arriba "
            "(cita frases o párrafos concretos cuando preguntes). "
            "Luego entrega un 'mapa de tensiones' con 3-5 puntos: qué afirmaciones necesitan más evidencia, "
            "qué voces están ausentes, y qué consecuencias no exploradas podría tener esta narrativa."
        )
        output, tokens, error = await self.ask_llm(system, user, temperature=0.5)
        questions = [
            "¿Qué supuestos del texto no pudieron ser tensionados porque el modelo no respondio?"
        ] if error else []
        return AgentResult(
            agent=self.name,
            output=output or "No se pudo generar la reflexión dialéctica.",
            questions=questions,
            tokens_used=tokens,
            error=error,
            metadata={"role": "dialectical_review", "target_text": target_text},
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
