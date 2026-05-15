from typing import Any

from agents.base import BaseAgent
from schemas import AgentResult, SessionState


class EthicalAgent(BaseAgent):
    name = "etico"

    async def run(self, state: SessionState) -> AgentResult:
        latest_article = next(
            (result.output for result in reversed(state.agent_results) if result.agent == "editorial"),
            state.input_text,
        )
        system = (
            "Eres el Auditor Ético de NOVA. Revisas contenido periodístico para detectar "
            "sesgos de género, raza, clase, territorio, fuentes ausentes, daño potencial, "
            "lenguaje estigmatizante y narrativas colonialistas. Tu salida debe ser "
            "práctica: riesgos, ajustes sugeridos y una versión corregida cuando aplique."
        )
        user = (
            f"Texto a auditar:\n{latest_article}\n\n"
            "Devuelve: 1) dictamen ético breve, 2) riesgos detectados, 3) correcciones "
            "concretas, 4) versión ajustada de los fragmentos problemáticos."
        )
        output, tokens, error = await self.ask_llm(system, user, temperature=0.2)
        warnings = ["No se pudo completar la auditoria etica con el modelo configurado."] if error else []
        return AgentResult(
            agent=self.name,
            output=output or "No se pudo generar la revisión ética.",
            warnings=warnings,
            tokens_used=tokens,
            error=error,
            metadata={"role": "ethical_review"},
        )

def create_ethical_agent(llm_instance) -> Any:
    """
    Crea el Agente Ético para NOVA v2.0 utilizando CrewAI.
    """
    from crewai import Agent

    return Agent(
        role="Auditor Ético y Especialista en Sesgos Algorítmicos",
        goal=(
            "Analizar borradores periodísticos para detectar y corregir sesgos de género, "
            "raza, geografía y evitar narrativas impuestas por el Norte Global."
        ),
        backstory=(
            "Eres un auditor implacable que asegura un lenguaje equitativo y transparente. "
            "Tu misión es garantizar que las noticias de NOVA no utilicen masculinidades "
            "genéricas innecesarias y que las fuentes representen la diversidad real del "
            "contexto regional, eliminando cualquier rastro de sesgo colonialista."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        memory=True
    )
