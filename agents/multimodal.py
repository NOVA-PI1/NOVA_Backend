from typing import Any

from agents.base import BaseAgent
from schemas import AgentResult, SessionState


class MultimodalAgent(BaseAgent):
    name = "multimodal"

    async def run(self, state: SessionState) -> AgentResult:
        if not state.metadata.get("images"):
            return AgentResult(
                agent=self.name,
                output=(
                    "Sin imagenes adjuntas. Sugerencia base: definir una visual editorial "
                    "sobria, contextual y no estereotipada si el articulo lo requiere."
                ),
                metadata={"role": "visual_direction", "mode": "stub"},
            )

        latest_article = next(
            (result.output for result in reversed(state.agent_results) if result.agent == "editorial"),
            state.input_text,
        )
        system = (
            "Eres el Director de Arte y Diversidad Visual de NOVA. Propones tratamiento "
            "visual periodístico evitando estereotipos, exotización y representaciones "
            "genéricas de Latinoamérica. No generas imágenes; produces prompts y criterios."
        )
        user = (
            f"Artículo base:\n{latest_article}\n\n"
            "Entrega: 1) necesidad visual principal, 2) prompt para imagen editorial "
            "realista, 3) prompt para gráfico o infografía si aplica, 4) alertas de "
            "representación que deben evitarse."
        )
        output, tokens, error = await self.ask_llm(system, user, temperature=0.3)
        return AgentResult(
            agent=self.name,
            output=output or "No se pudo generar la propuesta multimodal.",
            tokens_used=tokens,
            error=error,
            metadata={"role": "visual_direction", "mode": "llm"},
        )

def create_multimodal_agent(llm_instance) -> Any:
    """
    Crea el Agente Multimodal para NOVA v2.0 utilizando CrewAI.
    """
    from crewai import Agent

    return Agent(
        role="Director de Arte y Diversidad Visual",
        goal=(
            "Detectar la necesidad de apoyo visual en la noticia y generar prompts "
            "detallados (para DALL-E u otros) que aseguren representación diversa."
        ),
        backstory=(
            "Eres un curador visual enfocado en evitar los estereotipos visuales que la IA "
            "suele generar sobre Latinoamérica. Tus prompts describen escenas realistas, "
            "dignas y representativas de la etnia, género y contexto regional del Sur Global."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        memory=True
    )
