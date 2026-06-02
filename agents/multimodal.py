from typing import Any

from agents.base import BaseAgent
from schemas import AgentResult, SessionState


class MultimodalAgent(BaseAgent):
    name = "multimodal"

    async def run(self, state: SessionState) -> AgentResult:
        operation = state.metadata.get("operation", "generate")
        output_format = state.metadata.get("output_format", "article")
        base_text = str(state.metadata.get("target_text") or "").strip() or next(
            (result.output for result in reversed(state.agent_results) if result.agent == "editorial"),
            state.input_text,
        )
        if operation == "format":
            return await self._format_social_output(state, base_text, str(output_format))

        if not state.metadata.get("images"):
            return AgentResult(
                agent=self.name,
                output=(
                    "Sin imagenes adjuntas. Sugerencia base: definir una visual editorial "
                    "sobria, contextual y no estereotipada si el articulo lo requiere."
                ),
                metadata={"role": "visual_direction", "mode": "stub"},
            )

        system = (
            "Eres el Director de Arte y Diversidad Visual de NOVA. Propones tratamiento "
            "visual periodístico evitando estereotipos, exotización y representaciones "
            "genéricas de Latinoamérica. No generas imágenes; produces prompts y criterios."
        )
        user = (
            f"Artículo base:\n{base_text}\n\n"
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

    async def _format_social_output(self, state: SessionState, base_text: str, output_format: str) -> AgentResult:
        format_guides = {
            "article": "Devuelve una versión editorial pulida para artículo web, con título, entradilla y cuerpo.",
            "twitter_thread": "Convierte el texto en un hilo de X/Twitter de 6 a 8 publicaciones numeradas, con un cierre que invite a leer o verificar.",
            "instagram_post": "Convierte el texto en un post de Instagram con gancho, cuerpo breve, cierre y 5 hashtags sobrios.",
            "instagram_carousel": "Convierte el texto en un carrusel textual de Instagram de 6 a 8 placas, cada una con título corto y texto escaneable.",
            "linkedin_post": "Convierte el texto en un post de LinkedIn con tono profesional, contexto, aprendizaje y pregunta final.",
            "caption": "Convierte el texto en un caption breve, claro y publicable, con una llamada a la acción.",
        }
        guide = format_guides.get(output_format, format_guides["article"])
        web_sources = "\n".join(f"- {hit.title}: {hit.url}" for hit in state.web_hits[:5])
        system = (
            "Eres la adaptadora editorial de NOVA para formatos sociales. Mantienes precisión, "
            "matiz y cuidado ético. No agregas datos nuevos; si usas contexto web, lo marcas como referencia."
        )
        user = (
            f"Texto base a transformar:\n{base_text}\n\n"
            f"Formato solicitado: {output_format}\n"
            f"Regla de formato: {guide}\n\n"
            f"Referencias web disponibles, si aplican:\n{web_sources or 'Sin referencias web.'}\n\n"
            "Entrega solo la adaptación final."
        )
        output, tokens, error = await self.ask_llm(system, user, temperature=0.35)
        return AgentResult(
            agent=self.name,
            output=output or f"No se pudo generar la adaptación para {output_format}.",
            tokens_used=tokens,
            error=error,
            metadata={"role": "social_format", "mode": "llm", "output_format": output_format},
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
