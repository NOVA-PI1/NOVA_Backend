from typing import TYPE_CHECKING, Any

from agents.base import BaseAgent
from schemas import AgentResult, SessionState

if TYPE_CHECKING:
    from tools.bcl_tool import BCLTool


class EditorialAgent(BaseAgent):
    name = "editorial"

    async def run(self, state: SessionState) -> AgentResult:
        sources = "\n\n".join(
            f"- Fuente: {hit.source} | score={hit.score}\n{hit.text}"
            for hit in state.knowledge_hits[:5]
        ) or "No hay resultados BCL relevantes para este tema."
        system = (
            "Eres el Editor Jefe Periodístico de NOVA. Redactas artículos claros, "
            "verificables y útiles para una audiencia latinoamericana. Evita inventar "
            "datos: si una afirmación no está sustentada, márcala como pendiente de "
            "verificación. Entrega un artículo con título, entradilla, cuerpo y cierre."
        )
        user = (
            f"Tema o encargo del usuario:\n{state.input_text}\n\n"
            f"Contexto recuperado de la BCL:\n{sources}\n\n"
            "Escribe un primer artículo periodístico en español. No incluyas notas sobre "
            "tu proceso; entrega solamente el contenido editorial."
        )
        output, tokens, error = await self.ask_llm(system, user, temperature=0.35)
        fallback = (
            "Titulo: Borrador pendiente de modelo\n\n"
            f"Entradilla: {state.input_text}\n\n"
            "Cuerpo: No se pudo contactar el modelo configurado. Este borrador sirve "
            "como salida estable para que el flujo NOVA continue y el error sea visible.\n\n"
            "Cierre: Reintenta con un proveedor LLM disponible antes de publicar."
        )
        return AgentResult(
            agent=self.name,
            output=output or fallback,
            tokens_used=tokens,
            error=error,
            metadata={"role": "editorial"},
        )

def create_editorial_agent(llm_instance, bcl_tool: "BCLTool") -> Any:
    """
    Crea el Agente Editorial Jefe para NOVA v2.0 utilizando CrewAI.
    """
    from crewai import Agent

    return Agent(
        role="Editor Jefe Periodístico del Sur Global",
        goal=(
            "Investigar y redactar narrativas periodísticas veraces consultando "
            "obligatoriamente la Biblioteca de Consulta Local (BCL) para anclar "
            "la información en contextos regionales."
        ),
        backstory=(
            "Eres un editor experto con años de experiencia en periodismo latinoamericano. "
            "Rechazas las narrativas colonialistas y priorizas la transparencia y la soberanía informativa. "
            "NUNCA inventas datos; siempre utilizas tu herramienta BCLTool para buscar hechos "
            "reales antes de redactar. Tu estilo es claro, directo y con una ética impecable."
        ),
        tools=[bcl_tool],
        llm=llm_instance,
        verbose=True,  # Habilita la traza para la visibilidad de "Caja Blanca"
        allow_delegation=False,
        memory=True
    )
