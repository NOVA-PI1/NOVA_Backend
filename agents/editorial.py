from typing import TYPE_CHECKING, Any

from agents.base import BaseAgent
from schemas import AgentResult, SessionState

if TYPE_CHECKING:
    from tools.bcl_tool import BCLTool


class EditorialAgent(BaseAgent):
    name = "editorial"

    async def run(self, state: SessionState) -> AgentResult:
        operation = state.metadata.get("operation", "generate")
        target_text = str(state.metadata.get("target_text") or "").strip()
        sources = "\n\n".join(
            f"- Fuente: {hit.source} | score={hit.score}\n{hit.text}"
            for hit in state.knowledge_hits[:5]
        ) or "No hay resultados BCL relevantes para este tema."
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

        if operation in {"question", "format"}:
            return AgentResult(
                agent=self.name,
                output=target_text or state.input_text,
                metadata={"role": "editorial", "mode": "passthrough", "operation": operation},
            )

        system = (
            "Eres la editora periodística de NOVA: una compañera editorial con criterio propio, "
            "cercana y exigente. Escribes y revisas piezas en español con enfoque latinoamericano, "
            "sin inventar datos. Si una afirmación no está sustentada, márcala como pendiente de verificación. "
            "No expliques tu funcionamiento; entrega texto editorial útil y listo para trabajar."
        )
        if operation == "revise" and target_text:
            user = (
                f"Borrador activo a modificar:\n{target_text}\n\n"
                f"Instrucción editorial del periodista:\n{state.input_text}\n\n"
                f"Contexto BCL separado de la web:\n{sources}\n\n"
                f"Contexto web/citas, si aplica:\n{web_sources}\n\n"
                "Reescribe el borrador respetando la intención original, aplicando la instrucción "
                "y conservando lo que funcione. Entrega solo la nueva versión completa."
            )
        else:
            user = (
                f"Tema o encargo del periodista:\n{state.input_text}\n\n"
                f"Contexto recuperado de la BCL:\n{sources}\n\n"
                f"Contexto web/citas, si aplica:\n{web_sources}\n\n"
                "Escribe un artículo completo en español con título, subtítulo, entradilla, "
                "cuerpo desarrollado, contexto/antecedentes y cierre con una pregunta o acción concreta. "
                "Entrega solo el contenido editorial listo para editar."
            )
        output, tokens, error = await self.ask_llm(system, user, temperature=0.4)
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
            metadata={"role": "editorial", "operation": operation},
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
