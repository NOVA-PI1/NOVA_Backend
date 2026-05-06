from agents.base import BaseAgent
from schemas import AgentResult, SessionState


class MultimodalAgent(BaseAgent):
    name = "multimodal"

    async def run(self, state: SessionState) -> AgentResult:
        images = state.metadata.get("images", [])
        if not images:
            return AgentResult(
                agent=self.name,
                output="No se recibieron imagenes para procesar en esta sesion.",
                metadata={"mode": "stub", "image_count": 0},
            )

        return AgentResult(
            agent=self.name,
            output="Interfaz multimodal preparada; configure un modelo con vision para analisis de imagenes.",
            artifacts=[{"type": "image_reference", "value": image} for image in images],
            metadata={"mode": "stub", "image_count": len(images)},
        )
