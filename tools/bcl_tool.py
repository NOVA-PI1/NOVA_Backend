# pyrefly: ignore [missing-import]
from crewai.tools import BaseTool
from pydantic import Field
from bcl.loader import KnowledgeBaseService

class BCLTool(BaseTool):
    name: str = "bcl_tool"
    description: str = (
        "Útil para buscar información veraz, datos regionales y contexto histórico en la "
        "Biblioteca de Consulta Local (BCL). Úsala cuando necesites hechos contrastados "
        "antes de redactar o corregir una noticia."
    )
    kb_service: KnowledgeBaseService = Field(exclude=True)

    def _run(self, query: str) -> str:
        """
        Consulta la base de datos vectorial ChromaDB y devuelve los fragmentos más relevantes.
        """
        hits = self.kb_service.search(query)
        if not hits:
            return "No se encontró información relevante en la BCL local para esta consulta."
        
        context = []
        for hit in hits:
            context.append(f"[Fuente: {hit.source}]\n{hit.text}")
        
        return "\n\n---\n\n".join(context)
