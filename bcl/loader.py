from pathlib import Path

from config import Settings
from schemas import KnowledgeHit


class KnowledgeBaseService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._collection = None

    def _load_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb

            Path(self.settings.chroma_persist_path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self.settings.chroma_persist_path)
            self._collection = client.get_or_create_collection(self.settings.bcl_collection_name)
            return self._collection
        except Exception:
            self._collection = False
            return None

    def search(self, query: str, n_results: int = 3) -> list[KnowledgeHit]:
        collection = self._load_collection()
        if not collection:
            return []
        try:
            result = collection.query(query_texts=[query], n_results=n_results)
        except Exception:
            return []

        documents = result.get("documents", [[]])[0] if result else []
        distances = result.get("distances", [[]])[0] if result else []
        metadatas = result.get("metadatas", [[]])[0] if result else []
        hits: list[KnowledgeHit] = []

        for index, document in enumerate(documents):
            distance = distances[index] if index < len(distances) else None
            if distance is not None and distance > self.settings.bcl_relevance_threshold:
                continue
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            hits.append(
                KnowledgeHit(
                    text=document,
                    score=distance,
                    source=str(metadata.get("source", "bcl")),
                )
            )
        return hits
