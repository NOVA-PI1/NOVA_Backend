from __future__ import annotations

import json
from pathlib import Path
from config import Settings
from schemas import KnowledgeHit


class KnowledgeBaseService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._conn = None

    # ------------------------------------------------------------------
    # Conexión lazy a Supabase/PostgreSQL
    # ------------------------------------------------------------------
    def _get_conn(self):
        if self._conn is not None:
            return self._conn
        try:
            import psycopg2
            import psycopg2.extras

            self._conn = psycopg2.connect(
                self.settings.database_url,
                cursor_factory=psycopg2.extras.RealDictCursor,
            )
            self._conn.autocommit = True
            return self._conn
        except Exception:
            self._conn = False
            return None

    # ------------------------------------------------------------------
    # Búsqueda semántica — equivalente exacto al collection.query() de Chroma
    # ------------------------------------------------------------------
    def search(self, query: str, n_results: int = 3) -> list[KnowledgeHit]:
        conn = self._get_conn()
        if not conn:
            return []

        try:
            embedding = self._embed(query)
            if embedding is None:
                return []

            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    id,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (json.dumps(embedding), json.dumps(embedding), n_results),
            )
            rows = cur.fetchall()
        except Exception:
            return []

        hits: list[KnowledgeHit] = []
        for row in rows:
            # pgvector devuelve similitud (1 = igual), Chroma devolvía distancia (0 = igual)
            # Convertimos similitud → distancia para mantener tu threshold igual
            distance = 1 - float(row["similarity"])

            if distance > self.settings.bcl_relevance_threshold:
                continue

            metadata = row["metadata"] or {}
            hits.append(
                KnowledgeHit(
                    text=row["content"],
                    score=distance,
                    source=str(metadata.get("source", "bcl")),
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Indexar documentos nuevos (equivalente al collection.add() de Chroma)
    # ------------------------------------------------------------------
    def add_documents(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> bool:
        conn = self._get_conn()
        if not conn:
            return False

        metadatas = metadatas or [{} for _ in documents]

        try:
            cur = conn.cursor()
            for doc_id, content, metadata in zip(ids, documents, metadatas):
                embedding = self._embed(content)
                if embedding is None:
                    continue
                cur.execute(
                    """
                    INSERT INTO knowledge_chunks (id, content, embedding, metadata)
                    VALUES (%s, %s, %s::vector, %s)
                    ON CONFLICT (id) DO UPDATE
                        SET content   = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata  = EXCLUDED.metadata
                    """,
                    (doc_id, content, json.dumps(embedding), json.dumps(metadata)),
                )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Genera el embedding con OpenAI — mismo modelo que usaba Chroma internamente
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> list[float] | None:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.settings.openai_api_key)
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
            )
            return response.data[0].embedding
        except Exception:
            return None
"""
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
"""