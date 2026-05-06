import sqlite3
from threading import Lock

from db.models import agent_result_from_row, connect_sqlite, event_to_record, state_to_record
from schemas import AgentResult, BusEvent, KnowledgeHit, SessionState


class SQLiteSessionStore:
    def __init__(self, database_url: str) -> None:
        self.connection = connect_sqlite(database_url)
        self._lock = Lock()
        self.initialize()

    def initialize(self) -> None:
        with self._lock:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    input_text TEXT NOT NULL,
                    perfil TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    knowledge_hits TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS agent_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    output TEXT NOT NULL,
                    warnings TEXT NOT NULL,
                    questions TEXT NOT NULL,
                    artifacts TEXT NOT NULL,
                    tokens_used INTEGER NOT NULL,
                    metadata TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS bus_events (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self.connection.commit()

    def save_session(self, state: SessionState) -> None:
        record = state_to_record(state)
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO sessions (
                    session_id, input_text, perfil, metadata, knowledge_hits, status, created_at, updated_at
                ) VALUES (
                    :session_id, :input_text, :perfil, :metadata, :knowledge_hits, :status, :created_at, :updated_at
                )
                ON CONFLICT(session_id) DO UPDATE SET
                    input_text=excluded.input_text,
                    perfil=excluded.perfil,
                    metadata=excluded.metadata,
                    knowledge_hits=excluded.knowledge_hits,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                record,
            )
            self.connection.commit()

    def save_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            self.connection.commit()

    def save_agent_result(self, session_id: str, result: AgentResult) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO agent_results (
                    session_id, agent, output, warnings, questions, artifacts, tokens_used, metadata, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    result.agent,
                    result.output,
                    json_dump(result.warnings),
                    json_dump(result.questions),
                    json_dump(result.artifacts),
                    result.tokens_used,
                    json_dump(result.metadata),
                    result.error,
                ),
            )
            self.connection.commit()

    def save_event(self, event: BusEvent) -> None:
        record = event_to_record(event)
        with self._lock:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO bus_events (id, session_id, type, payload, created_at)
                VALUES (:id, :session_id, :type, :payload, :created_at)
                """,
                record,
            )
            self.connection.commit()

    def get_session(self, session_id: str) -> SessionState | None:
        with self._lock:
            session = self.connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                return None

            results = self.connection.execute(
                "SELECT * FROM agent_results WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

        return SessionState(
            session_id=session["session_id"],
            input_text=session["input_text"],
            perfil=json_load(session["perfil"], {}),
            metadata=json_load(session["metadata"], {}),
            knowledge_hits=[KnowledgeHit(**hit) for hit in json_load(session["knowledge_hits"], [])],
            agent_results=[agent_result_from_row(row) for row in results],
            status=session["status"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
        )

    def events_for_session(self, session_id: str) -> list[BusEvent]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM bus_events WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [
            BusEvent(
                id=row["id"],
                session_id=row["session_id"],
                type=row["type"],
                payload=json_load(row["payload"], {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]


def json_dump(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


def json_load(value: str, default: object) -> object:
    import json

    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default
