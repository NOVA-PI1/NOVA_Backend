import sqlite3
from threading import Lock

from db.models import agent_result_from_row, connect_sqlite, event_to_record, state_to_record
from schemas import AgentResult, BusEvent, KnowledgeHit, SessionState, SessionSummary


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

    def list_sessions(self, user_id: str | None = None, limit: int = 50) -> list[SessionSummary]:
        limit = max(1, min(limit, 200))
        query = "SELECT * FROM sessions"
        params: list[object] = []
        if user_id:
            query += " WHERE json_extract(perfil, '$.user_id') = ?"
            params.append(user_id)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self.connection.execute(query, params).fetchall()

        return [summary_from_record(dict(row)) for row in rows]


class PostgresSessionStore:
    def __init__(self, database_url: str) -> None:
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as error:
            raise RuntimeError(
                "PostgreSQL requires psycopg2-binary. Install backend dependencies with "
                "`pip install -r requirements.txt`."
            ) from error

        self.connection = psycopg2.connect(database_url)
        self.cursor_factory = psycopg2.extras.RealDictCursor
        self._lock = Lock()
        self.initialize()

    def initialize(self) -> None:
        with self._lock:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        input_text TEXT NOT NULL,
                        perfil JSONB NOT NULL,
                        metadata JSONB NOT NULL,
                        knowledge_hits JSONB NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id BIGSERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS agent_results (
                        id BIGSERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        agent TEXT NOT NULL,
                        output TEXT NOT NULL,
                        warnings JSONB NOT NULL,
                        questions JSONB NOT NULL,
                        artifacts JSONB NOT NULL,
                        tokens_used INTEGER NOT NULL,
                        metadata JSONB NOT NULL,
                        error TEXT,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS bus_events (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        payload JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL
                    );
                    """
                )
            self.connection.commit()

    def save_session(self, state: SessionState) -> None:
        import psycopg2.extras

        record = state_to_record(state)
        with self._lock:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions (
                        session_id, input_text, perfil, metadata, knowledge_hits, status, created_at, updated_at
                    ) VALUES (
                        %(session_id)s, %(input_text)s, %(perfil)s, %(metadata)s, %(knowledge_hits)s,
                        %(status)s, %(created_at)s, %(updated_at)s
                    )
                    ON CONFLICT(session_id) DO UPDATE SET
                        input_text=EXCLUDED.input_text,
                        perfil=EXCLUDED.perfil,
                        metadata=EXCLUDED.metadata,
                        knowledge_hits=EXCLUDED.knowledge_hits,
                        status=EXCLUDED.status,
                        updated_at=EXCLUDED.updated_at
                    """,
                    json_record(record, psycopg2.extras.Json),
                )
            self.connection.commit()

    def save_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                    (session_id, role, content),
                )
            self.connection.commit()

    def save_agent_result(self, session_id: str, result: AgentResult) -> None:
        import psycopg2.extras

        with self._lock:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO agent_results (
                        session_id, agent, output, warnings, questions, artifacts, tokens_used, metadata, error
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        result.agent,
                        result.output,
                        psycopg2.extras.Json(result.warnings),
                        psycopg2.extras.Json(result.questions),
                        psycopg2.extras.Json(result.artifacts),
                        result.tokens_used,
                        psycopg2.extras.Json(result.metadata),
                        result.error,
                    ),
                )
            self.connection.commit()

    def save_event(self, event: BusEvent) -> None:
        import psycopg2.extras

        record = event_to_record(event)
        with self._lock:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO bus_events (id, session_id, type, payload, created_at)
                    VALUES (%(id)s, %(session_id)s, %(type)s, %(payload)s, %(created_at)s)
                    ON CONFLICT(id) DO UPDATE SET
                        session_id=EXCLUDED.session_id,
                        type=EXCLUDED.type,
                        payload=EXCLUDED.payload,
                        created_at=EXCLUDED.created_at
                    """,
                    json_record(record, psycopg2.extras.Json),
                )
            self.connection.commit()

    def get_session(self, session_id: str) -> SessionState | None:
        with self._lock:
            with self.connection.cursor(cursor_factory=self.cursor_factory) as cursor:
                cursor.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
                session = cursor.fetchone()
                if session is None:
                    return None
                cursor.execute(
                    "SELECT * FROM agent_results WHERE session_id = %s ORDER BY id ASC",
                    (session_id,),
                )
                results = cursor.fetchall()

        return SessionState(
            session_id=session["session_id"],
            input_text=session["input_text"],
            perfil=session["perfil"] or {},
            metadata=session["metadata"] or {},
            knowledge_hits=[KnowledgeHit(**hit) for hit in (session["knowledge_hits"] or [])],
            agent_results=[agent_result_from_mapping(row) for row in results],
            status=session["status"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
        )

    def events_for_session(self, session_id: str) -> list[BusEvent]:
        with self._lock:
            with self.connection.cursor(cursor_factory=self.cursor_factory) as cursor:
                cursor.execute(
                    "SELECT * FROM bus_events WHERE session_id = %s ORDER BY created_at ASC",
                    (session_id,),
                )
                rows = cursor.fetchall()
        return [
            BusEvent(
                id=row["id"],
                session_id=row["session_id"],
                type=row["type"],
                payload=row["payload"] or {},
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def list_sessions(self, user_id: str | None = None, limit: int = 50) -> list[SessionSummary]:
        limit = max(1, min(limit, 200))
        params: list[object] = []
        query = "SELECT * FROM sessions"
        if user_id:
            query += " WHERE perfil->>'user_id' = %s"
            params.append(user_id)
        query += " ORDER BY updated_at DESC LIMIT %s"
        params.append(limit)

        with self._lock:
            with self.connection.cursor(cursor_factory=self.cursor_factory) as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()

        return [summary_from_record(row) for row in rows]


def create_session_store(database_url: str):
    if database_url.startswith("postgresql://") or database_url.startswith("postgres://"):
        return PostgresSessionStore(database_url)
    return SQLiteSessionStore(database_url)


def json_dump(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


def json_load(value: str, default: object) -> object:
    import json

    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def json_record(record: dict, json_wrapper) -> dict:
    wrapped = dict(record)
    for key in ("perfil", "metadata", "knowledge_hits", "payload"):
        if key in wrapped and isinstance(wrapped[key], str):
            wrapped[key] = json_wrapper(json_load(wrapped[key], {} if key != "knowledge_hits" else []))
    return wrapped


def agent_result_from_mapping(row: dict) -> AgentResult:
    return AgentResult(
        agent=row["agent"],
        output=row["output"],
        warnings=row["warnings"] or [],
        questions=row["questions"] or [],
        artifacts=row["artifacts"] or [],
        tokens_used=row["tokens_used"] or 0,
        metadata=row["metadata"] or {},
        error=row["error"],
    )


def summary_from_record(row: dict) -> SessionSummary:
    perfil = row.get("perfil")
    if isinstance(perfil, str):
        perfil = json_load(perfil, {})
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        metadata = json_load(metadata, {})

    title = (metadata or {}).get("title") or row.get("input_text") or "Investigacion sin titulo"
    title = " ".join(str(title).split())
    if len(title) > 80:
        title = title[:79].rsplit(" ", 1)[0] + "..."

    return SessionSummary(
        session_id=row["session_id"],
        title=title,
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        user_id=(perfil or {}).get("user_id"),
    )
