import json
import sqlite3
from pathlib import Path
from typing import Any

from schemas import AgentResult, BusEvent, SessionState


def sqlite_path_from_url(database_url: str) -> str:
    if database_url.startswith("sqlite:///"):
        return database_url.removeprefix("sqlite:///")
    if database_url.startswith("sqlite://"):
        return database_url.removeprefix("sqlite://")
    raise ValueError("Only sqlite database URLs are supported in v1")


def connect_sqlite(database_url: str) -> sqlite3.Connection:
    path = sqlite_path_from_url(database_url)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def agent_result_from_row(row: sqlite3.Row) -> AgentResult:
    return AgentResult(
        agent=row["agent"],
        output=row["output"],
        warnings=json.loads(row["warnings"] or "[]"),
        questions=json.loads(row["questions"] or "[]"),
        artifacts=json.loads(row["artifacts"] or "[]"),
        tokens_used=row["tokens_used"] or 0,
        metadata=json.loads(row["metadata"] or "{}"),
        error=row["error"],
    )


def state_to_record(state: SessionState) -> dict[str, Any]:
    return {
        "session_id": state.session_id,
        "input_text": state.input_text,
        "perfil": dumps(state.perfil),
        "metadata": dumps(state.metadata),
        "knowledge_hits": dumps([hit.model_dump(mode="json") for hit in state.knowledge_hits]),
        "status": state.status,
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
    }


def event_to_record(event: BusEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "session_id": event.session_id,
        "type": event.type,
        "payload": dumps(event.payload),
        "created_at": event.created_at.isoformat(),
    }
