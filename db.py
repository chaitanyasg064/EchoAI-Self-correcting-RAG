"""
db.py

SQLite-backed persistence layer for ChatGPT-style conversation history.
Uses only the standard library so no new dependency is required. Refreshing
the page never loses prior chats because all conversations/messages live
here, not in browser memory.
"""

import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, List, Optional, TypedDict

logger = logging.getLogger("echoai.db")

DB_PATH = "echoai.db"

_local = threading.local()


class Message(TypedDict):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str


class Conversation(TypedDict):
    id: str
    title: str
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_connection() -> sqlite3.Connection:
    """Thread-local connection; SQLite connections aren't safe to share across threads."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        _local.conn = conn
    return conn


@contextmanager
def _cursor() -> Iterator[sqlite3.Cursor]:
    conn = _get_connection()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def init_db() -> None:
    try:
        with _cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
        logger.info("SQLite database initialized at '%s'.", DB_PATH)
    except Exception:
        logger.exception("Failed to initialize SQLite database.")
        raise


def create_conversation(title: str = "New Chat") -> Conversation:
    conv_id = str(uuid.uuid4())
    now = _now()
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO conversations (id, title, summary, created_at, updated_at) VALUES (?, ?, '', ?, ?)",
            (conv_id, title, now, now),
        )
    return {"id": conv_id, "title": title, "created_at": now, "updated_at": now}


def list_conversations() -> List[Conversation]:
    with _cursor() as cur:
        cur.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> Optional[Conversation]:
    with _cursor() as cur:
        cur.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def rename_conversation(conversation_id: str, title: str) -> bool:
    with _cursor() as cur:
        cur.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), conversation_id),
        )
        return cur.rowcount > 0


def delete_conversation(conversation_id: str) -> bool:
    with _cursor() as cur:
        cur.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cur.rowcount > 0


def add_message(conversation_id: str, role: str, content: str) -> Message:
    msg_id = str(uuid.uuid4())
    now = _now()
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (msg_id, conversation_id, role, content, now),
        )
        cur.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    return {"id": msg_id, "conversation_id": conversation_id, "role": role, "content": content, "created_at": now}


def get_messages(conversation_id: str) -> List[Message]:
    with _cursor() as cur:
        cur.execute(
            "SELECT id, conversation_id, role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def search_conversations(query: str) -> List[Conversation]:
    like = f"%{query}%"
    with _cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.title LIKE ? OR m.content LIKE ?
            ORDER BY c.updated_at DESC
            """,
            (like, like),
        )
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def generate_title_from_text(text: str, max_len: int = 48) -> str:
    """Zero-latency title generation from the first user message (no extra LLM call)."""
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_len:
        return cleaned or "New Chat"
    return cleaned[: max_len - 1].rstrip() + "…"