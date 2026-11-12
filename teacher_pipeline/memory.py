"""SQLite long-term memory for the persona agent.

Stores explicit user profile, short-term messages, episodic summaries, and
session summaries. No sensitive/private content should be written by callers.
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3
from typing import Any


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profile (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    topics TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, session_id, key)
);
CREATE INDEX IF NOT EXISTS idx_messages_user_session
    ON messages(user_id, session_id, id);
CREATE INDEX IF NOT EXISTS idx_episodes_user
    ON episodes(user_id, id);
CREATE INDEX IF NOT EXISTS idx_summaries_user
    ON summaries(user_id, id);
"""


class Memory:
    def __init__(self, path: str = "persona_memory.db") -> None:
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # PM④ 并发铺路（T4 per-user Agent 多连接并存的前提）：WAL 让读不阻塞写、写不阻塞读；
        # busy_timeout=5000 让短暂锁竞争排队而不是立刻抛 database is locked。单连接场景无副作用。
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def ensure_user(self, user_id: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?)",
            (user_id, now()),
        )
        self.conn.commit()

    def add_message(self, user_id: str, session_id: str, role: str, content: str) -> None:
        self.ensure_user(user_id)
        self.conn.execute(
            "INSERT INTO messages(user_id, session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, session_id, role, content, now()),
        )
        self.conn.commit()

    def recent_messages(self, user_id: str, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE user_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
            (user_id, session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def set_profile(self, user_id: str, key: str, value: str) -> None:
        self.ensure_user(user_id)
        self.conn.execute(
            "INSERT INTO profile(user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (user_id, key, value, now()),
        )
        self.conn.commit()

    def get_profile(self, user_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value FROM profile WHERE user_id=? ORDER BY key",
            (user_id,),
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def add_episode(self, user_id: str, session_id: str, summary: str, topics: str = "") -> None:
        self.ensure_user(user_id)
        self.conn.execute(
            "INSERT INTO episodes(user_id, session_id, summary, topics, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, session_id, summary, topics, now()),
        )
        self.conn.commit()

    def recent_episodes(self, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT session_id, summary, topics, created_at FROM episodes "
            "WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def set_summary(self, user_id: str, session_id: str, summary: str) -> None:
        self.ensure_user(user_id)
        self.conn.execute(
            "INSERT INTO summaries(user_id, session_id, summary, created_at) VALUES (?, ?, ?, ?)",
            (user_id, session_id, summary, now()),
        )
        self.conn.commit()

    def last_summary(self, user_id: str) -> str:
        row = self.conn.execute(
            "SELECT summary FROM summaries WHERE user_id=? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return row["summary"] if row else ""

    def set_state(self, user_id: str, session_id: str, key: str, value: str) -> None:
        self.ensure_user(user_id)
        self.conn.execute(
            "INSERT INTO state(user_id, session_id, key, value, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, session_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (user_id, session_id, key, value, now()),
        )
        self.conn.commit()

    def get_state(self, user_id: str, session_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value FROM state WHERE user_id=? AND session_id=?",
            (user_id, session_id),
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def forget(self, user_id: str) -> None:
        for table in ("messages", "profile", "episodes", "summaries", "state", "users"):
            self.conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
        self.conn.commit()

    def snapshot(self, user_id: str) -> dict[str, Any]:
        return {
            "profile": self.get_profile(user_id),
            "last_summary": self.last_summary(user_id),
            "episodes": self.recent_episodes(user_id),
            "messages": self.conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE user_id=?", (user_id,)
            ).fetchone()["n"],
        }

    def export_json(self, user_id: str) -> str:
        return json.dumps(self.snapshot(user_id), ensure_ascii=False, indent=2)
