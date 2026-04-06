"""
持久化层，使用 SQLite（无需额外服务，开箱即用）。
表结构：patients, session_records
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any

DB_PATH = os.getenv("DB_PATH", "psy_nav.db")


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            student_id   TEXT PRIMARY KEY,
            first_visit  TEXT NOT NULL,
            last_visit   TEXT NOT NULL,
            verbal_style TEXT,
            total_sessions INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS session_records (
            session_id    TEXT PRIMARY KEY,
            student_id    TEXT NOT NULL,
            start_time    TEXT NOT NULL,
            end_time      TEXT,
            turn_count    INTEGER DEFAULT 0,
            risk_level    TEXT,
            broad_category TEXT,
            differential_json TEXT,
            facts_json    TEXT,
            report_json   TEXT
        );
        """)


def load_patient(student_id: str) -> dict[str, Any]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE student_id = ?", (student_id,)
        ).fetchone()
        if row is None:
            return {"is_new": True}
        return {"is_new": False, "verbal_style": row["verbal_style"]}


def upsert_patient(student_id: str, verbal_style: str | None = None) -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        existing = conn.execute(
            "SELECT student_id FROM patients WHERE student_id = ?", (student_id,)
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO patients (student_id, first_visit, last_visit, verbal_style, total_sessions) "
                "VALUES (?, ?, ?, ?, 1)",
                (student_id, now, now, verbal_style),
            )
        else:
            conn.execute(
                "UPDATE patients SET last_visit=?, total_sessions=total_sessions+1"
                + (", verbal_style=?" if verbal_style else "") + " WHERE student_id=?",
                (now, verbal_style, student_id) if verbal_style else (now, student_id),
            )


def create_session(student_id: str) -> str:
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO session_records (session_id, student_id, start_time) VALUES (?, ?, ?)",
            (session_id, student_id, now),
        )
    return session_id


def close_session(
    session_id: str,
    turn_count: int,
    risk_level: str,
    broad_category: str,
    differential: list,
    facts: dict,
    report: dict,
) -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            """UPDATE session_records SET
               end_time=?, turn_count=?, risk_level=?, broad_category=?,
               differential_json=?, facts_json=?, report_json=?
               WHERE session_id=?""",
            (
                now, turn_count, risk_level, broad_category,
                json.dumps(differential, ensure_ascii=False),
                json.dumps(facts, ensure_ascii=False),
                json.dumps(report, ensure_ascii=False),
                session_id,
            ),
        )
