"""
持久化层，使用 MySQL。
表结构：patients, session_records, messages（每轮对话）
"""
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import pymysql
import pymysql.cursors


def _cfg() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
    }


@contextmanager
def _conn(db: str | None = None):
    cfg = _cfg()
    if db:
        cfg["database"] = db
    conn = pymysql.connect(**cfg)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    db_name = os.getenv("DB_NAME", "psy_nav")

    # 建库
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )

    # 建表
    with _conn(db_name) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                student_id    VARCHAR(30)  PRIMARY KEY,
                first_visit   DATETIME     NOT NULL,
                last_visit    DATETIME     NOT NULL,
                verbal_style  VARCHAR(20),
                total_sessions INT         DEFAULT 0
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS session_records (
                session_id        VARCHAR(36)  PRIMARY KEY,
                student_id        VARCHAR(30)  NOT NULL,
                start_time        DATETIME     NOT NULL,
                end_time          DATETIME,
                turn_count        INT          DEFAULT 0,
                risk_level        VARCHAR(20),
                broad_category    VARCHAR(50),
                differential_json JSON,
                facts_json        JSON,
                report_json       JSON,
                INDEX idx_student (student_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # 每一轮对话完整存储
            cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id            BIGINT       AUTO_INCREMENT PRIMARY KEY,
                session_id    VARCHAR(36)  NOT NULL,
                student_id    VARCHAR(30)  NOT NULL,
                turn          INT          NOT NULL,
                role          VARCHAR(10)  NOT NULL COMMENT 'user | assistant',
                content       TEXT         NOT NULL,
                created_at    DATETIME     NOT NULL,
                INDEX idx_session (session_id),
                INDEX idx_student_msg (student_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)


def _db() -> str:
    return os.getenv("DB_NAME", "psy_nav")


def load_patient(student_id: str) -> dict[str, Any]:
    with _conn(_db()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM patients WHERE student_id=%s", (student_id,))
            row = cur.fetchone()
    if row is None:
        return {"is_new": True, "verbal_style": None}
    return {"is_new": False, "verbal_style": row["verbal_style"]}


def upsert_patient(student_id: str, verbal_style: str | None = None) -> None:
    now = datetime.utcnow()
    with _conn(_db()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT student_id FROM patients WHERE student_id=%s", (student_id,))
            exists = cur.fetchone()
            if not exists:
                cur.execute(
                    "INSERT INTO patients (student_id, first_visit, last_visit, verbal_style, total_sessions) "
                    "VALUES (%s, %s, %s, %s, 1)",
                    (student_id, now, now, verbal_style),
                )
            else:
                if verbal_style:
                    cur.execute(
                        "UPDATE patients SET last_visit=%s, verbal_style=%s, total_sessions=total_sessions+1 "
                        "WHERE student_id=%s",
                        (now, verbal_style, student_id),
                    )
                else:
                    cur.execute(
                        "UPDATE patients SET last_visit=%s, total_sessions=total_sessions+1 "
                        "WHERE student_id=%s",
                        (now, student_id),
                    )


def create_session(student_id: str) -> str:
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    with _conn(_db()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO session_records (session_id, student_id, start_time) VALUES (%s, %s, %s)",
                (session_id, student_id, now),
            )
    return session_id


def save_message(session_id: str, student_id: str, turn: int, role: str, content: str) -> None:
    """每轮对话实时写入，不等会话结束。"""
    now = datetime.utcnow()
    with _conn(_db()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (session_id, student_id, turn, role, content, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (session_id, student_id, turn, role, content, now),
            )


def close_session(
    session_id: str,
    turn_count: int,
    risk_level: str,
    broad_category: str,
    differential: list,
    facts: dict,
    report: dict,
) -> None:
    now = datetime.utcnow()
    with _conn(_db()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE session_records SET
                   end_time=%s, turn_count=%s, risk_level=%s, broad_category=%s,
                   differential_json=%s, facts_json=%s, report_json=%s
                   WHERE session_id=%s""",
                (
                    now, turn_count, risk_level, broad_category,
                    json.dumps(differential, ensure_ascii=False),
                    json.dumps(facts, ensure_ascii=False),
                    json.dumps(report, ensure_ascii=False),
                    session_id,
                ),
            )
