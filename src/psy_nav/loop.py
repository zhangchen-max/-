"""
会话主循环。负责：
- 学号采集
- 每轮调用 process_turn
- 结束时生成报告并持久化
"""
from __future__ import annotations

import re
import uuid

from .agents import process_turn, run_report
from .db import close_session, create_session, init_db, load_patient, upsert_patient
from .models import LLMClient
from .schema import SessionState


async def chat_loop() -> None:
    init_db()
    llm = LLMClient()

    print("\n" + "=" * 60)
    print("  心理健康初诊信息采集系统 V2")
    print("  输入 exit 结束对话")
    print("=" * 60)

    state = SessionState()

    # ── 学号采集 ─────────────────────────────────────────────────────────
    print("\n系统> 你好，欢迎使用。在开始之前，请告诉我你的学号。")
    while not state.student_id:
        raw = input("\n你> ").strip()
        if raw.lower() in ("exit", "quit", "退出"):
            return
        if re.match(r"^\d{6,15}$", raw):
            state.student_id = raw
        else:
            print("系统> 请输入6-15位数字学号。")

    # ── 加载历史 ─────────────────────────────────────────────────────────
    db_data = load_patient(state.student_id)
    state.is_returning = not db_data["is_new"]
    if not db_data["is_new"] and db_data.get("verbal_style"):
        state.verbal_style = db_data["verbal_style"]

    upsert_patient(state.student_id)
    state.session_id = create_session(state.student_id)

    if state.is_returning:
        opening = "欢迎回来。今天想聊什么，或者有什么新的变化？"
    else:
        opening = "好的，学号已记录。你可以从任何让你困扰的事情开始说，不用担心，这里是安全的。"

    print(f"\n系统> {opening}")

    # ── 主对话循环 ───────────────────────────────────────────────────────
    while not state.session_ended:
        raw = input("\n你> ").strip()
        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "退出"):
            print("\n系统> 好的，感谢你今天的分享。如果有需要随时可以回来。")
            break

        state.turn_count += 1
        response, ended = await process_turn(state, raw, llm)
        print(f"\n系统> {response}")

        if ended:
            break

    # ── 生成报告 ─────────────────────────────────────────────────────────
    print("\n[正在生成医生报告...]")
    report = await run_report(state, llm)
    state.session_report = report

    # 持久化
    upsert_patient(state.student_id, state.verbal_style if state.verbal_style != "unknown" else None)
    close_session(
        session_id=state.session_id,
        turn_count=state.turn_count,
        risk_level=state.risk_level,
        broad_category=state.broad_category,
        differential=[
            {
                "disorder": h.disorder,
                "disorder_cn": h.disorder_cn,
                "probability": h.probability,
            }
            for h in state.differential
        ],
        facts={k: {"value": v.value, "confidence": v.confidence} for k, v in state.facts.items()},
        report=report,
    )

    print("\n" + "=" * 60)
    print("  医生报告（仅供接诊医生参考）")
    print("=" * 60)
    import json
    print(json.dumps(report, ensure_ascii=False, indent=2))
