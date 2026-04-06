"""
五个 Agent 的实现。每个 Agent 只做一件事。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from .models import LLMClient
from .prompts import (
    DIALOGUE_SYSTEM,
    REASONER_SYSTEM,
    REPORT_SYSTEM,
    RISK_SYSTEM,
    TRIAGE_SYSTEM,
)
from .schema import Fact, Hypothesis, SessionState


# ---------------------------------------------------------------------------
# Agent 1: Triage
# ---------------------------------------------------------------------------

async def run_triage(state: SessionState, llm: LLMClient) -> dict[str, Any]:
    """
    前2轮运行一次。
    输入：前几轮对话
    输出：broad_category + initial differential + verbal_style
    """
    history_text = "\n".join(
        f"{'患者' if m['role']=='user' else '系统'}: {m['content']}"
        for m in state.history[-4:]
    )
    user_msg = f"以下是患者入场时说的话：\n{history_text}"

    result = await llm.chat(
        system=TRIAGE_SYSTEM,
        user=user_msg,
        max_tokens=800,
        temperature=0.2,
    )

    return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------------
# Agent 2: Clinical Reasoner
# ---------------------------------------------------------------------------

async def run_reasoner(
    state: SessionState,
    patient_input: str,
    llm: LLMClient,
) -> dict[str, Any]:
    """
    每轮运行。更新鉴别诊断表，确定下一个探查目标。
    """
    payload = json.dumps(
        {
            "patient_input": patient_input,
            "context": state.to_context_dict(),
            "recent_history": state.history[-6:],
        },
        ensure_ascii=False,
    )

    result = await llm.chat(
        system=REASONER_SYSTEM,
        user=payload,
        max_tokens=1200,
        temperature=0.2,
    )
    return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------------
# Agent 3: Dialogue Agent
# ---------------------------------------------------------------------------

async def run_dialogue(
    state: SessionState,
    patient_input: str,
    probe_target: str,
    session_should_end: bool,
    llm: LLMClient,
) -> str:
    """
    每轮运行。把探查目标翻译成一句自然的话。
    """
    payload = json.dumps(
        {
            "patient_latest_input": patient_input,
            "probe_target": probe_target,
            "session_should_end": session_should_end,
            "verbal_style": state.verbal_style,
            "alliance_score": state.alliance_score,
            "last_assistant_output": state.last_assistant_output,
            "recent_history": state.history[-4:],
        },
        ensure_ascii=False,
    )

    result = await llm.chat(
        system=DIALOGUE_SYSTEM,
        user=payload,
        max_tokens=200,
        temperature=0.6,
    )

    if isinstance(result, dict):
        return result.get("response", "能多告诉我一些吗？")
    return "能多告诉我一些吗？"


# ---------------------------------------------------------------------------
# Agent 4: Risk Monitor
# ---------------------------------------------------------------------------

async def run_risk(patient_input: str, llm: LLMClient) -> dict[str, Any]:
    """并行运行，实时危机识别。"""
    result = await llm.chat(
        system=RISK_SYSTEM,
        user=patient_input,
        max_tokens=200,
        temperature=0.1,
    )
    if isinstance(result, dict):
        return result
    return {"risk_level": "low", "is_crisis": False, "risk_factors": []}


# ---------------------------------------------------------------------------
# Agent 5: Report Generator
# ---------------------------------------------------------------------------

async def run_report(state: SessionState, llm: LLMClient) -> dict[str, Any]:
    """会话结束时生成医生报告。"""
    payload = json.dumps(
        {
            "broad_category": state.broad_category,
            "differential": [
                {
                    "disorder": h.disorder,
                    "disorder_cn": h.disorder_cn,
                    "probability": h.probability,
                    "supporting": h.supporting,
                    "against": h.against,
                    "critical_unknowns": h.critical_unknowns,
                }
                for h in state.differential
            ],
            "facts": {
                k: {"value": v.value, "confidence": v.confidence, "source": v.source}
                for k, v in state.facts.items()
            },
            "treatment_history": state.treatment_history,
            "risk_level": state.risk_level,
            "risk_factors": state.risk_factors,
            "insight_status": state.insight_status,
            "verbal_style": state.verbal_style,
            "turn_count": state.turn_count,
            "recent_history": state.history[-8:],
        },
        ensure_ascii=False,
    )

    result = await llm.chat(
        system=REPORT_SYSTEM,
        user=payload,
        max_tokens=1800,
        temperature=0.2,
    )
    return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------------
# 主处理函数：每轮调用
# ---------------------------------------------------------------------------

async def process_turn(
    state: SessionState,
    patient_input: str,
    llm: LLMClient,
) -> tuple[str, bool]:
    """
    每轮的核心逻辑。
    返回 (assistant_response, session_ended)
    """
    # ── 分诊（前2轮完成后执行一次）────────────────────────────────────────
    if not state.triage_done and state.turn_count >= 2:
        triage = await run_triage(state, llm)
        _apply_triage(state, triage)
        state.triage_done = True

    # ── 并行：风险识别 + 推理 ────────────────────────────────────────────
    risk_task = asyncio.create_task(run_risk(patient_input, llm))

    if state.triage_done:
        reasoner_task = asyncio.create_task(run_reasoner(state, patient_input, llm))
        risk_result, reasoner_result = await asyncio.gather(risk_task, reasoner_task)
        _apply_reasoner(state, reasoner_result)
    else:
        risk_result = await risk_task
        reasoner_result = {}

    # ── 风险处理 ─────────────────────────────────────────────────────────
    _apply_risk(state, risk_result)

    if state.crisis_active:
        response = (
            "谢谢你告诉我这些，我很担心你现在的状态。"
            "请马上联系身边可信任的人，或拨打 120 / 心理援助热线 400-161-9995。"
            "你现在身边有人吗？"
        )
        state.session_ended = True
        state.last_assistant_output = response
        _append_history(state, patient_input, response)
        return response, True

    # ── 对话生成 ─────────────────────────────────────────────────────────
    session_should_end = reasoner_result.get("session_should_end", False)
    probe_target = state.probe_target

    response = await run_dialogue(state, patient_input, probe_target, session_should_end, llm)

    # 屏蔽病名（面向患者的输出）
    for name in ["精神分裂症", "双相情感障碍", "抑郁症", "焦虑症", "强迫症", "惊恐障碍"]:
        response = response.replace(name, "你描述的这些情况")

    state.last_assistant_output = response
    _append_history(state, patient_input, response)

    if session_should_end:
        state.session_ended = True

    return response, state.session_ended


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _apply_triage(state: SessionState, triage: dict) -> None:
    state.broad_category = triage.get("broad_category", "unknown")
    state.verbal_style = triage.get("verbal_style", state.verbal_style)
    state.insight_status = triage.get("insight_status", "unknown")

    for item in triage.get("differential", []):
        h = Hypothesis(
            disorder=item.get("disorder", ""),
            disorder_cn=item.get("disorder_cn", ""),
            probability=float(item.get("probability", 0.3)),
            supporting=item.get("supporting", []),
            against=item.get("against", []),
            critical_unknowns=item.get("critical_unknowns", []),
        )
        state.differential.append(h)

    for k, v in triage.get("initial_facts", {}).items():
        state.facts[k] = Fact(value=v, confidence=0.6, first_turn=state.turn_count, source="patient")


def _apply_reasoner(state: SessionState, result: dict) -> None:
    if not result:
        return

    # 更新事实
    for k, v in result.get("new_facts", {}).items():
        state.facts[k] = Fact(
            value=v,
            confidence=0.7,
            first_turn=state.turn_count,
            source="patient",
        )

    # 更新鉴别诊断
    updated = result.get("updated_differential", [])
    if updated:
        state.differential = [
            Hypothesis(
                disorder=item.get("disorder", ""),
                disorder_cn=item.get("disorder_cn", ""),
                probability=float(item.get("probability", 0.3)),
                supporting=item.get("supporting", []),
                against=item.get("against", []),
                critical_unknowns=item.get("critical_unknowns", []),
            )
            for item in updated
        ]

    # 探查目标
    probe = result.get("probe_target", "")
    if probe:
        state.probe_target = probe
        if probe not in state.probed_topics:
            state.probed_topics.append(probe)

    # 关系 & 风格
    state.insight_status = result.get("insight_status", state.insight_status)
    if result.get("verbal_style"):
        state.verbal_style = result["verbal_style"]
    delta = float(result.get("alliance_delta", 0))
    state.alliance_score = max(0.0, min(1.0, state.alliance_score + delta))

    # 治疗史
    th = result.get("treatment_history_update", {})
    if th:
        state.treatment_history.update(th)


def _apply_risk(state: SessionState, result: dict) -> None:
    state.risk_level = result.get("risk_level", "low")
    state.risk_factors = result.get("risk_factors", [])
    state.crisis_active = result.get("is_crisis", False)


def _append_history(state: SessionState, patient_input: str, response: str) -> None:
    state.history.append({"role": "user", "content": patient_input})
    state.history.append({"role": "assistant", "content": response})
    # 保留最近20轮
    if len(state.history) > 40:
        state.history = state.history[-40:]
