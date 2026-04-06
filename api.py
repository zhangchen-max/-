"""
HTTP API — 供测试系统调用。
默认端口 8000。

原生接口：
  POST   /sessions                    创建会话
  POST   /sessions/{session_id}/turns 发送一轮患者输入
  POST   /sessions/{session_id}/end   结束会话并生成报告
  GET    /sessions/{session_id}/state 查看当前状态摘要

OpenAI 兼容接口（供测试框架 / benchmark 调用）：
  GET    /v1/models
  POST   /v1/chat/completions

  benchmark 模式：自动检测 prompt 中含 allowed_slots / must_ask_slots → 输出纯 JSON
  普通对话模式：按 psy-nav 会话逻辑走，自然语言回复
"""
from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.psy_nav.agents import process_turn, run_report
from src.psy_nav.db import close_session, create_session, init_db, save_message, upsert_patient
from src.psy_nav.models import LLMClient
from src.psy_nav.schema import SessionState

# ── 全局状态 ─────────────────────────────────────────────────────────────────
_sessions: dict[str, SessionState] = {}   # session_id → state
_user_session: dict[str, str] = {}        # student_id → session_id  (普通对话)
_llm: LLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm
    init_db()
    _llm = LLMClient()
    yield


app = FastAPI(title="PsyNav V2 API", version="0.2.0", lifespan=lifespan)


# ── Benchmark Prompts ─────────────────────────────────────────────────────────

_BENCHMARK_SELECTOR_SYSTEM = """
你是精神科诊断访谈优化模块。接收 benchmark 格式的结构化 prompt，只输出一个合法 JSON，不能有任何其他文字、标点、代码块标记。

【选 slot 规则（按优先级）】
1. 从 must_ask_slots 中选一个尚未出现在"已问slot"里的 slot
2. 若 must_ask_slots 全部已问，从 allowed_slots 中选信息增益最大的未问 slot
3. 若满足以下全部条件则输出 FINAL：
   - must_ask_slots 全部已问完
   - 已问 slot 数量 >= 6
4. slot_id 必须严格来自 allowed_slots 列表，不能造新词

【utterance 要求】
- 中文，自然，有共情，一句话，不超过40字
- 只针对该 slot 问一个问题

【严格输出规则】
- 只输出一行 JSON，不加 ```，不加注释，不加换行前缀
- 必须是合法 JSON，可直接 json.loads() 解析

输出示例（二选一）：
{"action":"ASK","slot_id":"duration","utterance":"这些症状大概持续多久了？"}
{"action":"FINAL"}
""".strip()

_BENCHMARK_DIAGNOSIS_SYSTEM = """
你是精神科诊断报告生成模块。根据提供的访谈观察和会话状态，输出符合 benchmark 格式的诊断 JSON。
只输出 JSON，不能有任何其他文字、注释、代码块标记。

【diagnosis label 完整列表（只能用这里的值）】
OCD, BDD, OCPD, OS_OCRD,
Schizophrenia, Schizoaffective_Disorder, Brief_Psychotic_Disorder,
Bipolar_Disorder, MDD, Dysthymia, PMDD,
GAD, Panic_Disorder, Social_Anxiety, PTSD,
BPD, IED, PGD,
SSD, ADJ, ADHD, ASD, OTHER

【各诊断关键特征（辅助判断）】
- OCD：反复闯入性思维 + 强迫行为，自我不和谐，耗时>1h/天
- BDD：对外貌缺陷先占观念（实际不存在或轻微），反复检查/回避
- OCPD：完美主义/控制欲/固执，自我和谐，影响人际，非先占思维
- OS_OCRD：其他强迫相关障碍（囤积障碍/拔毛癖/抠皮症等）
- BPD：情感不稳定、身份认同混乱、慢性空虚感、冲动、自伤、恐惧被遗弃、不稳定关系
- IED：反复爆发性攻击，与诱因不成比例，事后懊悔
- PGD：失去重要他人后持续>12个月强烈哀伤，功能严重受损
- PMDD：月经前1-2周严重情绪症状（易怒/抑郁/焦虑），月经后缓解
- SSD：持续躯体症状伴过度担忧/就医行为，无法以躯体疾病完全解释
- ADJ：可识别应激源后3个月内出现症状，不符合其他具体诊断标准

【输出规则】
- diagnosis_topk 必须包含 3-5 个候选，概率总和不超过 1.5（允许重叠）
- 第2-5名要有合理概率（>=0.05），不能全部堆在第1名
- justification.support_slots 只能包含 asked_slots 里有的 slot
- 输出一个合法 JSON 对象，可直接 json.loads() 解析

【输出格式】
{"diagnosis_topk":[{"label":"OCD","confidence":0.65},{"label":"BDD","confidence":0.20},{"label":"OCPD","confidence":0.10}],"risk_flags":{"suicide":false,"violence":false,"psychosis":false,"substance":false},"justification":[{"claim":"反复强迫思维和仪式化行为","support_slots":["intrusive_thoughts","compulsive_behavior"]}],"next_steps":[{"type":"suggest_assessment","text":"建议Y-BOCS量表评估。"}],"calibration":{"uncertainty_statement":"当前结论基于有限访谈信息，需进一步评估。"}}
""".strip()


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    student_id: str

class StartResponse(BaseModel):
    session_id: str
    opening_message: str

class TurnRequest(BaseModel):
    content: str

class TurnResponse(BaseModel):
    response: str
    ended: bool
    state: dict[str, Any]

class EndResponse(BaseModel):
    report: dict[str, Any]
    state: dict[str, Any]

class _OAIMessage(BaseModel):
    role: str
    content: str

class _OAIChatRequest(BaseModel):
    model: str = "psy-nav"
    messages: List[_OAIMessage]
    user: Optional[str] = "anonymous"
    stream: Optional[bool] = False


# ── 原生路由 ──────────────────────────────────────────────────────────────────

@app.post("/sessions", response_model=StartResponse, summary="创建新会话")
async def create_new_session(req: StartRequest):
    upsert_patient(req.student_id)
    state = SessionState(student_id=req.student_id)
    state.session_id = create_session(req.student_id)
    _sessions[state.session_id] = state
    opening = "你好，你可以从任何让你困扰的事情开始说，不用担心，这里是安全的。"
    return StartResponse(session_id=state.session_id, opening_message=opening)


@app.post("/sessions/{session_id}/turns", response_model=TurnResponse, summary="发送患者输入")
async def send_turn(session_id: str, req: TurnRequest):
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")
    if state.session_ended:
        raise HTTPException(status_code=400, detail="session already ended")

    state.turn_count += 1
    response, ended = await process_turn(state, req.content, _llm)
    save_message(session_id, state.student_id, state.turn_count, "user", req.content)
    save_message(session_id, state.student_id, state.turn_count, "assistant", response)

    return TurnResponse(response=response, ended=ended, state=_state_summary(state))


@app.post("/sessions/{session_id}/end", response_model=EndResponse, summary="结束会话，生成报告")
async def end_session(session_id: str):
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")

    report = await run_report(state, _llm)
    state.session_report = report
    state.session_ended = True

    close_session(
        session_id=session_id,
        turn_count=state.turn_count,
        risk_level=state.risk_level,
        broad_category=state.broad_category,
        differential=[
            {"disorder": h.disorder, "disorder_cn": h.disorder_cn, "probability": h.probability}
            for h in state.differential
        ],
        facts={k: {"value": v.value, "confidence": v.confidence} for k, v in state.facts.items()},
        report=report,
    )
    _sessions.pop(session_id, None)
    return EndResponse(report=report, state=_state_summary(state))


@app.get("/sessions/{session_id}/state", summary="查看当前状态")
async def get_state(session_id: str):
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")
    return _state_summary(state)


# ── OpenAI 兼容路由 ───────────────────────────────────────────────────────────

@app.get("/v1/models", summary="OpenAI 兼容：模型列表")
async def oai_models():
    return {
        "object": "list",
        "data": [{"id": "psy-nav", "object": "model", "created": 0, "owned_by": "psy-nav"}],
    }


@app.post("/v1/chat/completions", summary="OpenAI 兼容：对话接口")
async def oai_chat(req: _OAIChatRequest):
    student_id = req.user or "anonymous"
    last_user = _last_user_msg(req.messages)

    # ── 路由判断 ──────────────────────────────────────────────────────────────
    if _is_benchmark_ask_prompt(last_user):
        content = await _benchmark_ask(last_user)

    elif _is_benchmark_diagnosis_prompt(last_user):
        content = await _benchmark_diagnosis(last_user, student_id)

    else:
        # 普通 psy-nav 会话
        content = await _normal_turn(req, student_id)

    return _oai_response(content)


# ── Benchmark 处理 ────────────────────────────────────────────────────────────

def _is_benchmark_ask_prompt(text: str) -> bool:
    return "allowed_slots" in text and "must_ask_slots" in text

def _is_benchmark_diagnosis_prompt(text: str) -> bool:
    return "diagnosis_topk" in text or (
        "FINAL" in text and ("诊断" in text or "diagnosis" in text.lower())
    )

def _extract_json(raw) -> str:
    """
    健壮地从 LLM 输出中提取 JSON 字符串。
    LLMClient 已经尝试过 json.loads；这里处理它失败后返回原始字符串的情况。
    """
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    text = str(raw).strip()
    # 去掉 markdown 代码块
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # 尝试直接解析
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    # 找第一个完整 JSON 对象
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    # 兜底：返回安全的 ASK 或错误标记
    return text


async def _benchmark_ask(user_prompt: str) -> str:
    """选 slot 或输出 FINAL，返回纯 JSON 字符串。"""
    result = await _llm.chat(
        system=_BENCHMARK_SELECTOR_SYSTEM,
        user=user_prompt,
        max_tokens=200,
        temperature=0.1,
    )
    return _extract_json(result)


async def _benchmark_diagnosis(user_prompt: str, student_id: str) -> str:
    """生成诊断 JSON。优先用 psy-nav 的状态，没有就靠 LLM 从 prompt 推断。"""
    session_id = _user_session.get(student_id)
    state = _sessions.get(session_id) if session_id else None

    extra_context = ""
    if state:
        extra_context = json.dumps({
            "psynav_differential": [
                {"disorder": h.disorder, "disorder_cn": h.disorder_cn, "probability": h.probability}
                for h in state.differential
            ],
            "risk_level": state.risk_level,
            "risk_factors": state.risk_factors,
            "insight_status": state.insight_status,
            "broad_category": state.broad_category,
        }, ensure_ascii=False)

    payload = f"{user_prompt}\n\n[psy-nav内部状态参考]\n{extra_context}" if extra_context else user_prompt

    result = await _llm.chat(
        system=_BENCHMARK_DIAGNOSIS_SYSTEM,
        user=payload,
        max_tokens=800,
        temperature=0.1,
    )
    return _extract_json(result)


# ── 普通 psy-nav 会话 ─────────────────────────────────────────────────────────

async def _normal_turn(req: _OAIChatRequest, student_id: str) -> str:
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        return "你好，请告诉我你的困扰。"

    system_msgs = [m for m in req.messages if m.role == "system"]
    force_end = any("end_session" in m.content for m in system_msgs)

    session_id = _user_session.get(student_id)
    state = _sessions.get(session_id) if session_id else None

    if state is None or state.session_ended:
        upsert_patient(student_id)
        state = SessionState(student_id=student_id)
        state.session_id = create_session(student_id)
        _sessions[state.session_id] = state
        _user_session[student_id] = state.session_id

    if force_end:
        return await _finalize_session(state, student_id)

    patient_input = user_msgs[-1].content
    state.turn_count += 1
    response, ended = await process_turn(state, patient_input, _llm)
    save_message(state.session_id, student_id, state.turn_count, "user", patient_input)
    save_message(state.session_id, student_id, state.turn_count, "assistant", response)

    if ended:
        await _finalize_session(state, student_id)

    return response


async def _finalize_session(state: SessionState, student_id: str) -> str:
    report = await run_report(state, _llm)
    state.session_report = report
    state.session_ended = True
    close_session(
        session_id=state.session_id,
        turn_count=state.turn_count,
        risk_level=state.risk_level,
        broad_category=state.broad_category,
        differential=[
            {"disorder": h.disorder, "disorder_cn": h.disorder_cn, "probability": h.probability}
            for h in state.differential
        ],
        facts={k: {"value": v.value, "confidence": v.confidence} for k, v in state.facts.items()},
        report=report,
    )
    _user_session.pop(student_id, None)
    return f"[会话已结束]\n{report.get('recommended_diagnosis', '待评估')}"


# ── 辅助 ──────────────────────────────────────────────────────────────────────

def _last_user_msg(messages: list) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""

def _oai_response(content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "psy-nav",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

def _state_summary(state: SessionState) -> dict:
    top = state.top_hypothesis()
    return {
        "session_id": state.session_id,
        "student_id": state.student_id,
        "turn_count": state.turn_count,
        "broad_category": state.broad_category,
        "probe_target": state.probe_target,
        "alliance_score": state.alliance_score,
        "risk_level": state.risk_level,
        "insight_status": state.insight_status,
        "session_ended": state.session_ended,
        "top_hypothesis": {
            "disorder_cn": top.disorder_cn,
            "probability": top.probability,
        } if top else None,
        "differential": [
            {
                "disorder": h.disorder,
                "disorder_cn": h.disorder_cn,
                "probability": h.probability,
                "critical_unknowns": h.critical_unknowns[:3],
            }
            for h in state.differential
        ],
    }
