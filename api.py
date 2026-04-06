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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.psy_nav.agents import process_turn, run_report
from src.psy_nav.db import close_session, create_session, init_db, save_message, upsert_patient
from src.psy_nav.models import LLMClient
from src.psy_nav.schema import SessionState

# ── 全局状态 ─────────────────────────────────────────────────────────────────
_sessions: dict[str, SessionState] = {}       # session_id → state
_user_session: dict[str, str] = {}            # student_id → session_id  (普通对话)
_bench_awaiting_diag: set[str] = set()        # student_id：已发 FINAL，等诊断请求
_llm: LLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm
    init_db()
    _llm = LLMClient()
    yield


app = FastAPI(title="PsyNav V2 API", version="0.2.0", lifespan=lifespan)


# ── Benchmark Prompts ─────────────────────────────────────────────────────────
# benchmark 只有 11 个诊断标签，与 psy-nav 内部标签体系完全不同，不要混用

_PIB_LABELS = "ADJ, PGD, OCD, BDD, OS_OCRD, SSD, PMDD, IED, BPD, OCPD, OTHER"

_PIB_CLINICAL_NOTES = """
【各诊断关键特征】
OCD：闯入性思维+强迫行为，自我不和谐，耗时>1h/天
BDD：对外貌缺陷先占观念（客观上不存在或轻微），反复检查/回避
OCPD：完美主义/控制欲/固执，自我和谐，人格模式（无闯入性思维）
OS_OCRD：囤积障碍/拔毛癖/抠皮症等其他强迫相关障碍
BPD：情感不稳、身份混乱、慢性空虚、冲动自伤、恐惧被遗弃
IED：反复爆发性攻击，与诱因不成比例，发作间无攻击性
PGD：丧亲后>12个月持续强烈哀伤+身份感丧失，与MDD区别是核心是哀伤非快感缺乏
PMDD：严格月经黄体期情绪症状，月经后消失，有明确周期性
SSD：持续躯体症状+过度健康焦虑/就医，核心在躯体（非泛化焦虑）
ADJ：明确应激源后3月内起病，不满足其他具体障碍标准时优先考虑
OTHER：以上均不符合时使用
""".strip()

_BENCHMARK_SELECTOR_SYSTEM = f"""
你是精神科诊断访谈模块。根据 benchmark 结构化 prompt 选择下一个动作，只输出一个合法 JSON。

【选 slot 规则（按优先级）】
1. 从 must_ask_slots 中选一个未在"已问slot"里的 slot
2. must_ask_slots 全部已问后，从 allowed_slots 选信息增益最大的未问 slot
3. 同时满足以下条件则输出 FINAL：must_ask_slots 全部已问 且 已问数量 >= 6
4. slot_id 必须来自 allowed_slots，不能造新词

【utterance 要求】中文，自然，一句话，不超过40字

【严格输出规则】只输出一行 JSON，不加任何其他文字或代码块标记

示例：
{{"action":"ASK","slot_id":"duration","utterance":"这些症状大概持续多久了？"}}
{{"action":"FINAL"}}
""".strip()

_BENCHMARK_FINAL_SYSTEM = f"""
你是精神科诊断报告生成模块。根据访谈观察输出 final_output JSON。只输出 JSON，不加任何其他文字。

【diagnosis label 只能用这11个】
{_PIB_LABELS}

{_PIB_CLINICAL_NOTES}

【输出规则】
- diagnosis_topk：3-5个候选，必须包含 OTHER，confidence 之和不超过 1.5
- slot_predictions：直接用 observations 字段的内容
- risk_flags：基于观察到的症状填写 suicide/violence/psychosis/substance
- justification.support_slots：只能用 asked_slots 里出现过的 slot_id，不得凭空填写
- next_steps：只建议评估/转介，严禁写具体药物/剂量
- calibration.uncertainty_statement：说明结论基于有限信息

【输出格式】
{{"diagnosis_topk":[{{"label":"OCD","confidence":0.65}},{{"label":"BDD","confidence":0.20}},{{"label":"OTHER","confidence":0.08}}],"slot_predictions":{{}},"risk_flags":{{"suicide":false,"violence":false,"psychosis":false,"substance":false}},"justification":[{{"claim":"观察到强迫思维和仪式化行为","support_slots":["intrusive_thoughts","ritual_behavior_present"]}}],"next_steps":[{{"type":"suggest_assessment","text":"建议Y-BOCS量表评估。"}}],"calibration":{{"uncertainty_statement":"当前结论基于有限访谈信息，需进一步评估。"}}}}
""".strip()

_BENCHMARK_SLOT_FILLING_SYSTEM = """
你是精神科访谈数据抽取模块。从患者文本中抽取结构化 slot 值。只输出 JSON，不加任何其他文字。

【输出格式】{"slot_predictions": {"slot_id": value, ...}}
【规则】
- 只填写能从文本明确推断的 slot，不要猜测
- boolean slot 用 true/false
- duration 格式用 "X months" / "X weeks" / "X years"
- ordinal/categorical 用题目给出的选项值
- 不确定的不填，宁缺毋滥
""".strip()

_BENCHMARK_STATIC_DDX_SYSTEM = f"""
你是精神科鉴别诊断模块。根据 slot_predictions 推断最可能的诊断。只输出 JSON，不加任何其他文字。

【diagnosis label 只能用这11个】
{_PIB_LABELS}

{_PIB_CLINICAL_NOTES}

【输出格式】{{"diagnosis_topk":[{{"label":"OCD","confidence":0.65}},{{"label":"BDD","confidence":0.20}},{{"label":"OTHER","confidence":0.08}}]}}
【规则】3-5个候选，必须包含 OTHER，confidence 之和不超过1.5
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

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def web_ui():
    return HTMLResponse(content=_WEB_UI_HTML)


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
    task = _detect_pib_task(req.messages, last_user)

    if task == "none":
        content = await _normal_turn(req, student_id)
    elif task == "slot_filling":
        content = await _pib_slot_filling(last_user)
    elif task == "static_ddx":
        content = await _pib_static_ddx(last_user)
    elif task == "dynamic_action":
        content = await _pib_dynamic_action(last_user, student_id)
    else:  # dynamic_final
        content = await _pib_dynamic_final(last_user)

    return _oai_response(content)


# ── Benchmark 检测 & 处理 ─────────────────────────────────────────────────────

def _detect_pib_task(messages: list, user_text: str) -> str:
    """
    返回：slot_filling | static_ddx | dynamic_action | dynamic_final | none
    判断依据：system message 特征 + user message 关键词
    """
    # system message：benchmark 两种写法（有无空格）
    is_bench = any(
        m.role == "system" and (
            "只能输出JSON" in m.content or
            "只能输出 JSON" in m.content or
            "只输出JSON" in m.content or
            "只输出 JSON" in m.content
        )
        for m in messages
    )
    if not is_bench:
        return "none"

    # 按关键词区分任务类型
    if "允许的 slot_id 列表" in user_text or "从输入文本抽取结构化 slot" in user_text:
        return "slot_filling"
    if "允许的标签" in user_text or "基于已抽取 slots 给出鉴别诊断" in user_text:
        return "static_ddx"
    if "must_ask_slots" in user_text and "allowed_slots" in user_text:
        return "dynamic_action"
    # 动态 FINAL 的第二次调用：含 asked_slots + observations，但无 allowed_slots
    if "asked_slots" in user_text or "observations" in user_text:
        return "dynamic_final"
    # 兜底：benchmark session 但无法识别具体任务 → 当作 dynamic_final
    return "dynamic_final"


def _extract_json(raw) -> str:
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(0))
            return match.group(0)
        except json.JSONDecodeError:
            pass
    return text


async def _pib_slot_filling(user_prompt: str) -> str:
    """Static Stage 1：从文本抽取 slot 值。"""
    result = await _llm.chat(
        system=_BENCHMARK_SLOT_FILLING_SYSTEM,
        user=user_prompt,
        max_tokens=1200,
        temperature=0.0,
    )
    return _extract_json(result)


async def _pib_static_ddx(user_prompt: str) -> str:
    """Static Stage 2：基于 slot 做鉴别诊断。"""
    result = await _llm.chat(
        system=_BENCHMARK_STATIC_DDX_SYSTEM,
        user=user_prompt,
        max_tokens=400,
        temperature=0.0,
    )
    return _extract_json(result)


async def _pib_dynamic_action(user_prompt: str, student_id: str) -> str:
    """Dynamic：选 ASK slot 或决定 FINAL。"""
    result = await _llm.chat(
        system=_BENCHMARK_SELECTOR_SYSTEM,
        user=user_prompt,
        max_tokens=200,
        temperature=0.0,
    )
    return _extract_json(result)


async def _pib_dynamic_final(user_prompt: str) -> str:
    """Dynamic：输出 final_output JSON（diagnosis + slot_predictions + ...）。"""
    result = await _llm.chat(
        system=_BENCHMARK_FINAL_SYSTEM,
        user=user_prompt,
        max_tokens=800,
        temperature=0.0,
    )
    out = _extract_json(result)
    # 后处理：确保 OTHER 在 diagnosis_topk 里，support_slots 只含合法 slot
    try:
        obj = json.loads(out)
        obj = _sanitize_final_output(obj, user_prompt)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return out


def _sanitize_final_output(obj: dict, prompt: str) -> dict:
    """确保 final_output 通过 benchmark 的 _validate_final_output 检查。"""
    # 1. 提取 asked_slots（从 prompt 文本中解析）
    asked_slots: set[str] = set()
    m = re.search(r"asked_slots[:\s]+(\[.*?\])", prompt, re.DOTALL)
    if m:
        try:
            asked_slots = set(json.loads(m.group(1)))
        except Exception:
            pass

    # 2. 确保 OTHER 在 diagnosis_topk
    topk = obj.get("diagnosis_topk", [])
    labels = [item.get("label") if isinstance(item, dict) else item for item in topk]
    if "OTHER" not in labels:
        topk.append({"label": "OTHER", "confidence": 0.05})
        obj["diagnosis_topk"] = topk

    # 3. label 只保留 PIB 合法标签
    pib_valid = set(_PIB_LABELS.replace(" ", "").split(","))
    obj["diagnosis_topk"] = [
        item for item in obj.get("diagnosis_topk", [])
        if isinstance(item, dict) and item.get("label") in pib_valid
    ]
    if not obj["diagnosis_topk"]:
        obj["diagnosis_topk"] = [{"label": "OTHER", "confidence": 1.0}]

    # 4. justification support_slots 只保留 asked_slots 里的
    if asked_slots:
        for jitem in obj.get("justification", []):
            if isinstance(jitem, dict):
                jitem["support_slots"] = [
                    s for s in (jitem.get("support_slots") or [])
                    if s in asked_slots
                ]

    # 5. 确保必要字段存在
    obj.setdefault("slot_predictions", {})
    obj.setdefault("risk_flags", {"suicide": False, "violence": False, "psychosis": False, "substance": False})
    obj.setdefault("justification", [{"claim": "基于访谈观察", "support_slots": []}])
    obj.setdefault("next_steps", [{"type": "suggest_assessment", "text": "建议进一步评估。"}])
    obj.setdefault("calibration", {"uncertainty_statement": "当前结论基于有限访谈信息，需进一步评估。"})

    return obj


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


# ── Web UI ────────────────────────────────────────────────────────────────────

_WEB_UI_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PsyNav — 心理健康初诊采集</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #f5f5f7; color: #1d1d1f; height: 100vh; display: flex; }

  /* 左侧对话区 */
  #chat-panel {
    flex: 1; display: flex; flex-direction: column; max-width: 700px;
    margin: 0 auto; padding: 20px;
  }
  h1 { font-size: 18px; font-weight: 600; margin-bottom: 16px; color: #333; }

  #messages {
    flex: 1; overflow-y: auto; display: flex; flex-direction: column;
    gap: 12px; padding: 16px; background: #fff; border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }
  .msg { max-width: 80%; padding: 10px 14px; border-radius: 16px;
         line-height: 1.6; font-size: 15px; white-space: pre-wrap; }
  .msg.system { align-self: center; background: #e9e9eb; color: #555;
                font-size: 13px; border-radius: 8px; }
  .msg.assistant { align-self: flex-start; background: #e9f0ff; color: #1a1a2e; }
  .msg.user     { align-self: flex-end;   background: #007aff; color: #fff; }

  #input-row {
    display: flex; gap: 10px; margin-top: 14px;
  }
  #user-input {
    flex: 1; padding: 12px 16px; border: 1.5px solid #d1d1d6;
    border-radius: 24px; font-size: 15px; outline: none;
    transition: border-color 0.2s;
  }
  #user-input:focus { border-color: #007aff; }
  button {
    padding: 12px 20px; border: none; border-radius: 24px;
    font-size: 14px; cursor: pointer; font-weight: 500;
    transition: opacity 0.15s;
  }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  #send-btn { background: #007aff; color: #fff; }
  #end-btn  { background: #ff3b30; color: #fff; }

  /* 右侧状态面板 */
  #state-panel {
    width: 280px; padding: 20px; display: flex; flex-direction: column; gap: 14px;
    overflow-y: auto;
  }
  .card {
    background: #fff; border-radius: 10px; padding: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
  }
  .card h3 { font-size: 12px; font-weight: 600; color: #8e8e93;
             text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 13px; font-weight: 500;
  }
  .badge.low      { background: #d1fae5; color: #065f46; }
  .badge.medium   { background: #fef3c7; color: #92400e; }
  .badge.high,
  .badge.critical { background: #fee2e2; color: #991b1b; }
  .badge.default  { background: #e5e7eb; color: #374151; }

  .hypothesis-row {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 6px;
  }
  .hypothesis-name { font-size: 14px; }
  .prob-bar-wrap { width: 80px; height: 6px; background: #e5e7eb; border-radius: 3px; }
  .prob-bar { height: 100%; border-radius: 3px; background: #007aff; }
  .prob-val { font-size: 12px; color: #6b7280; min-width: 34px; text-align: right; }

  #report-area { display: none; }
  #report-area pre {
    background: #1e1e2e; color: #cdd6f4; padding: 14px;
    border-radius: 8px; font-size: 12px; overflow-x: auto;
    white-space: pre-wrap; word-break: break-all;
    max-height: 60vh; overflow-y: auto;
  }

  /* 学号输入覆层 */
  #login-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.3);
    display: flex; align-items: center; justify-content: center; z-index: 100;
  }
  #login-box {
    background: #fff; padding: 32px; border-radius: 16px;
    width: 320px; box-shadow: 0 8px 32px rgba(0,0,0,0.15);
  }
  #login-box h2 { font-size: 20px; margin-bottom: 20px; }
  #student-id-input {
    width: 100%; padding: 12px; border: 1.5px solid #d1d1d6;
    border-radius: 10px; font-size: 15px; margin-bottom: 14px; outline: none;
  }
  #start-btn { width: 100%; background: #007aff; color: #fff; padding: 13px; font-size: 15px; }
  #login-error { color: #ff3b30; font-size: 13px; margin-top: 8px; }
</style>
</head>
<body>

<div id="login-overlay">
  <div id="login-box">
    <h2>心理健康初诊采集</h2>
    <input id="student-id-input" type="text" placeholder="请输入学号（6-15位数字）" maxlength="15">
    <button id="start-btn" onclick="startSession()">开始</button>
    <div id="login-error"></div>
  </div>
</div>

<div id="chat-panel">
  <h1>PsyNav · 初诊信息采集</h1>
  <div id="messages"></div>
  <div id="input-row">
    <input id="user-input" type="text" placeholder="输入你想说的..." disabled
           onkeydown="if(event.key==='Enter' && !event.shiftKey){event.preventDefault();sendMsg();}">
    <button id="send-btn" onclick="sendMsg()" disabled>发送</button>
    <button id="end-btn" onclick="endSession()" disabled>结束</button>
  </div>
</div>

<div id="state-panel">
  <div class="card">
    <h3>当前状态</h3>
    <div style="font-size:13px; line-height:2;">
      <div>谱系：<span id="st-category" class="badge default">—</span></div>
      <div>轮次：<span id="st-turns">—</span></div>
      <div>风险：<span id="st-risk" class="badge low">low</span></div>
      <div>联盟：<span id="st-alliance">—</span></div>
      <div>探查：<span id="st-probe" style="color:#007aff;font-size:13px;">—</span></div>
    </div>
  </div>

  <div class="card">
    <h3>鉴别诊断</h3>
    <div id="st-differential">—</div>
  </div>

  <div class="card" id="report-area">
    <h3>医生报告</h3>
    <pre id="report-content"></pre>
  </div>
</div>

<script>
let sessionId = null;
let ended = false;

async function startSession() {
  const sid = document.getElementById('student-id-input').value.trim();
  if (!/^\\d{6,15}$/.test(sid)) {
    document.getElementById('login-error').textContent = '请输入6-15位数字学号';
    return;
  }
  const res = await fetch('/sessions', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({student_id: sid})
  });
  const data = await res.json();
  sessionId = data.session_id;
  document.getElementById('login-overlay').style.display = 'none';
  document.getElementById('user-input').disabled = false;
  document.getElementById('send-btn').disabled = false;
  document.getElementById('end-btn').disabled = false;
  addMsg('assistant', data.opening_message);
  document.getElementById('user-input').focus();
}

async function sendMsg() {
  if (!sessionId || ended) return;
  const input = document.getElementById('user-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg('user', text);
  setLoading(true);

  const res = await fetch('/sessions/' + sessionId + '/turns', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({content: text})
  });
  const data = await res.json();
  setLoading(false);
  addMsg('assistant', data.response);
  updateState(data.state);

  if (data.ended) {
    ended = true;
    addMsg('system', '会话已结束，正在生成报告...');
    await fetchReport();
  }
}

async function endSession() {
  if (!sessionId || ended) return;
  ended = true;
  setLoading(true);
  addMsg('system', '正在生成报告...');
  await fetchReport();
}

async function fetchReport() {
  setLoading(true);
  const res = await fetch('/sessions/' + sessionId + '/end', {method:'POST'});
  const data = await res.json();
  setLoading(false);
  document.getElementById('user-input').disabled = true;
  document.getElementById('send-btn').disabled = true;
  document.getElementById('end-btn').disabled = true;
  showReport(data.report);
  if (data.state) updateState(data.state);
}

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  const box = document.getElementById('messages');
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function setLoading(on) {
  document.getElementById('send-btn').disabled = on;
  document.getElementById('user-input').disabled = on;
}

function updateState(s) {
  if (!s) return;
  document.getElementById('st-turns').textContent = s.turn_count || '—';
  document.getElementById('st-probe').textContent = s.probe_target || '—';

  const cat = document.getElementById('st-category');
  cat.textContent = s.broad_category || '—';
  cat.className = 'badge default';

  const riskEl = document.getElementById('st-risk');
  riskEl.textContent = s.risk_level || 'low';
  riskEl.className = 'badge ' + (s.risk_level || 'low');

  const al = s.alliance_score !== undefined ? (s.alliance_score * 100).toFixed(0) + '%' : '—';
  document.getElementById('st-alliance').textContent = al;

  const diffEl = document.getElementById('st-differential');
  if (s.differential && s.differential.length) {
    diffEl.innerHTML = s.differential.map(h => `
      <div class="hypothesis-row">
        <span class="hypothesis-name">${h.disorder_cn}</span>
        <div class="prob-bar-wrap"><div class="prob-bar" style="width:${(h.probability*100).toFixed(0)}%"></div></div>
        <span class="prob-val">${(h.probability*100).toFixed(0)}%</span>
      </div>
    `).join('');
  } else {
    diffEl.textContent = '等待分诊...';
  }
}

function showReport(report) {
  const area = document.getElementById('report-area');
  area.style.display = 'block';
  document.getElementById('report-content').textContent = JSON.stringify(report, null, 2);
}

document.getElementById('student-id-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') startSession();
});
</script>
</body>
</html>"""
