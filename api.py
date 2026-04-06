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

【各诊断关键特征与鉴别要点】
- OCD：反复闯入性思维 + 强迫行为，自我不和谐，耗时>1h/天，与 OCPD 区别：OCPD 是人格模式非先占思维
- BDD：对外貌缺陷先占观念（客观不存在或轻微），反复镜子检查/回避/寻求安慰
- OCPD：完美主义/控制欲/固执/过度工作，自我和谐，影响人际，无强迫思维
- OS_OCRD：囤积障碍、拔毛癖、抠皮症、疾病焦虑障碍
- BPD：情感高度不稳定、身份认同混乱、慢性空虚感、冲动自伤、极度恐惧被遗弃、理想化/贬低交替
- IED：反复爆发性语言或肢体攻击，与诱因完全不成比例，两次发作间无攻击性，事后懊悔
- PGD（延长哀伤）：失去重要他人后 >12个月，持续强烈思念/悲痛，身份认同感丧失，无法接受死亡现实，功能显著受损；与 MDD 区别：核心是哀伤而非普遍性快感缺乏
- PMDD：严格月经周期相关——黄体期（排卵后至月经前1周）出现严重情绪症状，月经来潮后症状消失；与 MDD 区别：有明确周期性
- SSD（躯体症状障碍）：一个或多个持续躯体症状 + 过度健康焦虑/就医行为/症状相关思维耗时 >1h/天；无法以躯体疾病完全解释；与 GAD 区别：焦虑核心是躯体而非泛化担忧
- ADJ（适应障碍）：有明确可识别应激源 + 3个月内出现情绪/行为症状 + 症状严重程度超过正常应激反应 + 不符合其他具体诊断（OCD/MDD/BPD等）；当有应激源但症状不够重也不够特异时优先考虑

【三类易漏诊的特别提示】
- 如果有明确应激事件（失业/分手/失去亲人/搬迁等）且症状时间线与之吻合，且不满足其他具体障碍标准 → ADJ
- 如果有丧亲/失去重要关系且哀伤持续超过一年 → PGD（不要默认用 MDD）
- 如果以躯体不适为核心主诉、反复就医但检查无异常、对症状有过度担忧 → SSD（不要默认用 GAD）

【输出规则】
- diagnosis_topk 必须包含 3-5 个候选，所有 confidence 之和不超过 1.5
- 第2-5名要有合理概率（>=0.05），体现鉴别思路
- justification.support_slots 只能包含 asked_slots 里已有的 slot，不得凭空填写
- 输出一个合法 JSON 对象，可直接 json.loads() 解析

【输出格式示例】
{"diagnosis_topk":[{"label":"OCD","confidence":0.65},{"label":"BDD","confidence":0.20},{"label":"OCPD","confidence":0.10}],"risk_flags":{"suicide":false,"violence":false,"psychosis":false,"substance":false},"justification":[{"claim":"反复强迫思维和仪式化行为耗时超过1小时","support_slots":["intrusive_thoughts","compulsive_behavior"]}],"next_steps":[{"type":"suggest_assessment","text":"建议Y-BOCS量表评估。"}],"calibration":{"uncertainty_statement":"当前结论基于有限访谈信息，需进一步评估。"}}
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

    # ── 路由判断 ──────────────────────────────────────────────────────────────
    if _is_benchmark_session(req.messages):
        # benchmark 模式：system 含"只能输出JSON"
        if student_id in _bench_awaiting_diag:
            # 上一轮输出了 FINAL，本轮是诊断请求
            _bench_awaiting_diag.discard(student_id)
            content = await _benchmark_diagnosis(last_user, student_id)
        elif _is_benchmark_ask_prompt(last_user):
            result_str = await _benchmark_ask(last_user)
            # 如果模型决定 FINAL，记录状态等下一轮诊断
            try:
                parsed = json.loads(result_str)
                if parsed.get("action") == "FINAL":
                    _bench_awaiting_diag.add(student_id)
            except (json.JSONDecodeError, AttributeError):
                pass
            content = result_str
        else:
            # 兜底：直接走诊断
            content = await _benchmark_diagnosis(last_user, student_id)
    else:
        # 普通 psy-nav 会话
        content = await _normal_turn(req, student_id)

    return _oai_response(content)


# ── Benchmark 处理 ────────────────────────────────────────────────────────────

def _is_benchmark_session(messages: list) -> bool:
    """Benchmark 的 system prompt 含 '只能输出JSON'，以此作为主判断信号。"""
    for m in messages:
        if m.role == "system" and ("只能输出JSON" in m.content or "只输出JSON" in m.content):
            return True
    return False

def _is_benchmark_ask_prompt(text: str) -> bool:
    return "allowed_slots" in text and "must_ask_slots" in text

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
