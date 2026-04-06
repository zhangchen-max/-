"""
核心数据结构。
V2 设计核心：用鉴别诊断表（Differential）驱动对话，而非症状清单。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# 宽泛诊断大类（快速分诊阶段的输出）
# ---------------------------------------------------------------------------

BroadCategory = Literal[
    "psychotic_spectrum",       # 精神病性谱系（幻觉/妄想/思维形式障碍）
    "mood_spectrum",            # 情感障碍谱系（抑郁/躁狂）
    "anxiety_spectrum",         # 焦虑谱系（GAD/惊恐/OCD/PTSD）
    "neurodevelopmental",       # 神经发育（ADHD/ASD）
    "organic",                  # 器质性（脑器质性/物质所致/躯体疾病）
    "unknown",
]


# ---------------------------------------------------------------------------
# 鉴别诊断条目
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    disorder: str               # 英文 key，如 "schizophrenia"
    disorder_cn: str            # 中文名，如 "精神分裂症"
    probability: float          # 当前概率估计 0-1
    supporting: list[str]       # 支持该假设的已知事实
    against: list[str]          # 反对该假设的已知事实
    critical_unknowns: list[str]  # 若要确认/排除，还需要知道什么


# ---------------------------------------------------------------------------
# 已确立的临床事实
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    value: Any                  # 事实内容
    confidence: float           # 0-1
    first_turn: int
    source: Literal["patient", "observation", "collateral"]


# ---------------------------------------------------------------------------
# 主会话状态
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    # ── 身份 ─────────────────────────────────────────────────────────────
    student_id: str = ""
    session_id: str = ""
    turn_count: int = 0

    # ── 分诊 & 鉴别 ───────────────────────────────────────────────────────
    broad_category: str = "unknown"
    triage_done: bool = False           # 前2轮分诊是否完成
    differential: list[Hypothesis] = field(default_factory=list)

    # ── 已知事实（结构化，键名标准化）────────────────────────────────────
    facts: dict[str, Fact] = field(default_factory=dict)
    treatment_history: dict[str, Any] = field(default_factory=dict)

    # ── 探查状态 ──────────────────────────────────────────────────────────
    probe_target: str = ""          # 本轮 Reasoner 确定要探查的方向
    probed_topics: list[str] = field(default_factory=list)  # 已探查过的话题

    # ── 关系 & 表达风格 ───────────────────────────────────────────────────
    verbal_style: str = "unknown"   # expressive / silent / hyperverbal / resistant / somatic
    alliance_score: float = 0.5
    insight_status: str = "unknown" # intact / partial / impaired / unknown

    # ── 风险 ──────────────────────────────────────────────────────────────
    risk_level: str = "low"
    risk_factors: list[str] = field(default_factory=list)
    crisis_active: bool = False

    # ── 对话历史 ──────────────────────────────────────────────────────────
    history: list[dict[str, str]] = field(default_factory=list)
    last_assistant_output: str = ""

    # ── 报告 ──────────────────────────────────────────────────────────────
    session_report: dict[str, Any] | None = None
    session_ended: bool = False

    # ── 是否回诊 ──────────────────────────────────────────────────────────
    is_returning: bool = False

    def top_hypothesis(self) -> Hypothesis | None:
        if not self.differential:
            return None
        return max(self.differential, key=lambda h: h.probability)

    def to_context_dict(self) -> dict:
        """给 LLM 用的精简上下文（不含完整 history）。"""
        return {
            "turn": self.turn_count,
            "broad_category": self.broad_category,
            "triage_done": self.triage_done,
            "differential": [
                {
                    "disorder": h.disorder,
                    "disorder_cn": h.disorder_cn,
                    "probability": h.probability,
                    "supporting": h.supporting,
                    "against": h.against,
                    "critical_unknowns": h.critical_unknowns,
                }
                for h in self.differential
            ],
            "facts": {
                k: {"value": v.value, "confidence": v.confidence}
                for k, v in self.facts.items()
            },
            "treatment_history": self.treatment_history,
            "probe_target": self.probe_target,
            "probed_topics": self.probed_topics,
            "verbal_style": self.verbal_style,
            "alliance_score": self.alliance_score,
            "insight_status": self.insight_status,
            "risk_level": self.risk_level,
        }
