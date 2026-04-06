# PsyNav V2 — 精神科初诊信息采集系统

基于假设驱动鉴别诊断的精神科访谈 Agent，使用 DeepSeek LLM + MySQL + FastAPI 构建。

---

## 设计思路

### 核心问题

传统症状清单式采集（逐项打勾"有没有幻听/有没有妄想"）有两个缺陷：
1. 问题顺序固定，遇到不配合或表达方式特殊的患者容易失效
2. 采集的是症状列表，不是鉴别诊断所需的关键分叉点

PsyNav V2 的设计参考张道龙老师的假设优先访谈方法：**先建假设，再用信息增益驱动问题**。

---

### 整体流程

```
患者输入
   │
   ├─ [前2轮] 分诊 Agent
   │    └─ 判断谱系大类（精神病性/情感/焦虑/神经发育/器质性）
   │    └─ 生成 2-4 个初始假设，每个带支持证据、反对证据、关键未知项
   │
   └─ [每轮并行]
        ├─ Risk Monitor：实时扫危机信号（自杀/暴力意图），独立于诊断流程
        ├─ Reasoner：更新假设概率 + 选下一个探查目标
        └─ Dialogue：把探查目标翻译成一句自然中文问句
```

---

### 三个关键设计

**1. 假设驱动，不是症状累积**

每个假设（Hypothesis）是一个活的数据结构：

```
disorder: "schizophrenia"
probability: 0.72
supporting: ["持续幻听", "思维被控制感", "功能下降超过6个月"]
against: []
critical_unknowns: ["情感症状时间线", "是否有躁狂发作史"]
```

每轮患者说的内容会更新所有假设的概率——支持证据出现则升，反对证据出现则降。这是贝叶斯更新的逻辑，但由 LLM 推理执行而非精确计算。

**2. 信息增益选题**

Reasoner 每轮从所有假设的 `critical_unknowns` 里选一个探查目标，选择标准：**问了之后最能区分剩余候选假设**。

例如：当精神分裂症（p=0.6）和双相障碍伴精神病性特征（p=0.3）同时在列时，优先问"情感症状和精神病症状哪个先出现"——这一个问题能同时为两个假设提供强证据，而不是追问幻觉细节（只对精分有用）。

精神病谱系有专项探查目标表（13项），防止 Reasoner 在 duration/mood_history 之间反复循环。

**3. broad_category 动态更新**

分诊在第2轮完成，但诊断会随对话演进。当最高概率假设超过 0.65 时，系统自动将 `broad_category` 更新为对应谱系，后续 Dialogue Agent 的语气和探查策略随之调整。

---

### 会话结束逻辑

不依赖置信度阈值，而是由 Reasoner 判断"关键未知项是否已基本填完"。当继续问的边际收益很低时，Reasoner 输出 `session_should_end: true`，系统生成医生报告。

报告是结构化临床文档（给接诊医生看），不是给患者的结论。患者只听到"感谢分享，建议去就诊"。

---

### 架构

```
入口
├── main.py          CLI 交互（python main.py）
└── api.py           HTTP API（uvicorn，端口 8000）
     ├── /sessions/* 原生接口
     └── /v1/*       OpenAI 兼容接口（含 benchmark 模式）

核心逻辑
└── src/psy_nav/
     ├── agents.py   五个 Agent + process_turn 主循环
     ├── prompts.py  所有 Agent 的 System Prompt
     ├── schema.py   SessionState / Hypothesis / Fact 数据结构
     ├── models.py   LLM 客户端（AsyncOpenAI + tenacity 重试）
     ├── db.py       MySQL 持久化（patients / session_records / messages）
     └── loop.py     CLI 会话主循环
```

**五个 Agent 职责：**

| Agent | 触发时机 | 输出 |
|---|---|---|
| Triage | 第2轮，仅一次 | broad_category + 初始 differential |
| Reasoner | 每轮 | 更新 differential + probe_target |
| Dialogue | 每轮 | 患者听到的一句话 |
| Risk Monitor | 每轮（并行） | risk_level + is_crisis |
| Report | 会话结束时 | 医生报告 JSON |

---

## 快速开始

**环境变量（.env）**

```
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=psy_nav
```

**运行 API 服务**

```bash
pip install -r requirements.txt
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

**CLI 模式**

```bash
python main.py
```

**接口说明**

```
POST /sessions              创建会话，传 {"student_id": "xxx"}
POST /sessions/{id}/turns   发送患者输入，传 {"content": "..."}
POST /sessions/{id}/end     结束会话，返回完整医生报告
GET  /sessions/{id}/state   查看当前诊断状态
```

---

## Benchmark 模式

当 OpenAI 请求的 system message 包含 `"只能输出JSON"` 时，自动切换为 benchmark 协议：

- ASK prompt（含 `allowed_slots` / `must_ask_slots`）→ 输出 `{"action":"ASK","slot_id":"...","utterance":"..."}`
- FINAL 之后的诊断请求 → 输出标准诊断 JSON（diagnosis_topk + risk_flags + justification）

普通对话不受影响。
