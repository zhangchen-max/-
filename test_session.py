"""
端到端测试：使用真实案例对话测试 V2 系统
案例：张道龙案例一（精神分裂症，24岁男性，3年病程）
"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from src.psy_nav.db import create_session, close_session, init_db, save_message, upsert_patient
from src.psy_nav.models import LLMClient
from src.psy_nav.agents import process_turn, run_report
from src.psy_nav.schema import SessionState

PATIENT_SCRIPT = [
    "是大脑神经感觉高度的紧张紧绷，然后会带来强烈强度的焦虑感。",
    "是感觉是大脑自己有问题，出现脑神经组织总是扭挤在一起，有点神经紊乱。",
    "主要是麻痹僵直的状态，神经麻痹僵直的状态。",
    "感觉就像脑袋被固定在一个类似有机玻璃一样的真空罩里，让人喘不上来气。",
    "大概是从脑袋和眼睛两个地方可以感受到别人说话传出的信号能从我这边擦肩而过，然后我接受不到。",
    "有的时候感觉自己的脑袋是通透的，就是连一阵风好像都能浮过来，直接从旁边掠走。",
    "没有人的时候，一紧张或者做东西，摆放书籍，整理日常用品顺序、时间，位置不对就会感觉非常紊乱。",
    "感觉正常应该是大脑支配我做这件事情，但现在感觉是大脑在支配我，因为他生病了让我难受，他让我来完成这些事情，我是处于被动的状态。",
    "对，他经常跟我这样说。",
    "大约都是白天的时候对话。",
    "三年了，我是阶段性住院的，最长一次是春夏秋三个季度。",
    "用过氯氮平，在北大六院的时候开过，效果很不好，会有神经痉挛，高度兴奋紧张。",
    "做过三个疗程的ECT，第一次感觉挺好的，后两次效果不太好。",
]


async def run_test():
    init_db()
    llm = LLMClient()

    state = SessionState(student_id="202301888")
    upsert_patient(state.student_id)
    state.session_id = create_session(state.student_id)

    print("=" * 65)
    print("  V2 测试：张道龙案例一（精神分裂症）")
    print("=" * 65)

    for i, patient_input in enumerate(PATIENT_SCRIPT, 1):
        state.turn_count = i
        print(f"\n[患者 T{i}]: {patient_input}")

        response, ended = await process_turn(state, patient_input, llm)

        save_message(state.session_id, state.student_id, i, "user", patient_input)
        save_message(state.session_id, state.student_id, i, "assistant", response)

        print(f"[系统回复]: {response}")

        # 状态摘要
        top = state.top_hypothesis()
        hyp_str = f"{top.disorder_cn} p={top.probability:.2f}" if top else "无"
        print(f"  category={state.broad_category} | probe={state.probe_target} | "
              f"alliance={state.alliance_score:.2f} | risk={state.risk_level}")
        print(f"  top_hypothesis={hyp_str}")
        if state.differential:
            for h in state.differential:
                print(f"    [{h.disorder_cn}] p={h.probability:.2f} "
                      f"unknowns={h.critical_unknowns[:2]}")

        if ended:
            print("\n[系统判断会话结束]")
            break

    # 生成报告
    print("\n[生成医生报告...]")
    report = await run_report(state, llm)
    state.session_report = report

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

    print("\n" + "=" * 65)
    print("  医生报告")
    print("=" * 65)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run_test())
