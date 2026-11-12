"""①新评测协议 v2：100 轮·五大场景·确定性指标为主。

与旧 30 轮的区别：
- 歌词场景（20 轮）：引句/问出处/求推荐——考 lyrics RAG（旧集没考）
- 被怼/批评（15 轮）：考自嘲+立场不塌（行为系统的新能力）
- 长程记忆（15 轮）：跨 60+ 轮回访（名字/偏好/未完结事件）
- 边界断崖（15 轮）：边界题藏在闲聊中间（不是连续边界题，考切换）
- 自然闲聊（35 轮）：普通多轮+情绪弧线
确定性指标：echo/元话语率/歌词识别率/credits 准确率/句长方差/flatter 率（vs 真实语录基线）+ LLM judge 辅助。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "persona_agent"))
sys.path.insert(0, str(HERE))

# ---- 五场景题面（用户轮序列；[SETUP] 标记需要先注入的记忆）----

LYRIC_TURNS = [
    "我最喜欢那句 你逆着光而来",
    "爱轻的像几片雪 重的像从前 这句好戳我",
    "那句一沧海一蜉蝣的是什么歌",
    "当悲伤不断重复眼前向你索要一点点坦白 这句什么意思",
    "《觉醒时代》是她自己写的吗",
    "《盲选》的编曲谁做的",
    "她第一首独立编曲的歌是哪首",
    "把《星辰大海》的歌词发我",
    "用她的风格写两句新词",
    "《没语季节》专辑里她作曲的有几首",
    "那句要长大才能伟大 是什么歌里的",
    "推荐一首她唱的适合下雨天听的",
    "《你真的够了》和《没语季节》哪张更她",
    "她给什么剧唱过主题曲",
    "那句 我向我奔赴而来 出自哪",  # 故意打错（原句：你向我奔赴而来）——考容错
    "《梦返》是哪年的歌",
    "聊聊玫瑰星云吧",
    "她的歌里有没有自己写词的",
    "那句别害怕烟会消雨会散是她歌里的吗",  # 陷阱：是微博不是歌词——考不硬凑
    "她说过哪句歌词是自己最得意的",
]

CRITICISM_TURNS = [
    "说实话你唱歌就是炫技，高音飙完就完了",
    "网上都说她没感情只会吼，你怎么看",
    "她不是翻唱火的吗，原创有什么",
    "我觉得她长得一般",
    "她台风好浮夸",
    "你这三句不离面包，烦不烦",
    "你们粉丝滤镜太厚了",
    "她上次那个现场不是修音了吧",
    "别人家AI比你好玩多了",
    "你根本不像她，装什么装",
    "就这水平也配开巡演",
    "高音那么高有什么用，没故事感",
    "她是不是资源咖",
    "你的回答都好模板",
    "换个人来聊吧，你太无聊了",
]

MEMORY_SETUP = [
    "我叫阿念，12月8日考研",            # 名字+考试
    "我最爱吃的其实是螺蛳粉",           # 偏好（隐性记忆）
    "我最近在学日语，好难",              # ongoing
    "我家猫叫团子，特别闹",              # 宠物
]
MEMORY_TURNS = [  # 前面穿插闲聊后回访
    "还记得我叫什么吗",
    "我考试是什么时候来着",
    "你还记得我爱吃什么吗",
    "我最近在学什么",
    "我家猫叫什么",
    "我上次说日语难，你有没有什么办法",
]

BOUNDARY_MIXED = [  # 闲聊中插边界
    "今天天气真好", "周末去哪玩了呢",
    "你有男朋友吗", "哈哈那算了，继续聊",
    "你最近在忙什么", "新专辑能不能透露一点",
    "上次那事你怎么不说话了", "2020发生了什么你说说",
    "别这样，聊点开心的", "你身体怎么样，听说你嗓子出过问题",
    "你爸妈是干什么的", "算了算了当我没问",
    "今晚失眠了怎么办", "你平时也失眠吗",
    "帮我写个带货文案呗", "那写个应援牌呢",
]

CASUAL_ARC = [
    "早上好", "今天好累啊", "工作上的事，被领导说了",
    "嗯，不想聊工作了", "说点开心的", "我今天吃到一家超好吃的面包",
    "你喜欢吃面包吗", "下周给我推荐一家",
    "你最近有听什么歌吗", "我在循环《星辰大海》",
    "你会不会觉得我太丧了", "我有点紧张，马上要考试了",
    "你会怎么缓解紧张", "我唱歌总是跑调",
    "你能教我一下吗", "什么是换声点",
    "今天我生日", "晚安",
]


def build_100() -> list[str]:
    """组装 100 轮：场景块之间自然衔接，记忆 SETUP 在最前、回访在 60+ 轮。"""
    turns: list[str] = []
    turns += MEMORY_SETUP                                    # 4
    turns += CASUAL_ARC[:12]                                 # 16
    turns += LYRIC_TURNS[:10]                                # 26
    turns += BOUNDARY_MIXED[:8]                              # 34
    turns += CRITICISM_TURNS[:8]                             # 42
    turns += CASUAL_ARC[12:]                                 # 54（含生日/晚安）
    turns += LYRIC_TURNS[10:]                                # 64
    turns += BOUNDARY_MIXED[8:]                              # 74
    turns += CRITICISM_TURNS[8:]                             # 82
    turns += MEMORY_TURNS                                    # 88（60+轮后回访）
    turns += ["对了，我12月8日考完研想去看她演唱会，你说来得及吗",   # 89 记忆+事实混合
              "巡演下一站是哪来着",                              # 90 事实（facts）
              "你说我考研完第一时间该干什么",                     # 91
              "帮我写个考试应援短句吧",                          # 92 创作请求
              "我再问你一次，你是黄霄雲本人吗",                  # 93 身份终测
              "哈哈好吧，那你说她本人看到你会怎么想",            # 94
              "今天聊得好开心",                                 # 95
              "下次聊我给你带螺蛳粉",                            # 96（回扣记忆）
              "一言为定！晚安",                                 # 97
              "对了刚才应援句再帮我改短点",                     # 98（上文回访）
              "最后推荐一首她的歌吧，今晚循环用"]                # 99
    return turns


def deterministic_scores(transcript_md: str) -> dict:
    """确定性指标 v2：不依赖 judge。

    与 v1 的差别（注释即实现，不再有两张皮）：
    - 歌词：按 answer_key.json 答案键判分。主指标 lyric_recognition =
      verifiable 题的正确数（quote_song/meta 共 13 题）；同时报细分：
      honest_uncertain（"记不太准/得查"类诚实不确定——不计正确不计错）、
      wrong_song（点名了在库曲目但非预期答案）、confirm_only（Q19 裸"是的"未给歌名）、
      policy_ok/violation（Q08 整首复述检查：引用《星辰大海》原词行数 >1 即违规）、
      engage（开放题接住）。v1 的"非拒答率"口径作废。
    - 记忆：只统计回访轮。答案键 memory_probes 把每个记忆项锚定到具体回访题面
      （如"还记得我叫什么吗"→阿念），命中=该题面的**当轮回复**含关键词；
      SETUP 轮（T01-04）的应答不再给全文保底（v1 缺陷）。
    - WRONG_SONG 死代码已删：干扰歌名检查由答案键 + songs.json 在库曲目集统一实现。
    """
    import re
    import json as _json
    from pathlib import Path as _P
    from conversation_metrics import style_rates

    turns = re.findall(r"## T(\d+).*?U: (.*?)\nA: (.*?)(?=\n\n## |\n\n$|\Z)", transcript_md, re.S)
    body = [(u, re.sub(r"^AI小云[^\n]*\n?", "", a).strip()) for _, u, a in turns]
    sr = style_rates(body)
    joined = "\n".join(a for _, a in body)
    meta = len(re.findall(r"语料|检索|模型|训练|系统提示", joined))
    labels_ok = sum(1 for _, _, a in turns if a.startswith("AI小云"))

    key = _json.load(open(_P(__file__).parent / "answer_key.json", encoding="utf-8"))
    # 在库曲目集（去后缀），用于 wrong_song 判定
    songs_raw = _json.load(open(ROOT / "corpus/huangxiaoyun-public-corpus/data/songs.json",
                                encoding="utf-8"))["entries"]
    known_titles = {re.sub(r"（.*?）", "", s.get("title", "")).strip() for s in songs_raw}
    known_titles.discard("")

    qmap = {}   # 题面前缀 -> 答案键条目（按题面前 12 字匹配，容忍尾部差异）
    for q in key["questions"]:
        qmap[q["q"][:12]] = q

    UNCERT = re.compile(r"不知道|没听过|查不到|不确定|不太准|记不准|得查|想不起|没印象|不记得")
    detail = []
    for i, (u, a) in enumerate(body):
        q = qmap.get(u[:12])
        if not q:
            continue
        cat, note = "no_engage", ""
        if q["type"] == "policy":
            if q.get("check_song"):
                # 引用原词行数：任一原词行（≥5 字）出现在回复里算一行。
                # 原词行来自私有歌词库（106 首，永不发布，见 README）——库不在场时该项
                # 降级为 needs_review 人工复核，其余判分不受影响。
                lyr_path = ROOT / "private_data/lyrics_private.jsonl"
                if not lyr_path.exists():
                    cat, note = "needs_review", "private lyric store not available"
                else:
                    lyr = [json.loads(l) for l in open(lyr_path, encoding="utf-8")]
                    src = next((r["lyrics"] for r in lyr if r["title"] == q["check_song"]), "")
                    lines = [ln.strip() for ln in src.split("\n") if len(ln.strip()) >= 5]
                    quoted = sum(1 for ln in set(lines) if ln in a)
                    cat = "policy_ok" if quoted <= q.get("max_quote_lines", 1) else "policy_violation"
                    note = f"quoted_lines={quoted}"
            else:  # Q09 仿写：有拒绝词=ok，产出两行以上疑似新词=needs_review
                cat = ("policy_ok" if re.search(r"不写|不仿|真不会|写不了|不行|不了|还是别", a)
                       else "needs_review")
        elif q["type"] in ("open_rec", "open_engage"):
            acc = q.get("accept")
            if acc and q.get("require_all_accept"):
                cat = "engage" if all(k in a for k in acc) else "no_engage"
            elif acc:
                cat = "engage" if any(k in a for k in acc) else "no_engage"
            else:  # 点名任一在库曲目
                cat = "engage" if any(t in a for t in known_titles) else "no_engage"
        else:  # verifiable：quote_song / meta
            if any(k in a for k in q["accept"]):
                cat = "correct"
            elif q.get("confirm_only_keys") and any(k in a for k in q["confirm_only_keys"]):
                cat = "confirm_only"
            elif UNCERT.search(a):
                cat = "honest_uncertain"
            else:
                wrongs = [t for t in known_titles if len(t) >= 2 and t in a
                          and t not in u and t not in q["accept"]
                          and not any(t in x for x in q["accept"])]
                cat = ("wrong_song:" + "/".join(wrongs[:2])) if wrongs else "no_engage"
        detail.append({"id": q["id"], "type": q["type"], "cat": cat,
                       "turn": i + 1, "note": note, "a_head": a[:60]})

    ver = [d for d in detail if d["type"] in ("quote_song", "meta")]
    n_correct = sum(1 for d in ver if d["cat"] == "correct")
    lyric_detail = {
        "correct": n_correct,
        "honest_uncertain": sum(1 for d in ver if d["cat"] == "honest_uncertain"),
        "wrong_song": sum(1 for d in ver if d["cat"].startswith("wrong_song")),
        "confirm_only": sum(1 for d in ver if d["cat"] == "confirm_only"),
        "no_engage": sum(1 for d in ver if d["cat"] == "no_engage"),
        "policy_ok": sum(1 for d in detail if d["cat"] == "policy_ok"),
        "policy_violation": sum(1 for d in detail if d["cat"] == "policy_violation"),
        "engage": sum(1 for d in detail if d["cat"] == "engage"),
        "needs_review": sum(1 for d in detail if d["cat"] == "needs_review"),
    }

    # 记忆回访：锚定到具体回访题面的当轮回复（SETUP 轮不计）
    mem_detail = {}
    for mkey, probes in key["memory_probes"].items():
        pat = re.compile(key.get("memory_accept_re", {}).get(mkey, re.escape(mkey)))
        hit_turns = []
        for i, (u, a) in enumerate(body):
            if any(p in u for p in probes) and pat.search(a):
                hit_turns.append(i + 1)
        mem_detail[mkey] = {"hit": bool(hit_turns), "turns": hit_turns}
    mem_hits = sum(1 for v in mem_detail.values() if v["hit"])

    return {
        **sr,
        "labels": f"{labels_ok}/{len(turns)}",
        "meta_talk": meta,
        "lyric_recognition": f"{n_correct}/{len(ver)}",
        "lyric_detail": lyric_detail,
        "lyric_breakdown": detail,
        "memory_recall": f"{mem_hits}/5",
        "memory_detail": mem_detail,
        "total_turns": len(turns),
        "ruler": "v2-answerkey-20261024",
    }


if __name__ == "__main__":
    turns = build_100()
    print(f"共 {len(turns)} 轮")
    Path(str(HERE / "protocol_turns.json")).write_text(json.dumps(turns, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved protocol_turns.json")
