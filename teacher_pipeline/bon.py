"""L2 · Best-of-N 两级重排：Tier-1 免费规则 + Tier-2 listwise judge。

依据 research_archive/2026-10-23_synthesis_testtime_plan.md：
- N=4 并行采样（生成端放开设温，选择端收紧）
- Tier-1：echo n-gram / 附和开头 / 开头形状重复 / 长度带 / AI 味词表 / 近重复去重
- Tier-2：一次 listwise judge，附 persona 锚 + 真实语录；score = persona × min(quality,1)
- 分差小不迷信 judge，用 Tier-1 序兜底
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# 她的真实公开语录（judge 锚，出处见 corpus_public_safe_all.jsonl）
ANCHOR_QUOTES = [
    "我就说我一上台雨就停了，她们不信！",
    "宝贝们别吃醋了！晚安，明天见！",
    "灯光亮起的瞬间，我在半空，努力地想要记住你们的每一张脸庞。",
]

AI_SLOP = re.compile(r"作为一?个?|总之|希望这|帮到你|以下是|首先|其次|综上所述|愿你也|记住，")
ACK_OPEN = re.compile(r"^(对的|好的|嗯嗯?|是啊?|没错|确实如此|当然啦?|是呀|对呀|哈哈，?好)[，,。！!～\s]")
PLAN_RE = re.compile(r"<plan>.*?</plan>", re.S)
LABEL_RE = re.compile(r"^AI小云[^\n]*\n?", re.S)
HOOKS = ("面包", "舞台", "老歌", "笑死", "魔星", "宇宙无敌", "开嗓", "唱", "吃", "练")


def _body(text: str) -> str:
    text = PLAN_RE.sub("", text or "")
    text = LABEL_RE.sub("", text.strip())
    return text.strip()


def _lcs_len(a: str, b: str) -> int:
    """最长公共子串长度（短文本 O(nm) 足够）。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _ngrams(s: str, n: int = 4) -> set:
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def tier1(cand: str, user_text: str, recent_opens: list[str]) -> int:
    """100 起分；负分=硬淘汰。"""
    b = _body(cand)
    if len(b) < 4 or len(b) > 60:
        return -1
    if AI_SLOP.search(b):
        return -1
    score = 100
    if ACK_OPEN.match(b):
        score -= 40
    e = _lcs_len(b, user_text)
    if e >= 8:
        return -1
    if e >= 5:
        score -= 35
    if b[:4] in recent_opens:
        score -= 30
    if any(h in b for h in HOOKS):
        score += 5
    if 10 <= len(b) <= 45:
        score += 5
    return score


def _dedupe(cands: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """近重复（4-gram Jaccard>0.6）只留分高的。"""
    out: list[tuple[str, int]] = []
    grams: list[set] = []
    for c, s in cands:
        g = _ngrams(_body(c))
        dup = False
        for i, (c2, s2) in enumerate(out):
            inter = len(g & grams[i])
            union = len(g | grams[i])
            if union and inter / union > 0.6:
                if s > s2:
                    out[i] = (c, s)
                    grams[i] = g
                dup = True
                break
        if not dup:
            out.append((c, s))
            grams.append(g)
    return out


JUDGE_SYSTEM = """你是人格保真评审，不是聊天机器人，也不参与对话。你评审的对象是“AI小云”（基于歌手黄霄雲公开语料的人格模拟）的候选回复。

人物锚（她的真实公开语录）：
- 我就说我一上台雨就停了，她们不信！
- 宝贝们别吃醋了！晚安，明天见！
- 灯光亮起的瞬间，我在半空，努力地想要记住你们的每一张脸庞。

人物要点：短句落点、口语、会自嘲、聊吃聊舞台会来劲；讨厌的是通用温柔陪聊腔（“辛苦啦，抱抱你，想说的话我都在听”这类放在任何 AI 身上都成立的句子）。

对每个候选打两个分（0-10 整数）：
- persona：像不像她本人说话。通用陪聊腔/客服腔=0-3；有点她的味道=4-6；一句就能听出是她=7-10。
- quality：是否接住用户这句话、自然、长度合适（闲聊 10-40 字最佳）。

只输出 JSON：{"scores":[{"i":1,"persona":7,"quality":8},...]}"""


def _judge_scores(api, survivors: list[str], user_text: str, retrieved: list[dict]) -> list[tuple[int, int]]:
    """一次 listwise 调用；失败返回 []。"""
    lines = [f"[{i}] {s.strip()}" for i, s in enumerate(survivors, 1)]
    evid = ""
    if retrieved:
        evid = "\n[本轮相关公开语料（供参照，不要求复述）]\n" + "\n".join(
            f"- {d.get('text','')[:80]}" for d in retrieved[:2])
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"用户最新一句：{user_text}\n{evid}\n\n候选：\n" + "\n".join(lines)
         + "\n\n只输出 JSON。"},
    ]
    raw = api(msgs, max_tokens=400, temperature=0.2)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        out = []
        for s in data.get("scores", []):
            i = int(s.get("i", 0)) - 1
            out.append((i, int(s.get("persona", 0)), int(s.get("quality", 0))))
        return [x for x in out if 0 <= x[0] < len(survivors)]
    except (ValueError, TypeError):
        return []


def run_bon(api, messages: list[dict], n: int, user_text: str,
            recent_opens: list[str], retrieved: list[dict], log_path: str,
            max_tokens: int = 600, temperature: float = 0.85,
            last_reply: str = "") -> tuple[str, dict]:
    """返回 (选中的原始回复, 元数据)。任何失败回退首候选。"""
    n = max(2, n)
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(api, messages, max_tokens, temperature) for _ in range(n)]
        cands = []
        for f in futs:
            try:
                cands.append(f.result())
            except Exception:  # noqa: BLE001
                cands.append("")
    cands = [c for c in cands if c and c.strip()]
    meta = {"n": n, "got": len(cands), "t1": [], "t2": None}
    if not cands:
        raise RuntimeError("all candidates failed")
    scored = [(c, tier1(c, user_text, recent_opens)) for c in cands]
    meta["t1"] = [s for _, s in scored]
    alive = _dedupe([(c, s) for c, s in scored if s >= 60])
    if not alive:  # 全灭：取最高分（可能是软伤）
        alive = sorted([x for x in scored if x[1] > 0] or scored[:1], key=lambda x: -x[1])[:1]
    alive.sort(key=lambda x: -x[1])
    if os.environ.get("PERSONA_BON_T1ONLY") == "1":
        # 无 judge 多样性选择：Tier-1 幸存者中选“与上一轮回复差异最大”的
        if len(alive) > 1 and last_reply:
            last_g = _ngrams(_body(last_reply))
            def dist_key(cs):
                g = _ngrams(_body(cs[0]))
                inter = len(g & last_g)
                union = len(g | last_g) or 1
                return cs[1] + 20.0 * (1.0 - inter / union)
            alive = sorted(alive, key=dist_key, reverse=True)
        meta["winner"] = "t1_diversity"
        _log(log_path, user_text, meta)
        return alive[0][0], meta
    if len(alive) == 1:
        meta["winner"] = "t1"
        _log(log_path, user_text, meta)
        return alive[0][0], meta
    survivors = [c for c, _ in alive]
    try:
        js = _judge_scores(api, survivors, user_text, retrieved)
    except Exception:  # noqa: BLE001
        js = []
    meta["t2"] = js
    best, best_v = 0, -1.0
    if js:
        for i, p, q in js:
            v = (p / 10.0) * min(q / 10.0, 1.0)
            if v > best_v:
                best, best_v = i, v
        top = sorted(js, key=lambda x: -(x[1] / 10.0) * min(x[2] / 10.0, 1.0))
        if len(top) >= 2 and (top[0][1] - top[1][1]) <= 1 and (top[0][2] - top[1][2]) <= 1:
            best = 0  # 分差在噪声带：不迷信 judge，用 Tier-1 序
            meta["winner"] = "t1_tiebreak"
        else:
            meta["winner"] = "judge"
        _log(log_path, user_text, meta)
        return survivors[best], meta
    meta["winner"] = "t1_judge_fail"
    _log(log_path, user_text, meta)
    return survivors[0], meta


def _log(path: str, user_text: str, meta: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%F %T"), "user": user_text[:60], **meta},
                               ensure_ascii=False) + "\n")
    except OSError:
        pass
