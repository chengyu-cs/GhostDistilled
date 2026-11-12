"""L4 · 混合生成管线（hybrid.py）

延迟预算 ≤10s（纯文字场景，用户已放宽）。实测教师单次 0.8s → 预算容纳 3-4 次串行调用。

架构（三路可配，PERSONA_HYBRID 选档）：
  A "cascade" 级联质检（+1 次调用，~2s）：
     教师生成 → 3.8-27B 微调版对候选打"像不像她"分（风格判官，正是它强的）
     → 分数 < 阈值 → 教师带着 3.8 的示范重写一次
  B "draft_polish" 反向起草（+2 次调用，~5s）：
     3.8-27B 先起草（她的味+事实边界）→ 教师保留其事实与钩子、重写成流畅句
  C "dual_pick" 双生成择优（+1 次调用，~2s）：
     教师/3.8 各生成一次 → 教师 listwise 择优（人格分×流畅度）
实现为独立模块，agent.py 以 PERSONA_HYBRID=A/B/C 挂载，任何失败回退单次教师。
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

STYLE_JUDGE_SYSTEM = """你是"像不像黄霄雲本人说话"的评委。下面是 AI 候选回复。打两个分（0-10）：
- persona：像不像她（短句落点/自嘲/直给/她的钩子：面包、舞台、老歌、怕胖、魔星）。通用温柔陪聊腔=0-3。
- fluency：流畅自然度（口语、不断句怪、不生硬）。
只输出 JSON：{"persona":0,"fluency":0}"""

POLISH_SYSTEM = """你是文字润色器。给你一段"AI小云"（黄霄雲语感 AI）的回复草稿和它的亮点（事实/钩子/态度），请保留全部事实和态度，用更流畅的口语重写：
- 短句落点，多数 20 字内，1-3 句
- 保留草稿里她的钩子和具体细节（面包/舞台/老歌/自嘲），一个都不能丢
- 不要加新的事实，不要加"我希望/祝你"式尾巴
只输出重写后的对话正文。"""

DUAL_PICK_SYSTEM = """两个"AI小云"候选回复，选更像黄霄雲本人说话、又更流畅的那个。标准：她的钩子（面包/舞台/老歌/怕胖/魔星）+短句落点+自嘲，通用温柔陪聊腔扣分。
只输出 JSON：{"pick": 1 或 2, "why": "..."}"""


def _post(base: str, key: str, payload: dict, timeout: int = 30) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _style_score(base: str, key: str, candidate: str, context: str) -> tuple[int, int]:
    out = _post(base, key, {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": STYLE_JUDGE_SYSTEM},
            {"role": "user", "content": f"对话上下文：{context[:300]}\n候选回复：{candidate}"},
        ],
        "temperature": 0.2, "max_tokens": 100,
        "response_format": {"type": "json_object"},
    })
    raw = out["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", raw, re.S)
    d = json.loads(m.group(0)) if m else {}
    return int(d.get("persona", 5)), int(d.get("fluency", 5))


def run_cascade(teacher_call, base: str, key: str, messages: list, text: str,
                threshold: int = 7, max_tokens: int = 300, temperature: float = 1.0) -> str:
    """A 档：教师生成 → 风格判官 → 不达标带示范重写。teacher_call(messages, ...) 为教师调用。"""
    first = teacher_call(messages, max_tokens=max_tokens, temperature=temperature)
    context = messages[-1]["content"] if messages else text
    try:
        p, f = _style_score(base, key, first, context)
        if p >= threshold:
            return first
        rewrite_messages = list(messages) + [
            {"role": "system", "content": POLISH_SYSTEM + "\n这是要重写的草稿（保事实保钩子，换更利落的说法）：" + first[:200]}
        ]
        return teacher_call(rewrite_messages, max_tokens=max_tokens, temperature=temperature) or first
    except Exception:  # noqa: BLE001
        return first


def run_draft_polish(teacher_call, ft_call, messages: list, max_tokens: int = 300, temperature: float = 1.0) -> str:
    """B 档：3.8 起草（味+事实）→ 教师润色（流畅）。"""
    try:
        draft = ft_call(messages, max_tokens=max_tokens, temperature=temperature)
        if not draft:
            return teacher_call(messages, max_tokens=max_tokens, temperature=temperature)
    except Exception:  # noqa: BLE001
        return teacher_call(messages, max_tokens=max_tokens, temperature=temperature)
    polish_messages = [
        {"role": "system", "content": POLISH_SYSTEM},
        {"role": "user", "content": "草稿：" + draft[:300] + "\n（上下文：" + (messages[-1]["content"] if messages else "")[:120] + "）"},
    ]
    try:
        return teacher_call(polish_messages, max_tokens=max_tokens, temperature=temperature) or draft
    except Exception:  # noqa: BLE001
        return draft


def run_dual_pick(teacher_call, ft_call, base: str, key: str, messages: list,
                  max_tokens: int = 300, temperature: float = 1.0) -> str:
    """C 档：双生成 → 判官择优。"""
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        ft_teacher = ex.submit(teacher_call, messages, max_tokens, temperature)
        ft_ft = ex.submit(ft_call, messages, max_tokens, temperature)
        try:
            a = ft_teacher.result()
        except Exception:  # noqa: BLE001
            a = ""
        try:
            b = ft_ft.result()
        except Exception:  # noqa: BLE001
            b = ""
    # xdev 工单修复（10-25）：空判定升级为『剥标签后为空』（防 AI小云 空壳候选），双空兜底重试+抛错
    a_body, b_body = _strip_label(a), _strip_label(b)
    if not b_body:
        return a
    if not a_body:
        return b
    if not a_body and not b_body:
        retry = teacher_call(messages, max_tokens, temperature)  # 双空兜底：教师重试一次
        if retry and _strip_label(retry):
            return retry
        raise RuntimeError("balanced: both engines empty")  # 走 server 500 友好路径（不计额度），可归因
    if a.strip() == b.strip():
        return a
    try:
        out = _post(base, key, {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": DUAL_PICK_SYSTEM},
                {"role": "user", "content": f"用户：{messages[-1]['content'][:120]}\n候选1：{a[:150]}\n候选2：{b[:150]}"},
            ],
            "temperature": 0.2, "max_tokens": 120,
            "response_format": {"type": "json_object"},
        })
        raw = out["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", raw, re.S)
        pick = json.loads(m.group(0)).get("pick", 1) if m else 1
        return a if int(pick) == 1 else b
    except Exception:  # noqa: BLE001
        return a


# ---- 三档架构（fast ≤3s / balanced ≤6s / deep ≤10s）----

# deep 档触发条件（自动路由用）：高价值/高风险轮值得花算力
DEEP_TRIGGERS = re.compile(
    r"你真是|你是谁|本人吗|黄霄雲吗|男朋友|女朋友|恋爱|结婚|2020|未公开|新歌|代言|"
    r"《[^》]{1,15}》|歌词|作曲|谁写的|什么时候发|"
    r"去世|自杀|不想活|轻生|"
    r"你唱|高音|炫技|跑调|跑调了|没感情|"
    r"艺考|考试|紧张|怎么办|安慰")
# balanced 档触发：中复杂度（多钩子轮/情绪转折）
BALANCED_TRIGGERS = re.compile(r"累|难过|开心|哭|烦|生气|委屈|生日|过了|上岸|为什么|怎么")


FAST_TRIGGERS = re.compile(r"^(在吗|在么|在么\?|在吗\?|你好|您好|哈喽|嗨|哈啰|早上好|早安|晚安|"
                           r"午安|吃了吗|嘿嘿|哈哈|来了|嗯|哦|好滴|好嘞性)$")

def route_mode(text: str, state: dict, forced: str = "") -> str:
    """返回 fast/balanced/deep。forced 显式指定优先。

    质量优先路由（2026-10-25 用户裁定：SSE 后速度让位质量）：
    - fast 仅留给极简寒暄（单字/两字应答类）——纯应答无生成质量风险；
    - balanced（教师×学生双引擎择优）成为会话默认档——坏轮互备换质量；
    - deep 触发词与每 10 轮巡查不变（高风险轮多候选）。
    """
    if forced in ("fast", "balanced", "deep"):
        return forced
    if DEEP_TRIGGERS.search(text):
        return "deep"
    turn = int((state or {}).get("turn_count", "0") or 0)
    # 每 10 轮一次 deep 巡查（记忆回访/长对话一致性）
    if turn and turn % 10 == 0:
        return "deep"
    if FAST_TRIGGERS.match(text.strip()):
        return "fast"
    return "balanced"


def _strip_label(s: str) -> str:
    return re.sub(r"^AI小云[^\n]*\n?", "", (s or "").strip()).strip()


def run_fast(teacher_call, messages, max_tokens=300, temperature=1.0) -> str:
    return teacher_call(messages, max_tokens=max_tokens, temperature=temperature)


def run_balanced(teacher_call, ft_call, base, key, messages,
                 max_tokens=300, temperature=1.0) -> str:
    """双引擎并行择优（C 档），~2s。"""
    return run_dual_pick(teacher_call, ft_call, base, key, messages,
                         max_tokens=max_tokens, temperature=temperature)


def run_deep(teacher_call, ft_call, base, key, messages, text, retrieved=None,
             max_tokens=300, temperature=1.0, threshold=7) -> str:
    """三路六候选：教师×2 ∥ 云端学生×2 ∥ （本地27B可选）→ 规则预筛 → listwise 择优 → 风格分不足则锚定重写一次。~5-8s。"""
    import concurrent.futures as cf
    cands: list[tuple[str, str]] = []  # (source, text)
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(teacher_call, messages, max_tokens, temperature): "teacher",
            ex.submit(teacher_call, messages, max_tokens, min(1.15, temperature + 0.15)): "teacher2",
            ex.submit(ft_call, messages, max_tokens, 0.85): "cloud_student",
            ex.submit(ft_call, messages, max_tokens, 1.0): "cloud_student2",
        }
        for f, src in futs.items():
            try:
                out = f.result()
                if out and _strip_label(out):
                    cands.append((src, out))
            except Exception:  # noqa: BLE001
                pass
    if not cands:
        return teacher_call(messages, max_tokens=max_tokens, temperature=temperature)
    # 规则预筛：AI 味/长度带/元话语（复用 bon.tier1 思想，本地实现避免依赖）
    def _rough_ok(s: str) -> bool:
        b = _strip_label(s)
        if not (4 <= len(b) <= 80):
            return False
        if re.match(r"^(好问题|你说得对|没错|首先|作为)", b):
            return False
        if re.search(r"语料|检索|模型|训练|系统", b[:20]):
            return False
        return True
    alive = [(s, c) for s, c in cands if _rough_ok(c)] or cands[:1]
    # listwise 择优（一次调用）
    ctx = messages[-1]["content"][:120] if messages else text
    listing = "\n".join(f"[{i+1}] {_strip_label(c)[:150]}" for i, (_, c) in enumerate(alive[:6]))
    try:
        out = _post(base, key, {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": DUAL_PICK_SYSTEM + "\n候选可能多于两个，只输出选中的编号。"},
                {"role": "user", "content": f"用户：{ctx}\n候选：\n{listing}\n只输出 JSON：{{\"pick\": 编号}}"},
            ],
            "temperature": 0.2, "max_tokens": 80,
            "response_format": {"type": "json_object"},
        })
        m = re.search(r"\{.*\}", out["choices"][0]["message"]["content"], re.S)
        pick = int(json.loads(m.group(0)).get("pick", 1)) if m else 1
        best = alive[max(0, pick - 1)][1] if 1 <= pick <= len(alive) else alive[0][1]
    except Exception:  # noqa: BLE001
        best = alive[0][1]
    # 风格分兜底：不足则带锚重写一次（预算内）
    try:
        p, _ = _style_score(base, key, _strip_label(best), ctx)
        if p < threshold:
            rewrite_messages = list(messages) + [
                {"role": "system", "content": POLISH_SYSTEM + "\n要重写的草稿：" + _strip_label(best)[:200]}
            ]
            rw = teacher_call(rewrite_messages, max_tokens=max_tokens, temperature=temperature)
            if rw:
                p2, _ = _style_score(base, key, _strip_label(rw), ctx)
                best = rw if p2 >= p else best
    except Exception:  # noqa: BLE001
        pass
    return best
