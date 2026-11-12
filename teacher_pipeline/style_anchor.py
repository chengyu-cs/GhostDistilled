"""L2 · 语录 few-shot 块（依据 retrieval_fewshot 调研）：

- 跨话题风格锚 25-40 条（分层轮换，按会话种子轮换——不同会话看到不同锚，防截图撞车）
- 语域条件句 8-15 条（先分语域再取样，EA-RAG 式；不按话题相似检索——证据表明最差）
- MMR 式多样性：4-gram 重叠高的候选跳过
- 块放记忆后、扮演指令前；配"学句式禁照搬"指令；提供 8-gram 防复读检查
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_TAG = re.compile(r"#[^#]{1,20}#|https?://\S+|\[[^\]]{1,6}\]|@\S{1,20}")
BLAND = re.compile(r"转发|抽奖|福利|直播|链接|http|晚安|早安|生日快乐|谢谢大家|关注|嗓子|发炎|生病|感冒|失眠|住院|吃药")


def _clean(t: str) -> str:
    return _TAG.sub("", t or "").replace("\n", " ").strip()


def load_pools() -> dict[str, list[str]]:
    """分层语录池：main(主号书面)/alt(小号日常)/spoken(口语字幕)/concert(现场)"""
    pools: dict[str, list[str]] = {"main": [], "alt": [], "spoken": [], "concert": []}
    src = ROOT / "evidence_pool_curated.jsonl"
    rows = []
    if src.exists():
        rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    else:
        rows = [json.loads(l) for l in open(ROOT / "corpus_public_safe_all.jsonl", encoding="utf-8")]
    seen = set()
    for r in rows:
        if r.get("source_owner") not in ("her", "her_alt", None):
            continue
        t = _clean(r.get("text", ""))
        if not (8 <= len(t) <= 45) or BLAND.search(t):
            continue
        if t in seen:
            continue
        seen.add(t)
        st = r.get("source_type", "")
        if r.get("source_owner") == "her_alt" or st == "weibo_alt_api":
            pools["alt"].append(t)
        elif "asr" in st or r.get("platform") == "bilibili":
            pools["spoken"].append(t)
        elif r.get("platform") == "concert":
            pools["concert"].append(t)
        elif r.get("platform") == "weibo":
            pools["main"].append(t)
    # 池子不够的层用别的层补
    for k in pools:
        if len(pools[k]) < 10:
            other = [t for kk, v in pools.items() if kk != k for t in v]
            pools[k].extend(other[: 30])
    return pools


POOLS: dict[str, list[str]] | None = None


def _pools() -> dict[str, list[str]]:
    global POOLS
    if POOLS is None:
        POOLS = load_pools()
    return POOLS


def _ngrams(s: str, n: int = 4) -> set:
    s = re.sub(r"\s", "", s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def _mmr_pick(cands: list[str], k: int, seed_offset: int = 0) -> list[str]:
    if len(cands) <= k:
        return cands
    picked: list[str] = []
    grams: list[set] = []
    i = seed_offset % len(cands)
    while len(picked) < k and cands:
        c = cands[i % len(cands)]
        g = _ngrams(c)
        if not any(len(g & pg) / max(1, len(g | pg)) > 0.35 for pg in grams):
            picked.append(c)
            grams.append(g)
        cands = cands[:i] + cands[i + 1:] if i < len(cands) else cands[:-1]
        if not cands:
            break
        i = seed_offset % len(cands)
    return picked


def classify_register(user_text: str) -> str:
    t = user_text
    if any(k in t for k in ["累", "难过", "烦", "压力", "紧张", "焦虑", "哭", "丧", "难受", "安慰"]):
        return "alt"          # 求安慰/吐槽 → 小号日常腔
    if any(k in t for k in ["歌", "唱", "听", "专辑", "单曲", "循环", "跑调", "音"]):
        return "spoken"
    if any(k in t for k in ["演唱会", "舞台", "巡演", "现场", "门票", "场馆"]):
        return "concert"
    if any(k in t for k in ["吃", "面包", "好吃", "喝", "店"]):
        return "alt"
    return "main"


def build_block(session_id: str, user_text: str, n_anchor: int = 28, n_reg: int = 10) -> tuple[str, list[str]]:
    """返回 (prompt 块, 本轮使用的语录列表)。块为空串表示不可用。"""
    pools = _pools()
    seed = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
    anchors: list[str] = []
    regs = ["main", "alt", "spoken", "concert"]
    per = max(4, n_anchor // 4)
    for ri, reg in enumerate(regs):
        pool = pools.get(reg) or []
        if not pool:
            continue
        offset = (seed + ri * 7) % max(1, len(pool))
        anchors.extend(_mmr_pick(list(pool), per, offset))
    reg = classify_register(user_text)
    pool = pools.get(reg) or pools["main"]
    offset = (seed >> 3) % max(1, len(pool))
    cond = _mmr_pick(list(pool), n_reg, offset)
    used = anchors + cond
    if len(used) < 8:
        return "", []
    body = "\n".join(f"- {s}" for s in used)
    block = (f"\n\n[她的真实语录（学句式和语气，禁止整句照搬）]\n{body}")
    return block, used


def copy_check(reply: str, used: list[str]) -> bool:
    """回复与任一锚句 8-gram 以上重叠 → True（需重生成）"""
    rg = _ngrams(reply, 8)
    if not rg:
        return False
    for s in used:
        sg = _ngrams(s, 8)
        if sg and rg & sg:
            return True
    return False
