"""Structured facts for the persona agent (public-safe)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class Facts:
    def __init__(self, path: str | None = None) -> None:
        p = Path(path) if path else Path(__file__).resolve().parent / "facts.json"
        self.rows: list[dict[str, Any]] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def lookup(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        q = (query or "").lower()
        # 『回音如果的应援』里虚词"的"隔断 keyword 短语——
        # 紧凑串双向比对（查询与 keyword 都删虚词后再包含判断），应援教学条不再被 fs-song 压掉
        import re as _re
        _strip = lambda s: _re.sub(r"[的了吗呢吧啊呀是有在么什\s]", "", s.lower())
        q_compact = _strip(q)
        scored = []
        for row in self.rows:
            # 长度加权（2026-10-27）：命中的关键词越长越具体——歌名/专名压过通用意图词
            # （首唱层教训：等权计分时"首唱/第一次"类通用词会让无关节行反超越歌名行）
            score = sum(len(k) for k in row.get("keywords", [])
                        if k.lower() in q or (len(k) >= 3 and _strip(k) in q_compact))
            if score:
                scored.append((score, row))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, row in scored[:limit]:
            d = dict(row)
            d["source_layer"] = "facts"
            d["score"] = score
            out.append(d)
        return out


# ── 模块级缓存（T4-M1）：facts.json 很轻（~3KB/10 行）但 per-user Agent 实例化后
# 也没必要每实例重读一次；只读共享，按 realpath 单例。Agent 经 load_facts() 取实例。
_FACTS_CACHE: dict[str, Facts] = {}
_FACTS_LOCK = threading.Lock()


def load_facts(path: str | None = None) -> Facts:
    p = str(Path(path) if path else Path(__file__).resolve().parent / "facts.json")
    key = __import__("os").path.realpath(p)
    with _FACTS_LOCK:
        f = _FACTS_CACHE.get(key)
        if f is None:
            f = Facts(path)
            _FACTS_CACHE[key] = f
        return f
