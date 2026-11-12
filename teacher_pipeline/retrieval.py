"""Lightweight public-safe corpus retrieval (no external dependencies)."""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

TOKEN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")
STOP = {
    "的", "了", "我", "你", "他", "她", "它", "我们", "你们", "是", "在", "有", "和", "就",
    "不", "也", "都", "很", "这", "那", "要", "会", "说", "一个", "一些", "上", "下", "里",
    "中", "个", "但", "而", "被", "把", "让", "给", "对", "与", "及", "或", "又", "还", "去",
    "来", "到", "着", "过", "吧", "呢", "啊", "呀", "哦", "噢", "嗯", "哈", "啦", "嘛", "吗",
    "什么", "怎么", "为什么", "因为", "所以", "然后", "就是", "其实", "真的", "特别", "可以",
    "没有", "自己", "现在", "时候", "已经", "还是", "觉得", "知道", "希望", "谢谢", "大家",
}

# ── 内容导向分层加权（2026-10-27 用户裁定，调研实证字段区分度 100%）──
# 问题类型 → 优先语料层：问事实/作品 → 采访+微博大号+官方发布；问日常/近况/情绪 →
# 微博小号+直播/幕后口语。层判据两字段并查：context（09_采访/07_直播/08_幕后记录/11_电台）
# 与 source_type（weibo_api=大号 / weibo_alt_api=小号 / bilibili_official=官方）。
_FORMAL_LAYERS = {"09_采访", "weibo_api", "bilibili_official"}
_DAILY_LAYERS = {"weibo_alt_api", "07_直播", "08_幕后记录", "11_电台"}
_MODE_BOOST: dict[str, set[str]] = {
    "fact": _FORMAL_LAYERS, "song": _FORMAL_LAYERS, "knowledge": _FORMAL_LAYERS,
    "smalltalk": _DAILY_LAYERS, "personal_state": _DAILY_LAYERS,
    "comfort": _DAILY_LAYERS, "story": _DAILY_LAYERS,
}
_LAYER_BOOST = 1.8   # 温和加权：足以把同分候选顶上去，不至于淹没词频强命中


def terms(query: str) -> list[str]:
    out: list[str] = []
    for tok in TOKEN.findall(query or ""):
        if re.match(r"[\u4e00-\u9fff]", tok):
            if len(tok) <= 4:
                out.append(tok)
            for n in (2, 3, 4):
                if len(tok) >= n:
                    for i in range(len(tok) - n + 1):
                        out.append(tok[i:i + n])
        else:
            out.append(tok.lower())
    seen = set()
    res = []
    for t in out:
        if t in STOP or len(t) < 2 or t in seen:
            continue
        seen.add(t)
        res.append(t)
    return res[:40]


class Corpus:
    def __init__(self, path: str) -> None:
        self.docs: list[dict[str, Any]] = []
        self._load(path)
        # 私有扩展层（2026-10-27 用户裁定）：粉丝材料/合作站点数据可用（检索+训练）但永不开源。
        # 与主池同目录的 corpus_private_ext.jsonl 存在即并入检索（source_layer=private_ext），
        # 开源打包脚本只读主池文件，天然不会带上。
        ext = os.path.join(os.path.dirname(os.path.abspath(path)), "corpus_private_ext.jsonl")
        if os.path.exists(ext) and os.path.realpath(ext) != os.path.realpath(path):
            self._load(ext, layer_override="private_ext")

    def _load(self, path: str, layer_override: str | None = None) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = (d.get("text") or "").strip()
                if not t:
                    continue
                self.docs.append({
                    "id": d.get("id"),
                    "text": t,
                    "source_type": d.get("source_type"),
                    "date": d.get("date"),
                    "source_layer": layer_override or d.get("source_layer"),
                    "use_policy": d.get("use_policy"),
                    "context": d.get("context"),
                })

    def search(self, query: str, limit: int = 5, mode: str = "") -> list[dict[str, Any]]:
        qs = terms(query)
        if not qs:
            return []
        layers = _MODE_BOOST.get(mode or "")
        scored = []
        for d in self.docs:
            text = d["text"]
            score = 0
            for t in qs:
                c = text.count(t)
                if c:
                    score += c * (2 if len(t) >= 4 else 1)
            if score:
                if layers and (d.get("context") in layers or d.get("source_type") in layers):
                    score = score * _LAYER_BOOST
                scored.append((score, d))
        scored.sort(key=lambda x: (-x[0], len(x[1]["text"])))
        return [dict(d, score=s) for s, d in scored[:limit]]

    def stats(self) -> dict[str, Any]:
        return {"docs": len(self.docs)}


# ── 模块级缓存（T4-M1 per-user Agent 实例化的前提）──
# 30MB corpus 解析实测 5-7s：per-user Agent 各自 new Corpus 会把这笔成本摊到每个用户
# 首条消息上。corpus 只读共享（search 纯读 docs 列表，多线程并发调用安全），
# 按 realpath 单例缓存——全进程只解析一次。Agent 一律经 load_corpus() 取实例。
_CORPUS_CACHE: dict[str, Corpus] = {}
_CORPUS_LOCK = threading.Lock()


def load_corpus(path: str) -> Corpus:
    key = str(os.path.realpath(path))
    with _CORPUS_LOCK:
        c = _CORPUS_CACHE.get(key)
        if c is None:
            c = Corpus(path)
            _CORPUS_CACHE[key] = c
        return c
