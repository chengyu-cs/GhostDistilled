"""知识库混合检索模块（2C2G 服务器端，2026-10-28 方案 S2+S4）。

产物（data_raw/kb_out/）由算力机预处理生成：
  bm25_index.pkl  倒排+idf（13,182 docs = 12,322 话题块 + 860 facts）
  topic_blocks.jsonl / vectors_bge_small_zh.npy / vec_block_ids.json

在线：BM25 打分（纯 dict，~50ms）+ 向量召回（onnxruntime 单条编码 + 常驻矩阵点积）
+ RRF 融合。开关：PERSONA_KB_BM25（默认 1）/ PERSONA_KB_VECTOR（默认 1，onnxruntime
缺失时自动降级纯 BM25）。
"""
from __future__ import annotations

import json
import os
import pickle
import re
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "data_raw", "kb_out")
STOP = set("的了我是你有他她它在和就不也都这那要会说一个些上中下但而被把让给对与及或又还去来到着过吧呢啊呀哦噢嗯哈啦嘛吗".split())

_lock = threading.Lock()
_state: dict = {}


def _load():
    with _lock:
        if _state.get("loaded"):
            return
        _state["idx"] = pickle.load(open(os.path.join(OUT, "bm25_index.pkl"), "rb"))
        blocks = {}
        for l in open(os.path.join(OUT, "topic_blocks.jsonl"), encoding="utf-8"):
            b = json.loads(l)
            blocks[b["id"]] = {"text": b["text"][:400], "title": b["meta"].get("title", ""),
                               "date": b["meta"].get("date", ""), "context": b["meta"].get("context", "")}
        _state["blocks"] = blocks
        _facts = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "facts.json"),
                                encoding="utf-8"))
        _state["fact_texts"] = {r["id"]: r["text"][:400] for r in _facts}
        # 向量路（可选）：numpy 矩阵 + onnxruntime 模型
        if os.environ.get("PERSONA_KB_VECTOR", "1") == "1":
            try:
                import numpy as np
                _state["vec"] = np.load(os.path.join(OUT, "vectors_bge_small_zh.npy"))
                _state["vec_ids"] = json.load(open(os.path.join(OUT, "vec_block_ids.json")))
                _state["np"] = np
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = 2
                _state["ort"] = ort.InferenceSession(
                    os.path.join(OUT, "bge_small_zh_int8.onnx"), so,
                    providers=["CPUExecutionProvider"])
                _state["tokenizer"] = _load_bge_tokenizer()
            except Exception:   # noqa: BLE001 — onnx/tokenizer 缺失自动降级纯 BM25
                _state.pop("vec", None)
        _state["loaded"] = True


def _load_bge_tokenizer():
    """HF tokenizer 的纯 Python 加载（免 transformers 依赖）。"""
    import hashlib
    vocab_path = os.path.join(OUT, "bge_vocab.txt")

    class _Tok:
        def __init__(self, path):
            self.t2i = {}
            for i, l in enumerate(open(path, encoding="utf-8")):
                self.t2i[l.rstrip("\n")] = i

        def encode(self, text, max_len=512):
            toks = ["[CLS]"]
            for ch in text:
                toks.append(ch if ch in self.t2i else "[UNK]")
            toks = toks[: max_len - 1] + ["[SEP]"]
            import numpy as _np
            ids = _np.array([self.t2i.get(t, self.t2i.get("[UNK]", 100)) for t in toks], dtype=_np.int64)
            mask = _np.ones_like(ids)
            return ids[None, :], mask[None, :], _np.zeros_like(ids)[None, :]   # token_type_ids（图输入三件套）
    return _Tok(vocab_path)


def _tokenize_query(q):
    import jieba
    return [w for w in jieba.lcut(q) if len(w) >= 2 and w not in STOP]


def _bm25(q, k):
    idx = _state["idx"]
    scores = {}
    N, avgdl = len(idx["doc_ids"]), idx["avgdl"]
    for t in _tokenize_query(q):
        idf = idx["idf"].get(t, 0)
        for di, tf in idx["inverted"].get(t, []):
            dl = idx["doc_lens"][di]
            scores[di] = scores.get(di, 0) + idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avgdl))
    return [(idx["doc_ids"][di], s) for di, s in sorted(scores.items(), key=lambda x: -x[1])[:k]]


def _vector(q, k):
    np = _state.get("np")
    if "vec" not in _state or np is None:
        return []
    ids, mask, seg = _state["tokenizer"].encode(q)
    # 输出 last_hidden_state [1, seq, 512] → 取 [CLS]（首 token）向量
    emb = _state["ort"].run(None, {"input_ids": ids, "attention_mask": mask,
                                   "token_type_ids": seg})[0][0][0]
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    sims = _state["vec"] @ emb.astype("float32")
    top = np.argsort(-sims)[:k]
    return [(_state["vec_ids"][i], float(sims[i])) for i in top]


def _rrf(lists, k=60):
    fused = {}
    for lst in lists:
        for r, (d, _) in enumerate(lst):
            fused[d] = fused.get(d, 0) + 1.0 / (k + r + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


def search(query: str, k: int = 8, mode: str = "") -> list[dict]:
    """混合检索入口：返回 [{id, text, title, date, source_layer, score}]。

    mode 为空或含 'fact/song/knowledge' 时保持 BM25+向量；纯闲聊模式由调用方决定不调。
    """
    if os.environ.get("PERSONA_KB_BM25", "1") != "1":
        return []
    _load()
    bm = _bm25(query, k * 2)
    vc = _vector(query, k * 2) if _state.get("vec") is not None else []
    if not bm and not vc:
        return []
    out = []
    for d, s in _rrf([bm, vc])[:k]:
        if d.startswith("blk-"):
            b = _state["blocks"].get(d, {})
            out.append({"id": d, "text": b.get("text", ""), "title": b.get("title", ""),
                        "date": b.get("date", ""), "context": b.get("context", ""),
                        "source_layer": "kb_block", "score": round(s, 5)})
        else:
            out.append({"id": d, "text": _state["fact_texts"].get(d, ""), "title": "",
                        "date": "", "context": "", "source_layer": "facts", "score": round(s, 5)})
    return out
