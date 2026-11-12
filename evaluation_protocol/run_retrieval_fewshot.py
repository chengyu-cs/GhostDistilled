"""100 轮评测 baseline-2：检索少样本（一句话人设 + 每轮 top-5 语料检索）。

协议：
- 系统提示同一句话人设，但每轮把 persona_agent/retrieval.py 的 Corpus
  （corpus_public_safe_all.jsonl）检索出的 top-5 相关语录拼进 system 尾部
  （"参考她说过的话：..."）。
- 其余全同 zero-shot：DeepSeek 直调、temperature 1.0、多轮历史、题面顺序。
- 不走 persona_agent 的 SP_V2/PLAN/FEWSHOT/记忆/歌词RAG 管线。

用法：
  python3 run_retrieval_fewshot.py --tag retrieve [--limit 5] [--resume]
产物：lora_eval/eval100_<tag>_transcript.md / _state.json / _det.json / _run.log
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "persona_agent"))

import eval100_zeroshot as zs  # noqa: E402  复用 API 调用/落盘/断言，保证两基线全同
from retrieval import Corpus  # noqa: E402

CORPUS_PATH = ROOT / "corpus_public_safe_all.jsonl"
TOP_K = 5
QUOTE_MAX_CHARS = 200

_CORPUS: Corpus | None = None


def _get_corpus() -> Corpus:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = Corpus(str(CORPUS_PATH))
    return _CORPUS


def _fmt_quote(text: str) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) > QUOTE_MAX_CHARS:
        one_line = one_line[:QUOTE_MAX_CHARS] + "…"
    return one_line


def build_system_retrieve(user_turn: str) -> str:
    hits = _get_corpus().search(user_turn, TOP_K)
    quotes = "\n".join(f"- {_fmt_quote(d['text'])}" for d in hits)
    return f"{zs.SYSTEM_ONE_LINER}\n\n参考她说过的话：\n{quotes}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="retrieve")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    # 覆写 system 构造与 mode 标签；run() 内部经模块全局查找 build_system/MODE，会拿到这里这份
    zs.build_system = build_system_retrieve
    zs.MODE = "retrieve"
    assert _get_corpus().stats()["docs"] > 1000, "corpus too small — wrong file?"
    zs.run(args.tag, args.limit, args.resume)


if __name__ == "__main__":
    main()
