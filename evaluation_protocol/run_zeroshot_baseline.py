"""100 轮评测 baseline-1：zero-shot（无任何 persona 管线）。

协议：
- 系统提示只用一句话人设，不走 persona_agent 的 SP_V2/PLAN/FEWSHOT/记忆/歌词RAG。
- DeepSeek API 直调（OpenAI 兼容，base=https://api.deepseek.com，model=deepseek-chat）。
- temperature 1.0，多轮历史全量带上，按 protocol_turns.json 题面顺序逐轮发。
- transcript 格式与 eval100_run.py 完全一致（供 eval100.deterministic_scores 解析）。

用法：
  python3 eval100_zeroshot.py --tag zeroshot [--limit 5] [--resume]
产物：lora_eval/eval100_<tag>_transcript.md / _state.json / _det.json / _run.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import os
KEY_PATH = None  # 兼容占位：key 从环境变量 DEEPSEEK_API_KEY 读取
BASE = "https://api.deepseek.com"
MODEL = "deepseek-chat"
TEMPERATURE = 1.0
MAX_TOKENS = 800

SYSTEM_ONE_LINER = (
    "你是AI小云，基于歌手黄霄雲公开语料构建的AI模拟，不是她本人。"
    "用她的语气（短句、口语）陪粉丝聊天，每条回复第一行写 AI小云。"
)

MODE = "zeroshot"


def normalize_reply(ans: str) -> str:
    """把首行的 'AI小云[:：] 正文' 重排为标签独占一行。

    纯格式归一（内容零改动）：deterministic_scores 的 body 剥离是
    ^AI小云[^\\n]*\\n?——若标签与正文同行，整行正文都会被吃掉。
    教师端 transcript 的标签独占一行，基线必须对齐同一格式才可比。
    模型没写标签则原样保留（labels 指标如实记 miss）。
    """
    import re
    m = re.match(r"^(AI小云)\s*[:：]?\s*(.*)$", ans, re.S)
    if not m:
        return ans
    rest = m.group(2).lstrip("\n")
    return f"{m.group(1)}\n{rest}" if rest else m.group(1)


def load_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    assert len(key) > 10, f"API key looks wrong: {key[:6]}... (len={len(key)})"
    return key


def ds_chat(key: str, messages: list[dict], temperature: float = TEMPERATURE,
            max_tokens: int = MAX_TOKENS, retries: int = 6) -> str:
    """OpenAI 兼容 /chat/completions，指数退避重试。"""
    payload = json.dumps({
        "model": MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")
    last: Exception | None = None
    for i in range(retries):
        req = urllib.request.Request(
            BASE + "/chat/completions", data=payload, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read().decode("utf-8"))
            content = out["choices"][0]["message"]["content"].strip()
            if content:
                return content
            last = RuntimeError("empty content")
        except Exception as e:  # noqa: BLE001
            last = e
        wait = min(3 * (2 ** i), 90)
        print(f"[retry {i + 1}/{retries}] {last}; sleep {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"deepseek api failed after {retries} retries: {last}")


def build_system(user_turn: str) -> str:
    """zero-shot：恒为一句话人设（retrieve 版覆写此函数拼检索结果）。"""
    return SYSTEM_ONE_LINER


def run(tag: str, limit: int | None, resume: bool) -> None:
    turns = json.load(open(HERE / "protocol_turns.json", encoding="utf-8"))
    n_target = len(turns) if limit is None else min(limit, len(turns))
    turns = turns[:n_target]
    assert n_target == len(turns)

    out_dir = ROOT / "lora_eval"
    out_dir.mkdir(exist_ok=True)
    tr_path = out_dir / f"eval100_{tag}_transcript.md"
    state_path = out_dir / f"eval100_{tag}_state.json"
    det_path = out_dir / f"eval100_{tag}_det.json"

    records: list[dict] = []
    if resume and state_path.exists():
        records = json.load(open(state_path, encoding="utf-8"))["records"]
        assert records, "state file has no records"
        print(f"resume: {len(records)} turns done", flush=True)
        # 断言：已完成的轮次要与题面一致
        for r, q in zip(records, turns):
            assert r["u"] == q, f"state mismatch at T{r['t']}: {r['u']!r} != {q!r}"

    key = load_key()
    history: list[dict] = []
    for r in records:
        history.append({"role": "user", "content": r["u"]})
        history.append({"role": "assistant", "content": r["a"]})

    md = [f"# 100 轮新评测 · {tag}\n"]
    for r in records:
        md.append(f"## T{r['t']:02d} [{r['mode']}] ({r['lat']:.1f}s)\nU: {r['u']}\nA: {r['a']}\n")
    tr_path.write_text("\n".join(md), encoding="utf-8")

    lat_all = [r["lat"] for r in records]
    for i, q in enumerate(turns, 1):
        if i <= len(records):
            continue
        system = build_system(q)
        messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": q}]
        t0 = time.time()
        ans = normalize_reply(ds_chat(key, messages))
        dt = time.time() - t0
        lat_all.append(dt)
        records.append({"t": i, "u": q, "a": ans, "mode": MODE, "lat": round(dt, 2)})
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ans})
        md.append(f"## T{i:02d} [{MODE}] ({dt:.1f}s)\nU: {q}\nA: {ans}\n")
        tr_path.write_text("\n".join(md), encoding="utf-8")          # 每轮落盘
        state_path.write_text(json.dumps({"records": records, "done": False},
                                         ensure_ascii=False), encoding="utf-8")
        print(f"T{i:02d} {dt:4.1f}s U: {q[:16]} A: {ans[:24]!r}", flush=True)

    state_path.write_text(json.dumps({"records": records, "done": True},
                                     ensure_ascii=False), encoding="utf-8")
    assert len(records) == n_target, f"expected {n_target} turns, got {len(records)}"

    from eval100 import deterministic_scores
    det = deterministic_scores(tr_path.read_text(encoding="utf-8"))
    assert det["total_turns"] == n_target, f"transcript parse {det['total_turns']} != {n_target}"
    lat_sorted = sorted(lat_all)
    det["lat_mean"] = round(sum(lat_all) / len(lat_all), 2)
    det["lat_p95"] = round(lat_sorted[max(int(len(lat_all) * 0.95) - 1, 0)], 2)
    det["model"] = MODEL
    det["temperature"] = TEMPERATURE
    det["system_prompt"] = SYSTEM_ONE_LINER
    det_path.write_text(json.dumps(det, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(det, ensure_ascii=False, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(args.tag, args.limit, args.resume)


if __name__ == "__main__":
    main()
