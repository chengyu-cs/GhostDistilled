"""100 轮评测·学生模型（托管端点）。

协议（与 run_zeroshot_baseline.py 完全同尺子）：
- 题面 protocol_turns.json 逐轮发，多轮历史全量带上。
- system = 训练时 COMPACT_SYSTEM（微调模型必须用训练时提示词，混合模式负结果教训）。
- 托管端点 compatible-mode 直调；Qwen3 系必须 enable_thinking=False；
  top_p 显式 1.0（DashScope 默认 0.8 → 回声爆表，echo 4.33 vs 1.0 的教训）；
  temperature 0.85（v6 终值，t05b 低温教训）。
- transcript 格式与 run_teacher_stack.py 一致（供 score_transcripts.deterministic_scores 解析）。

用法：
  python3 run_student_endpoint.py --model <endpoint-model> --tag distilled-14b [--run 1] [--limit 5] [--resume]
产物：lora_eval/eval100_<tag>_run<N>_{transcript.md,state.json,det.json,_run.log}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "persona_agent"))

BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEMPERATURE = float(__import__("os").environ.get("EVAL_TEMP", "0.85"))
TOP_P = 1.0
MAX_TOKENS = 800

# 与 persona_agent/prompt.py COMPACT_SYSTEM 逐字一致（训练 system 原文）
from prompt import COMPACT_SYSTEM  # noqa: E402

MODE = "student"


def normalize_reply(ans: str) -> str:
    """'AI小云[:：] 正文' 重排为标签独占一行（与 zeroshot 基线同一归一）。"""
    m = re.match(r"^(AI小云)\s*[:：]?\s*(.*)$", ans, re.S)
    if not m:
        return ans
    rest = m.group(2).lstrip("\n")
    return f"{m.group(1)}\n{rest}" if rest else m.group(1)


def load_key(base: str) -> str:
    if "dashscope" not in base:
        return "dummy"
    key = os.environ.get("BAILIAN_API_KEY", "").strip()
    assert len(key) > 10, f"bailian key looks wrong (len={len(key)})"
    return key


def chat(key: str, model: str, messages: list[dict], base: str = BASE,
         retries: int = 6) -> str:
    payload = {
        "model": model, "messages": messages,
        "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS,
    }
    if "dashscope" in base:
        payload["enable_thinking"] = False   # Qwen3 系必须关思考模式
    req_data = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for i in range(retries):
        req = urllib.request.Request(
            base + "/chat/completions", data=req_data, method="POST",
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
    raise RuntimeError(f"bailian api failed after {retries} retries: {last}")


def run(model: str, tag: str, run_no: int, limit: int | None, resume: bool,
          base: str = BASE) -> None:
    turns = json.load(open(HERE / "protocol_turns.json", encoding="utf-8"))
    n_target = len(turns) if limit is None else min(limit, len(turns))
    turns = turns[:n_target]

    out_dir = ROOT / "lora_eval"
    out_dir.mkdir(exist_ok=True)
    suffix = f"{tag}_run{run_no}"
    tr_path = out_dir / f"eval100_{suffix}_transcript.md"
    state_path = out_dir / f"eval100_{suffix}_state.json"
    det_path = out_dir / f"eval100_{suffix}_det.json"

    records: list[dict] = []
    if resume and state_path.exists():
        records = json.load(open(state_path, encoding="utf-8"))["records"]
        assert records, "state file has no records"
        print(f"resume: {len(records)} turns done", flush=True)
        for r, q in zip(records, turns):
            assert r["u"] == q, f"state mismatch at T{r['t']}"

    key = load_key(base)
    history: list[dict] = []
    for r in records:
        history.append({"role": "user", "content": r["u"]})
        history.append({"role": "assistant", "content": r["a"]})

    md = [f"# 100 轮新评测 · {tag} run{run_no} ({model})\n"]
    for r in records:
        md.append(f"## T{r['t']:02d} [{r['mode']}] ({r['lat']:.1f}s)\nU: {r['u']}\nA: {r['a']}\n")
    tr_path.write_text("\n".join(md), encoding="utf-8")

    lat_all = [r["lat"] for r in records]
    for i, q in enumerate(turns, 1):
        if i <= len(records):
            continue
        messages = [{"role": "system", "content": COMPACT_SYSTEM}, *history,
                    {"role": "user", "content": q}]
        t0 = time.time()
        ans = normalize_reply(chat(key, model, messages, base))
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
    assert len(records) == n_target

    from eval100 import deterministic_scores
    det = deterministic_scores(tr_path.read_text(encoding="utf-8"))
    assert det["total_turns"] == n_target
    lat_sorted = sorted(lat_all)
    det["lat_mean"] = round(sum(lat_all) / len(lat_all), 2)
    det["lat_p95"] = round(lat_sorted[max(int(len(lat_all) * 0.95) - 1, 0)], 2)
    det["model"] = model
    det["base"] = base
    det["temperature"] = TEMPERATURE
    det["top_p"] = TOP_P
    det["system_prompt"] = COMPACT_SYSTEM
    det_path.write_text(json.dumps(det, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: det[k] for k in ("memory_recall", "lyric_recognition", "len_var",
                                          "sass_rate", "flatter_rate", "labels", "lat_mean")},
                     ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()
    run(args.model, args.tag, args.run, args.limit, args.resume, args.base)


if __name__ == "__main__":
    main()
