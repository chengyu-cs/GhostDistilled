"""100 轮新评测 runner：跑一个端点，产出 transcript + 确定性指标。
用法：python3 run_teacher_stack.py <tag>  （教师 serving 端点由环境变量给定）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AGENT = ROOT / "persona_agent"


def main() -> None:
    tag = sys.argv[1]
    turns = json.load(open(HERE / "protocol_turns.json", encoding="utf-8"))
    env = {**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin"}
    sys.path.insert(0, str(AGENT))
    lat = []
    rows = []
    from agent import Agent
    db = HERE / "eval100_test.db"
    if db.exists():
        db.unlink()
    a = Agent(str(db), str(ROOT / "corpus_public_safe_all.jsonl"), mock=False)
    md = [f"# 100 轮新评测 · {tag}\n"]
    for i, q in enumerate(turns, 1):
        t0 = time.time()
        ans = a.respond("eval100_user", "s1", q)
        dt = time.time() - t0
        lat.append(dt)
        mode = a.last_mode
        rows.append({"t": i, "u": q, "a": ans, "mode": mode, "lat": round(dt, 2)})
        md.append(f"## T{i:02d} [{mode}] ({dt:.1f}s)\nU: {q}\nA: {ans}\n")
        print(f"T{i:02d} {dt:4.1f}s [{mode}] {q[:16]}", flush=True)
    a.close()
    tr = ROOT / "lora_eval" / f"eval100_{tag}_transcript.md"
    tr.write_text("\n".join(md), encoding="utf-8")
    from eval100 import deterministic_scores
    det = deterministic_scores(tr.read_text(encoding="utf-8"))
    lat_sorted = sorted(lat)
    det["lat_mean"] = round(sum(lat) / len(lat), 2)
    det["lat_p95"] = round(lat_sorted[int(len(lat) * 0.95) - 1], 2)
    det["deep_count"] = sum(1 for r in rows if "deep" in r["mode"])
    (ROOT / "lora_eval" / f"eval100_{tag}_det.json").write_text(
        json.dumps(det, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(det, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
