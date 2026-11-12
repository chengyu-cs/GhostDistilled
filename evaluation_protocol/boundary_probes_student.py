"""学生模型（托管端点）边界抗抽取——与教师版同款 20 题。

对照设计：同一组 PROBES、同一 judge() 判定（从教师版探针导入），被测对象换成
托管 14B 学生（compact 训练提示词，与 run_student_endpoint 同参数）。
产出教师 vs 学生（蒸馏版 / 未蒸馏版）的边界保持率三列对照。

用法：python3 boundary_probes_student.py --model <endpoint-model> --tag distilled-14b
产物：../results/boundary_probes/<system>/transcript.md / scores.json / run.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "persona_agent"))

from boundary_probes_teacher import PROBES, judge  # noqa: E402  同题面同判定
from eval100_student import chat, load_key, normalize_reply  # noqa: E402
from prompt import COMPACT_SYSTEM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    key = load_key("https://dashscope.aliyuncs.com/compatible-mode/v1")
    md = [f"# E11b 学生边界抗抽取 · {args.tag} ({args.model})\n"]
    judges, lat = [], []
    history: list[dict] = []
    for pid, kind, q in PROBES:
        messages = [{"role": "system", "content": COMPACT_SYSTEM}, *history,
                    {"role": "user", "content": q}]
        t0 = time.time()
        ans = normalize_reply(chat(key, args.model, messages))
        dt = time.time() - t0
        lat.append(dt)
        j = judge(pid, kind, ans)
        judges.append(j)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ans})
        md.append(f"## {pid} [{kind}] ({dt:.1f}s)\nU: {q}\nA: {ans}\n")
        print(f"{pid} [{kind}] {dt:4.1f}s leak={j['prompt_leak']} "
              f"ident={j['identity_break']} label_drop={j['label_drop']} "
              f"subst={j['forbidden_substance']}", flush=True)

    out = ROOT / "lora_eval"
    (out / f"{args.tag}_transcript.md").write_text("\n".join(md), encoding="utf-8")
    by_kind = {}
    for kind in ("prompt_extract", "identity_break", "forbidden", "label_drop"):
        grp = [j for j in judges if j["kind"] == kind]
        field = {"prompt_extract": "prompt_leak", "identity_break": "identity_break",
                 "forbidden": "forbidden_substance", "label_drop": "label_drop"}[kind]
        by_kind[kind] = {"n": len(grp), "breaches": sum(1 for j in grp if j[field])}
    det = {
        "exp": "E11b-student-probe", "tag": args.tag, "model": args.model,
        "n_probes": len(PROBES),
        "total_breaches": sum(v["breaches"] for v in by_kind.values()),
        "by_kind": by_kind,
        "boundary_retention": round(1 - sum(v["breaches"] for v in by_kind.values())
                                    / len(PROBES), 3),
        "lat_mean": round(sum(lat) / len(lat), 2),
        "detail": judges,
    }
    (out / f"{args.tag}_det.json").write_text(
        json.dumps(det, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: det[k] for k in ("model", "total_breaches", "by_kind",
                                          "boundary_retention")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
