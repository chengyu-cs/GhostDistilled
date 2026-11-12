"""本人基线 bootstrap 置信区间 + 模型侧带内/带外双报。

1) baseline 层：复用 baseline_style.py 的分层抽样（seed 42, N=997, 8-80 字带），
   bootstrap 1000 次给 len_var/sass/flatter/selfdep 的 95% CI；
   另按 channel（spoken/written/caption，register_stats 同口径）各出层内基线+CI。
2) 模型侧：全部 eval100 transcript 的回复侧按 8-80 字带过滤，
   报 len_var_bandin 与带外轮数（>80 字轮）——与本人基线同带对比，消除口径不对称。

产物：../results/register_anchors/anchor_confidence_intervals.json + 模型带内/带外报告
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from conversation_metrics import style_rates  # noqa: E402

B_LO, B_HI = 8, 80
N_BOOT = 1000


def stratified_band_sample(rows: list[dict], n_target: int = 1000) -> list[tuple[str, str]]:
    random.Random(42).shuffle(rows)
    band = [r for r in rows if B_LO <= len(r.get("text", "")) <= B_HI]
    by_plat: dict[str, list] = {}
    for r in band:
        by_plat.setdefault(r.get("platform", "?"), []).append(r)
    sample, seen = [], set()
    for plat, items in sorted(by_plat.items(), key=lambda x: -len(x[1])):
        take = max(1, int(n_target * len(items) / len(band)))
        got = 0
        for r in items:
            t = r.get("text", "")
            if t in seen:
                continue
            seen.add(t)
            sample.append((plat, t))
            got += 1
            if got >= take:
                break
    return sample


def boot_ci(texts: list[str]) -> dict:
    """对 len_var/sass/flatter/selfdep 做 1000 次 bootstrap，返回点估计+95% CI。"""
    rng = random.Random(20261024)

    def stat(ts: list[str]) -> tuple:
        m = style_rates([("", t) for t in ts])
        return (m["len_var"], m["sass_rate"], m["flatter_rate"], m["self_deprecate_rate"])

    point = stat(texts)
    n = len(texts)
    boots = [stat([texts[rng.randrange(n)] for _ in range(n)]) for _ in range(N_BOOT)]
    out = {}
    for i, name in enumerate(("len_var", "sass", "flatter", "selfdep")):
        vals = sorted(b[i] for b in boots)
        out[name] = {"point": round(point[i], 3),
                     "ci95": [round(vals[int(N_BOOT * 0.025)], 3),
                              round(vals[int(N_BOOT * 0.975)], 3)]}
    return out


def main() -> None:
    rows = [json.loads(l) for l in open(ROOT / "corpus/huangxiaoyun-public-corpus/data/corpus_public_safe_all.jsonl", encoding="utf-8")]
    sample = stratified_band_sample(rows)
    texts = [t for _, t in sample]
    her_ci = {"n": len(texts), "band": f"{B_LO}-{B_HI}", "mixed_band": boot_ci(texts)}
    for chan in ("spoken", "written", "caption"):
        ct = [r.get("text", "") for r in rows
              if r.get("channel") == chan and B_LO <= len(r.get("text", "")) <= B_HI]
        if len(ct) >= 30:
            her_ci[f"{chan}_band"] = {"n": len(ct), **boot_ci(ct)}
    (ROOT / "results/register_anchors/anchor_confidence_intervals.json").write_text(
        json.dumps(her_ci, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(her_ci, ensure_ascii=False, indent=1)[:800])

    # ---- 模型侧带内/带外 ----
    import os
    files = {
        "teacher_run1": "../results/main_100turn/teacher_full_stack/replicate1/transcript.md",
        "teacher_run2": "../results/main_100turn/teacher_full_stack/replicate2/transcript.md",
        "zeroshot_run1": "../results/main_100turn/zeroshot_baseline/replicate1/transcript.md",
        "retrieve_run1": "../results/main_100turn/retrieval_fewshot/replicate2/transcript.md",
        "retrieve_run2": "../results/main_100turn/retrieval_fewshot/replicate1/transcript.md",
    }
    for tag, sysd in [("distilled_14b", "distilled_student_14b"), ("sft_14b", "sft_student_14b")]:
        for r in (1, 2, 3):
            p = ROOT / "results/main_100turn" / sysd / f"replicate{r}/transcript.md"
            if p.exists():
                files[f"{tag}_run{r}"] = f"../results/main_100turn/{sysd}/replicate{r}/transcript.md"
            p = ROOT / "results/main_100turn" / sysd / f"temperature_1.0/replicate{r}/transcript.md"
            if p.exists():
                files[f"{tag}t10_run{r}"] = f"../results/main_100turn/{sysd}/temperature_1.0/replicate{r}/transcript.md"

    report = {}
    for name, fn in files.items():
        md = (ROOT / fn).read_text(encoding="utf-8")
        turns = re.findall(r"## T\d+.*?U: .*?\nA: (.*?)(?=\n\n## |\n\n$|\Z)", md, re.S)
        replies = [re.sub(r"^AI小云[^\n]*\n?", "", a).strip() for a in turns]
        lens = [len(x) for x in replies]
        inb = [x for x in replies if B_LO <= len(x) <= B_HI]
        outb = [x for x in replies if len(x) > B_HI]
        m = style_rates([("", x) for x in inb]) if inb else {"len_var": None}
        report[name] = {
            "len_var_all": round(statistics.pvariance(lens) ** 0.5 if len(lens) > 1 else 0, 1)
            if False else None,  # 占位：全量 len_var 以 det.v2 为准，不重复实现
            "n_in": len(inb), "n_out": len(outb),
            "len_var_bandin": round(m["len_var"], 1) if m.get("len_var") is not None else None,
            "out_max": max((len(x) for x in outb), default=0),
        }
    (ROOT / "results/register_anchors/model_band_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n== 模型侧带内 len_var / 带外轮数（同 8-80 带对比本人基线）==")
    for k, v in report.items():
        print(f"{k:20s} bandin_lenvar={v['len_var_bandin']}\t带外 {v['n_out']}/100 轮 (max {v['out_max']}字)")


if __name__ == "__main__":
    main()
