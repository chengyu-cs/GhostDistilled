"""台前/后台语域统计表（corpus characterization，不承诺训练对照）。

口径与 baseline_style.py 完全一致（conversation_metrics.style_rates，8-80 字对话带）。

主表：channel（spoken/written/caption）× 年代切片（2014-19 学员期 / 2020-21 沉寂期
      / 2022-26 回归期）→ N / len_var / avg_len / sass / flatter / selfdep / care
副表：过滤效应——corpus_her（她全量）vs public_safe_all 中同 id 交集，被过滤子集的
      年代/平台/风格特征（"后台内容被 public-safe 层挡掉"的直接证据）

产物：../results/register_anchors/register_statistics.json + register_statistics.md（人读表）
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from conversation_metrics import style_rates  # noqa: E402

ERAS = [("2014-19", "2014", "2019"), ("2020-21", "2020", "2021"), ("2022-26", "2022", "2026")]


def era_of(date: str) -> str | None:
    y = (date or "")[:4]
    if not y.isdigit():
        return None
    for name, lo, hi in ERAS:
        if lo <= y <= hi:
            return name
    return None


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def band_texts(rows: list[dict]) -> list[str]:
    return [r.get("text", "") for r in rows if 8 <= len(r.get("text", "")) <= 80]


def stat_cell(rows: list[dict]) -> dict | None:
    texts = band_texts(rows)
    if len(texts) < 30:
        return None
    m = style_rates([("", t) for t in texts])
    lens = [len(t) for t in texts]
    return {
        "n_band": len(texts), "n_all": len(rows),
        "avg_len": round(sum(lens) / len(lens), 1),
        "len_var": m.get("len_var"),
        "sass": m.get("sass_rate"), "flatter": m.get("flatter_rate"),
        "selfdep": m.get("self_deprecate_rate"), "care": m.get("customer_care_rate"),
    }


def main() -> None:
    ps = load(str(ROOT / "corpus/huangxiaoyun-public-corpus/data/corpus_public_safe_all.jsonl"))
    her = load(str(ROOT / "corpus_her.jsonl"))

    # ---- 主表：channel × era ----
    main_table = {}
    for chan in ("spoken", "written", "caption"):
        rows_c = [r for r in ps if r.get("channel") == chan]
        main_table[f"{chan}_ALL"] = stat_cell(rows_c)
        for era, *_ in ERAS:
            main_table[f"{chan}_{era}"] = stat_cell(
                [r for r in rows_c if era_of(str(r.get("date", ""))) == era])

    # ---- 副表：过滤效应（her 全量 vs public-safe 交集 vs 被过滤差集）----
    her_ids = {r["id"] for r in her}
    ps_ids = {r["id"] for r in ps}
    kept = [r for r in ps if r["id"] in her_ids]              # 同源进 public-safe 的
    dropped = [r for r in her if r["id"] not in ps_ids]       # her 有但被过滤的
    flt = {
        "her_total": len(her), "public_safe_total": len(ps),
        "her_in_public_safe": len([r for r in her if r["id"] in ps_ids]),
        "her_dropped": len(dropped),
        "dropped_years": dict(Counter((str(r.get("date", ""))[:4] or "?") for r in dropped)),
        "dropped_platforms": dict(Counter(r.get("platform", "?") for r in dropped)),
        "dropped_style": stat_cell(dropped),
        "kept_style": stat_cell([r for r in kept if r.get("channel") == "spoken"]),
    }

    out = {"main_table": main_table, "filter_effect": flt}
    (ROOT / "results/register_anchors/register_statistics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# Register statistics: channel × era (8-80 char band)\n",
          "| 切片 | N(band/all) | avg_len | len_var | sass | flatter | selfdep | care |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in main_table.items():
        if v:
            md.append(f"| {k} | {v['n_band']}/{v['n_all']} | {v['avg_len']} | {v['len_var']} "
                      f"| {v['sass']} | {v['flatter']} | {v['selfdep']} | {v['care']} |")
    md.append(f"\n## 过滤效应：her {flt['her_total']} 条 → 进 public-safe {flt['her_in_public_safe']}"
              f"，被滤 {flt['her_dropped']} 条")
    md.append(f"- 被滤年份分布: {flt['dropped_years']}")
    md.append(f"- 被滤平台分布: {flt['dropped_platforms']}")
    if flt["dropped_style"]:
        md.append(f"- 被滤子集风格: {flt['dropped_style']}")
    (ROOT / "results/register_anchors/register_statistics.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
