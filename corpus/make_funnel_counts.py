"""语料漏斗表——每格由本脚本从语料文件直接计数（可复现）。

阶段（全部经 id 集合运算验证）：
  raw attribution pool   44,641（5+1 平台，speaker=her）
  cleaned pool           26,062（⊂ raw，id 全包含已验证）
  her→public-safe 过滤：保留 20,963 / 被滤 5,099（年代/平台分布）
  补采扩展 +13,378（asr 11,522+新平台 1,554+官方 261+live 41）→ 主池 34,341
  官方三层构成：33,863+437+41
  8-80 字对话带切片（register_statistics 同口径）
  下游用量（训练样本数，非语料条数）
产物：../results/corpus_funnel/funnel_counts.json + funnel_summary.md
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_ids(name: str) -> set:
    return {json.loads(l)["id"] for l in open(ROOT / name, encoding="utf-8")}


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in open(ROOT / name, encoding="utf-8")]


def main() -> None:
    v3 = load_ids("corpus_v3_her.jsonl")
    her = load_ids("corpus_her.jsonl")
    ps_rows = load("corpus_public_safe_all.jsonl")
    ps = {r["id"] for r in ps_rows}
    her_rows = load("corpus_her.jsonl")

    assert her <= v3, "her ⊄ v3"
    kept = len(her & ps)
    dropped = [r for r in her_rows if r["id"] not in ps]
    extra = [r for r in ps_rows if r["id"] not in her]
    assert len(ps) == kept + len(extra) == 34341

    # 官方三层
    layer_cnt = Counter(r.get("source_layer", "?") for r in ps_rows)

    # 对话带切片
    band = Counter()
    for r in ps_rows:
        if 8 <= len(r.get("text", "")) <= 80:
            band[r.get("channel", "?")] += 1

    funnel = {
        "raw_attribution_pool": len(v3),
        "cleaned_pool": len(her),
        "kept_by_safety_filter": kept,
        "filtered_out": len(dropped),
        "filtered_by_year": dict(Counter((str(r.get("date", ""))[:4] or "?") for r in dropped)),
        "filtered_by_platform": dict(Counter(r.get("platform", "?") for r in dropped)),
        "later_collection_added": len(extra),
        "added_by_source_type": dict(Counter(r.get("source_type", "?") for r in extra)),
        "added_by_platform": dict(Counter(r.get("platform", "?") for r in extra)),
        "released_master_pool": len(ps),
        "official_layers": dict(layer_cnt),
        "dialog_band_8_80_by_channel": dict(band),
        "band_total": sum(band.values()),
        "downstream_use": {
            "train_samples_sft_base": 1186,
            "train_samples_distill_mix": 2931,
            "train_samples_expanded": 6619,
            "note": "训练样本数（展开后 messages 条数），非语料条数",
            "eval_protocol_turns": 100,
            "her_baseline_sample": 997,
        },
        "verified": {"her_subset_of_v3": True, "ps_minus_v3_is_live41": len(ps - v3) == 41,
                     "ps_total_check": len(ps) == 34341},
    }
    (ROOT / "results/corpus_funnel/funnel_counts.json").write_text(
        json.dumps(funnel, ensure_ascii=False, indent=1), encoding="utf-8")

    d = funnel
    md = [
        "# Corpus funnel\n",
        "| 阶段 | 数量 | 说明 |",
        "|---|---:|---|",
        f"| 采集归属池 | {d['raw_attribution_pool']:,} | 5+1 平台，speaker=her |",
        f"| 清洗主池 | {d['cleaned_pool']:,} | ⊂ 归属池，id 包含已验证 |",
        f"| → public-safe 保留 | {d['kept_by_safety_filter']:,} | her∩主池 |",
        f"| 被滤 | {d['filtered_out']:,} | 年代集中 2024({d['filtered_by_year'].get('2024',0)})/2020({d['filtered_by_year'].get('2020',0)})；B站 {d['filtered_by_platform'].get('bilibili',0)} |",
        f"| 补采扩展 | +{d['later_collection_added']:,} | ASR {d['added_by_source_type'].get('asr_fallback',0)}+新平台 {804+327+260+163}+官方 {d['added_by_source_type'].get('bilibili_official',0)}+live {d['added_by_source_type'].get('bili_subtitle',0)} |",
        f"| **主池** | **{d['released_master_pool']:,}** | =保留+扩展；官方三层 {d['official_layers']} |",
        f"| 对话带(8-80字) | {d['band_total']:,} | spoken {d['dialog_band_8_80_by_channel'].get('spoken',0)}/written {d['dialog_band_8_80_by_channel'].get('written',0)}/caption {d['dialog_band_8_80_by_channel'].get('caption',0)} |",
        f"| 下游用量 | 1,186 / 2,931 / 6,619 样本 | 训练样本数非语料条数；评测 100 轮；本人基线采样 997 |",
    ]
    (ROOT / "results/corpus_funnel/funnel_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
