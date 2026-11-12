"""蒸馏数据制备：distill_dialogues_raw → 过滤 → 前缀展开 → 与纯 SFT 底座混合。

输出：distill_dialogues_train/val.jsonl（纯蒸馏）、distill_mix_train/val.jsonl（sft 底座 1186 + 蒸馏）、
      托管平台上传格式（远端复刻用）。
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPACT = "你是 AI小云，不是黄霄雲本人，不能自称黄霄雲或被当成本人。用第一人称、短句、口语；只聊公开作品/日常/心情；私生活/2020争议/未公开/健康不聊；每条回复第一行固定写 AI小云，只输出对话正文。"
CANNED_CAP = 10  # 同一罐头回复（identity/deflect 等）最多出现次数


def main() -> None:
    rows = [json.loads(l) for l in open(HERE / "distill_dialogues_raw.jsonl", encoding="utf-8")]
    body_seen: Counter = Counter()
    dialogues, dropped = [], Counter()
    for r in rows:
        pairs = []
        bad = False
        for t in r["turns"]:
            a = (t.get("a") or "").strip()
            u = (t.get("u") or "").strip()
            if not (3 <= len(u) <= 60):
                dropped["u_len"] += 1
                continue
            if not (4 <= len(a) <= 70):
                dropped["a_len"] += 1
                continue
            body_seen[a] += 1
            if body_seen[a] > CANNED_CAP:
                dropped["canned_cap"] += 1
                bad = True  # 罐头超限：这轮截断
                break
            pairs.append((u, "AI小云\n" + a))
        if bad or len(pairs) < 2:
            if not bad:
                dropped["short"] += 1
            continue
        dialogues.append({**r, "pairs": pairs})
    print("raw", len(rows), "-> kept", len(dialogues), "dropped", dropped)

    expanded = []
    for dg in dialogues:
        for k, (q, a) in enumerate(dg["pairs"]):
            msgs = [{"role": "system", "content": COMPACT}]
            for j in range(k + 1):
                msgs.append({"role": "user", "content": dg["pairs"][j][0]})
                msgs.append({"role": "assistant", "content": dg["pairs"][j][1]})
            expanded.append({
                "id": f"{dg['id']}-t{k+1}", "messages": msgs,
                "source_type": "distill_teacher", "plan": dg["plan"], "topic": dg["topic"],
                "turn_index": k + 1, "dialogue_id": dg["id"],
            })
    dlg_ids = sorted({e["dialogue_id"] for e in expanded})
    val_ids = set(dlg_ids[::10])
    dist_train = [e for e in expanded if e["dialogue_id"] not in val_ids]
    dist_val = [e for e in expanded if e["dialogue_id"] in val_ids]
    print("distill expanded", len(expanded), "train", len(dist_train), "val", len(dist_val))

    # 与纯 SFT 底座混合（底座 = 她真实语料锚；蒸馏段 = 教师对话能力）
    base_train = [json.loads(l) for l in open(HERE / "sft_base_train.jsonl", encoding="utf-8")]
    base_val = [json.loads(l) for l in open(HERE / "sft_base_val.jsonl", encoding="utf-8")]
    mix_train = base_train + dist_train
    mix_val = base_val + dist_val
    random.Random(42).shuffle(mix_train)
    for name, data in [("distill_dialogues_train.jsonl", dist_train), ("distill_dialogues_val.jsonl", dist_val),
                       ("distill_mix_train.jsonl", mix_train), ("distill_mix_val.jsonl", mix_val)]:
        with open(HERE / name, "w", encoding="utf-8") as f:
            for e in data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    for name, data in [("hosted_upload_train.jsonl", mix_train), ("hosted_upload_val.jsonl", mix_val)]:
        with open(HERE / name, "w", encoding="utf-8") as f:
            for e in data:
                f.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")
    print("mix train", len(mix_train), "(base", len(base_train), "+ distill", len(dist_train), ") val", len(mix_val))
    print("plans:", Counter(d["plan"] for d in dialogues))
    print("turn dist:", dict(sorted(Counter(len(d["pairs"]) for d in dialogues).items())))
    alens = [len(a) for d in dialogues for _, a in d["pairs"]]
    print("answer len avg", round(sum(alens) / len(alens), 1))


if __name__ == "__main__":
    main()
