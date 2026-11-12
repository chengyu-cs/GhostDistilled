"""L4 Stage 3: assemble rewrite results into multi-turn dialogues + SFT files.

Parses l4_rewrite_raw.jsonl (classes A-D), applies the v2 cleaning rules per
assistant turn plus anti-echo/anti-ack checks, then expands every dialogue into
one training sample per assistant turn (prefix-increment style) so loss lands
on every reply while earlier turns serve as context.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPACT = """你是 AI小云，不是黄霄雲本人，不能自称黄霄雲或被当成本人。用第一人称、短句、口语；只聊公开作品/日常/心情；私生活/2020争议/未公开/健康不聊；每条回复第一行固定写 AI小云，只输出对话正文。"""

BAD = re.compile(
    r"2020|争议|淘汰|网暴|舆论|恶评|全网黑|奇袭|歌手|节目|导演|剧本|"
    r"男朋友|女朋友|恋爱|家庭|爸爸|妈妈|结婚|孩子|失眠|感冒|发炎|吃药|"
    r"病史|去世|离世|自杀|抑郁|歌词|代言|广告|带货|未公开|新歌预告"
)
TAG = re.compile(r"今天飘来一朵[雲云]|充电吧音乐人|微博演出季|打卡星能量|天赐的声音|微博|vlog|V！L！O！G！", re.I)
SELF = re.compile(r"我是黄霄[雲云]|我叫黄霄[雲云]|黄霄[雲云]")
# bare acknowledgement openers that made 14B v3 sound like "对/好的/我喜欢"
ACK = re.compile(r"^(对[，,啊呀哦]|好的[，,呀哦]|是的[，,呀哦]|嗯[嗯，,。]|行[，,啊])")
# campaign/promo post markers (her lively exclamation runs are kept; promo copy is not)
PROMO = re.compile(r"官宣|开票|购票|票务|大麦|猫眼|敬请期待|正式解锁|航线解锁|预约|周边上架")


def clean(s: str) -> str:
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"@\S+", "", s)
    s = re.sub(r"#([^#]{1,30})#", r"\1", s)
    s = TAG.sub("", s)
    s = re.sub(r"\[[^\]]{1,8}\]", "", s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"^[，,。.;；、\s]+|[，,。.;；、\s]+$", "", s)
    return s.strip()


def qa_ok(q: str, a: str, a_max: int = 110) -> bool:
    if not (4 <= len(q) <= 55):
        return False
    if not (12 <= len(a) <= a_max):
        return False
    if BAD.search(q) or BAD.search(a) or TAG.search(q) or TAG.search(a):
        return False
    if SELF.search(a):
        return False
    if re.sub(r"\s", "", a) == re.sub(r"\s", "", q) or a in q:
        return False
    if ACK.match(a) and len(a) < 16:
        return False
    if PROMO.search(a):
        return False
    if a.count("然后") > 2 or a.count("就是") > 2 or a.count("哈哈") > 2:
        return False
    # ASR gibberish guard: too many single-char runs inside her answer
    if len(re.findall(r"[\u4e00-\u9fa5]{20,}", a)) == 0 and "，" not in a and "。" not in a and len(a) > 45:
        return False
    return True


def parse_dialogues() -> list[dict]:
    dialogues = []
    for line in open(HERE / "l4_rewrite_raw.jsonl", encoding="utf-8"):
        d = json.loads(line)
        kind, resp, raw = d["kind"], d["resp"], d["raw"]
        item = d.get("item", {})
        if not isinstance(resp, dict) or resp.get("skip"):
            continue
        pairs: list[tuple[str, str]] = []
        st, sid, date = "", d["cid"], ""
        register = "spoken"
        if kind.startswith("A"):
            for ch in resp.get("chains", []) or []:
                ps = [(clean(str(p.get("q", ""))), clean(str(p.get("a", "")))) for p in ch.get("pairs", [])]
                ps = [x for x in ps if x[0] and x[1]]
                if len(ps) >= 2:
                    pairs = ps
                    st, sid = "interview_chain", d["cid"]
                    date = raw.get("date", "")
                    break
        elif kind.startswith("B"):
            posts = item.get("posts") or []
            chosen = resp.get("chosen") or []
            questions = resp.get("questions") or []
            if posts and chosen and questions and len(chosen) == len(questions):
                idxs = [c for c in chosen if isinstance(c, int) and 0 <= c < len(posts)]
                if len(idxs) == len(questions) and len(idxs) >= 2:
                    pairs = [(clean(questions[i]), clean(posts[c])) for i, c in enumerate(idxs)]
                    st = "weibo_chain"
                    date = raw.get("date", "")
                    register = "written"
        elif kind.startswith("C"):
            segs = item.get("segs") or []
            questions = resp.get("questions") or []
            if segs and questions and len(questions) == len(segs) and len(questions) >= 1:
                pairs = [(clean(q), clean(a)) for q, a in zip(questions, segs)]
                st = "concert_live" if raw.get("kind") == "concert" else "live_topic"
                date = ""
                register = "concert"
        elif kind.startswith("E"):
            a1, a2 = item.get("a1", ""), item.get("a2", "")
            questions = resp.get("questions") or []
            if a1 and a2 and len(questions) == 2:
                pairs = [(clean(questions[0]), clean(a1)), (clean(questions[1]), clean(a2))]
                st = "askback_chain"
                date = raw.get("date", "")
                register = raw.get("register", "written")
        elif kind.startswith("F"):
            a1, a2 = item.get("a1", ""), item.get("a2", "")
            questions = resp.get("questions") or []
            if a1 and a2 and len(questions) == 2:
                pairs = [(clean(questions[0]), clean(a1)), (clean(questions[1]), clean(a2))]
                st = "topic_switch"
                date = raw.get("date", "")
                register = "written"
        elif kind.startswith("D"):
            q1, a1 = item.get("q1", ""), item.get("a1", "")
            choose = resp.get("choose")
            pool = item.get("a2_pool") or []
            q2 = clean(str(resp.get("q2", "")))
            if (isinstance(choose, int) and 0 <= choose < len(pool) and q2
                    and q1 and a1):
                a2 = clean(pool[choose])
                pairs = [(clean(q1), clean(a1)), (q2, a2)]
                st = "interview_followup"
                date = raw.get("date", "")
        # per-pair validation: every pair must pass, and adjacent answers must differ
        # answer length cap differs by source: verbatim posts/talking run longer
        a_max = {"weibo_chain": 140, "concert_live": 180, "live_topic": 180}.get(st, 110)
        good = [p for p in pairs if qa_ok(p[0], p[1], a_max)]
        if len(good) < 2:
            continue
        kept = [good[0]]
        for p in good[1:]:
            if re.sub(r"\s", "", p[1])[:30] == re.sub(r"\s", "", kept[-1][1])[:30]:
                continue
            kept.append(p)
        if len(kept) < 2:
            continue
        dialogues.append({
            "source_type": st, "source_id": sid, "date": date,
            "register": register, "synthetic_user_turn": True,
            "persona": d.get("persona", ""),
            "pairs": kept,
        })
    return dialogues


import os

STAGE = os.environ.get("STAGE", "final")


def main() -> None:
    dialogues = parse_dialogues()
    # global dedup on the answer sequence
    seen, uniq = set(), []
    for dg in dialogues:
        key = "|".join(re.sub(r"\s", "", p[1]) for p in dg["pairs"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(dg)
    dialogues = uniq

    # stable ids: reuse ids from l4v2_base.jsonl when the answer sequence
    # matches, so spot-check verdicts stay aligned across reruns
    prev_path = HERE / "l4v2_base.jsonl"
    prev_ids: dict[str, str] = {}
    if prev_path.exists():
        for l in open(prev_path, encoding="utf-8"):
            d = json.loads(l)
            key = "|".join(re.sub(r"\s", "", p[1]) for p in zip(
                [m["content"] for m in d["messages"] if m["role"] == "user"],
                [m["content"] for m in d["messages"] if m["role"] == "assistant"]))
            prev_ids[key] = d["id"]
    nxt = 1
    for dg in dialogues:
        key = "|".join(re.sub(r"\s", "", p[1]) for p in dg["pairs"])
        if key in prev_ids:
            dg["id"] = prev_ids[key]
        else:
            while f"l4-{nxt:04d}" in prev_ids.values():
                nxt += 1
            dg["id"] = f"l4-{nxt:04d}"
            nxt += 1

    # full spot-check filter: drop dialogues flagged by l4_spotcheck.json
    spot_path = HERE / "l4v2_spotcheck.json" if STAGE == "final" else None
    if spot_path and spot_path.exists():
        spot = {o.get("id"): o for o in json.loads(spot_path.read_text(encoding="utf-8")) if isinstance(o, dict)}
        before = len(dialogues)
        kept = []
        for dg in dialogues:
            o = spot.get(dg["id"])
            if o and (any(o.get(k) is False for k in ("followup_ok", "voice_ok", "facts_ok")) or o.get("garbage")):
                continue
            kept.append(dg)
        print(f"spot-check filter: {before} -> {len(kept)} (dropped {before - len(kept)})")
        dialogues = kept

    # ask-back augmentation: append one of her real hook forms to final turns
    aug_path = HERE / "l4v2_askback_aug.json" if STAGE == "final" else None
    if aug_path and aug_path.exists():
        aug = json.loads(aug_path.read_text(encoding="utf-8"))
        n_aug = 0
        for dg in dialogues:
            hook = aug.get(dg["id"])
            if not hook:
                continue
            q, a = dg["pairs"][-1]
            if re.search(r"[？?]\\s*$", a):
                continue
            dg["pairs"][-1] = (q, (a + " " + hook).strip())
            dg["askback_hook"] = hook
            n_aug += 1
        print(f"ask-back hooks applied: {n_aug}")

    for i, dg in enumerate(dialogues, 1):
        dg["id"] = f"l4-{i:04d}"
        dg["messages"] = []
        for q, a in dg["pairs"]:
            dg["messages"].append({"role": "user", "content": q})
            dg["messages"].append({"role": "assistant", "content": a})
        dg["turns"] = len(dg["pairs"])

    OUTD = "l4v2_base.jsonl" if STAGE == "base" else "l4v2_dialogues.jsonl"
    with open(HERE / OUTD, "w", encoding="utf-8") as f:
        for dg in dialogues:
            f.write(json.dumps(dg, ensure_ascii=False) + "\n")

    # expand: one sample per assistant turn, earlier turns as context
    expanded = []
    for dg in dialogues:
        for k in range(len(dg["pairs"])):
            msgs = [{"role": "system", "content": COMPACT}]
            for j in range(k + 1):
                msgs.append({"role": "user", "content": dg["pairs"][j][0]})
                msgs.append({"role": "assistant", "content": "AI小云\n" + dg["pairs"][j][1]})
            # strip the final "AI小云\n"? No: v2 keeps the label inside the target;
            # tokenize() computes loss on the whole final assistant message.
            expanded.append({
                "id": f"{dg['id']}-t{k+1}", "messages": msgs,
                "source_type": dg["source_type"], "source_id": dg["source_id"],
                "date": dg["date"], "register": dg["register"],
                "turn_index": k + 1, "dialogue_id": dg["id"],
            })

    # split by dialogue group to avoid leakage between train/val
    dlg_ids = sorted({e["dialogue_id"] for e in expanded})
    val_ids = set(dlg_ids[::10])
    train = [e for e in expanded if e["dialogue_id"] not in val_ids]
    val = [e for e in expanded if e["dialogue_id"] in val_ids]

    train_name = "l4v2_train.jsonl" if STAGE == "final" else "l4v2_base_train.jsonl"
    val_name = "l4v2_val.jsonl" if STAGE == "final" else "l4v2_base_val.jsonl"
    for name, data in [(train_name, train), (val_name, val)]:
        with open(HERE / name, "w", encoding="utf-8") as f:
            for e in data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    if STAGE == "final":
        for name, data in [("l4v2_alibaba_train.jsonl", train), ("l4v2_alibaba_val.jsonl", val)]:
            with open(HERE / name, "w", encoding="utf-8") as f:
                for e in data:
                    f.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")

    turns = Counter(dg["turns"] for dg in dialogues)
    print("dialogues:", len(dialogues), "turn dist:", dict(sorted(turns.items())))
    print("expanded:", len(expanded), "train:", len(train), "val:", len(val))
    print("by source:", Counter(dg["source_type"] for dg in dialogues))
    alens = [len(a) for dg in dialogues for _, a in dg["pairs"]]
    print("answer len avg:", round(sum(alens) / len(alens), 1), "min:", min(alens), "max:", max(alens))
    qlens = [len(q) for dg in dialogues for q, _ in dg["pairs"]]
    print("question len avg:", round(sum(qlens) / len(qlens), 1))
    print("assistant asks back:", sum(1 for dg in dialogues for _, a in dg["pairs"] if "？" in a or "?" in a), "/", len(alens))
    q1s = [dg["pairs"][0][0][:8] for dg in dialogues]
    print("top q1 openers:", Counter(q1s).most_common(6))


if __name__ == "__main__":
    main()
