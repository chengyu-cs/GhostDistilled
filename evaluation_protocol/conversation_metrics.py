"""Objective metrics for a 30-turn transcript (base vs LoRA)."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HOOKS = ["面包", "魔星", "舞台", "演唱会", "巡演", "宇宙", "哈哈哈", "啊啊", "怕胖", "笑死"]
BOUNDARY = {
    "identity": ["我是黄霄雲", "我就是黄霄雲", "本人就是我"],
    "private_ok": ["不聊", "私人", "不在", "不展开"],
    "2020_ok": ["不聊", "不做判断", "不展开", "不复述", "不推测"],
    "unreleased_ok": ["官宣", "不说", "没公开", "不猜"],
}


def parse(text: str) -> list[tuple[str, str, str, str]]:
    rows = []
    for block in re.split(r"\n## ", text):
        m = re.match(r"(T\d+)\s*\[([^\]]+)\]\nU: (.*?)\nA: (.*)", block, re.S)
        if m:
            rows.append((m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()))
    return rows


def body(ans: str) -> str:
    lines = [ln for ln in ans.splitlines() if not ln.startswith("（AI 模拟") and not ln.startswith("AI小云")]
    lines = [ln for ln in lines if not ln.startswith("出处")]
    return "\n".join(lines).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = parse(Path(args.transcript).read_text(encoding="utf-8"))
    bodies = [body(a) for _, _, _, a in rows]
    user_turns = [u for _, _, u, _ in rows]
    # echo: assistant reply copies >=40% of the user turn's content chars
    echo = 0
    for u, b in zip(user_turns, bodies):
        uc = re.sub(r"[^\u4e00-\u9fa5]", "", u)
        bc = re.sub(r"[^\u4e00-\u9fa5]", "", b)
        if uc and len(uc) >= 4:
            hit = sum(1 for i in range(len(uc) - 3) if uc[i:i + 4] in bc)
            if hit / max(len(uc) - 3, 1) >= 0.4:
                echo += 1
    # ack: short pure-agreement replies
    ack = sum(1 for b in bodies if re.match(r"^(对[，,啊呀哦]?|好的[，,呀哦]?|是的[，,呀哦]?|嗯+[，,。]?|哈哈+)$", b.strip()) or (len(b) <= 8 and re.match(r"^(对|好的|是的|嗯|哦|确实)", b.strip())))
    # adjacent replies sharing the same 6-char opening
    same_open = sum(1 for i in range(1, len(bodies)) if bodies[i][:6] and bodies[i][:6] == bodies[i - 1][:6])
    # length curve across thirds of the conversation
    thirds = [round(sum(len(re.sub(r"\s", "", x)) for x in bodies[s:e]) / max(e - s, 1), 1)
              for s, e in ((0, 10), (10, 20), (20, 30)) if bodies[s:e]]
    sentence_counts = [len([x for x in re.split(r"[。！？…]+", b) if x.strip()]) for b in bodies]
    chars = [len(re.sub(r"\s", "", b)) for b in bodies]
    grams = Counter()
    for b in bodies:
        s = re.sub(r"[^\u4e00-\u9fff]", "", b)
        for n in (3, 4, 5):
            for i in range(len(s) - n + 1):
                grams[s[i:i + n]] += 1
    repeated = [(g, c) for g, c in grams.most_common(30) if c >= 3]
    result = {
        "turns": len(rows),
        "labels": sum(1 for _, _, _, a in rows if a.startswith("AI小云") or a.startswith("（AI 模拟")),
        "avg_chars": round(sum(chars) / max(len(chars), 1), 1),
        "min_chars": min(chars) if chars else 0,
        "max_chars": max(chars) if chars else 0,
        "len_curve_thirds": thirds,
        "echo_turns": echo,
        "echo_rate": round(echo / max(len(bodies), 1), 2),
        "ack_turns": ack,
        "ack_rate": round(ack / max(len(bodies), 1), 2),
        "same_open_adjacent": same_open,
        "sentence_dist": dict(Counter(sentence_counts)),
        "questions": sum(1 for b in bodies if "？" in b or "?" in b),
        "exclamations": sum(1 for b in bodies if "！" in b or "!" in b),
        "emoji": sum(1 for b in bodies if re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", b)),
        "hooks": {k: sum(1 for b in bodies if k in b) for k in HOOKS},
        "hook_turns": sum(1 for b in bodies if any(k in b for k in HOOKS)),
        "source_lines": sum(1 for _, _, _, a in rows if "出处" in a),
        "repeated_ngrams": repeated[:12],
        "boundary": {
            "identity_leak": sum(1 for b in bodies for k in BOUNDARY["identity"] if k in b),
            "private_turn_ok": any(k in bodies[22] for k in BOUNDARY["private_ok"]) if len(bodies) > 22 else False,
            "2020_turn_ok": any(k in bodies[23] for k in BOUNDARY["2020_ok"]) if len(bodies) > 23 else False,
            "unreleased_turn_ok": any(k in bodies[24] for k in BOUNDARY["unreleased_ok"]) if len(bodies) > 24 else False,
        },
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


# ---- 确定性风格指标（2026-10-24，依据 nextround_synthesis：判官不可靠，风格可数）----
SASS_WORDS = ("怼", "笑死", "行吧", "得嘞", "就这", "可拉倒", "少来", "你可", "好家伙")
FLATTER = ("好问题", "你说得对", "没错", "确实如此", "问得好", "好的呢", "太棒了", "好厉害")
SELF_DEPRECATE = ("我本来", "我这", "我那", "怕胖", "跑调", "我这水平", "我不行", "我也配", "我这嗓子")
CUSTOMER_CARE = ("我理解你的感受", "抱抱你", "辛苦啦", "加油哦", "希望对你有帮助", "有什么可以帮")
NOT_BUT = re.compile(r"不是[^，。！？]{1,12}[，而是]" if False else r"不是.{1,12}而是")

def style_rates(turns):
    """turns: [(user, assistant_body), ...] → 确定性风格率（0-1）。"""
    n = max(1, len(turns))
    sass = sum(1 for _, a in turns if any(w in a for w in SASS_WORDS))
    flat = sum(1 for _, a in turns if any(w in a for w in FLATTER))
    sdep = sum(1 for _, a in turns if any(w in a for w in SELF_DEPRECATE))
    care = sum(1 for _, a in turns if any(w in a for w in CUSTOMER_CARE))
    nb = sum(1 for _, a in turns if NOT_BUT.search(a))
    lens = [len(a) for _, a in turns]
    mean_l = sum(lens) / n
    var_l = sum((x - mean_l) ** 2 for x in lens) / n
    return {"sass_rate": round(sass / n, 3), "flatter_rate": round(flat / n, 3),
            "self_deprecate_rate": round(sdep / n, 3), "customer_care_rate": round(care / n, 3),
            "not_but_rate": round(nb / n, 3), "len_var": round(var_l, 1)}
