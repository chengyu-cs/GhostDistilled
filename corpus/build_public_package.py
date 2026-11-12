"""构建开源语料包：黄霄雲公开语料库（粉丝整理·机器可读）。

内容决策（2026-10-23，与用户确认的撇清原则）：
- 只放"她本人"的公开安全文本（corpus_public_safe_all 34,341 条）
- 排除 timeline.json/entities.json（含 17b331 年表二手内容，DATA_PROVENANCE.md 记录在案）
- 排除 corpus_context.jsonl（859 条第三方粉丝评论，非她本人内容）
- 排除本项目 AI 产物（训练数据/评测/人格卡）
- B站条目从 id 推导原视频链接；保留全部原始字段
"""
from __future__ import annotations

import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PKG = ROOT / "open_package" / "huangxiaoyun-public-corpus"
DATA = PKG / "data"
INDEX = PKG / "index"

BV_RE = re.compile(r"^bili-(BV[0-9A-Za-z]{8,12})-\d+$")


def enrich(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        m = BV_RE.match(r.get("id", ""))
        if m and not r.get("source_url"):
            r["source_url"] = f"https://www.bilibili.com/video/{m.group(1)}"
        out.append(r)
    return out


def main() -> None:
    if PKG.exists():
        shutil.rmtree(PKG)
    DATA.mkdir(parents=True)
    INDEX.mkdir(parents=True)

    pool = [json.loads(l) for l in open(ROOT / "corpus/huangxiaoyun-public-corpus/data/corpus_public_safe_all.jsonl", encoding="utf-8")]
    pool = enrich(pool)
    with open(DATA / "corpus_public_safe_all.jsonl", "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    weibo = [r for r in pool if r.get("platform") == "weibo"]
    with open(DATA / "weibo_public_safe.jsonl", "w", encoding="utf-8") as f:
        for r in weibo:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    concert = [json.loads(l) for l in open(ROOT / "concert_talking.jsonl", encoding="utf-8")]
    with open(DATA / "concert_talking.jsonl", "w", encoding="utf-8") as f:
        for r in concert:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 视频索引（考古导航）
    videos: dict[str, dict] = {}
    for r in pool:
        m = BV_RE.match(r.get("id", ""))
        if not m:
            continue
        bv = m.group(1)
        v = videos.setdefault(bv, {"bv": bv, "url": r["source_url"], "title": r.get("title", ""),
                                   "date": r.get("date", ""), "segments": 0, "chars": 0})
        v["segments"] += 1
        v["chars"] += len(r.get("text", ""))
    with open(INDEX / "by_video.json", "w", encoding="utf-8") as f:
        json.dump(sorted(videos.values(), key=lambda x: -x["segments"]), f, ensure_ascii=False, indent=1)

    months = Counter((r.get("date") or "")[:7] for r in pool if r.get("date"))
    with open(INDEX / "by_month.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in sorted(months.items())}, f, ensure_ascii=False, indent=1)

    # 统计
    zh = sum(1 for r in pool for c in (r.get("text") or "") if "\u4e00" <= c <= "\u9fff")
    stats = [
        f"- 总条目：{len(pool):,}（全部为她本人声音，公开安全过滤后）",
        f"- 汉字总量：{zh:,}",
        f"- 时间范围：{min((r['date'] for r in pool if r.get('date')), default='')} → {max((r['date'] for r in pool if r.get('date')), default='')}",
        f"- 平台分布：{dict(Counter(r.get('platform','?') for r in pool).most_common())}",
        f"- 视频数（B站）：{len(videos)}",
        f"- 使用策略字段：{dict(Counter(r.get('use_policy','?') for r in pool).most_common())}",
    ]
    (PKG / "docs").mkdir()
    (PKG / "docs" / "stats.md").write_text("# 语料统计\n\n" + "\n".join(stats) + "\n", encoding="utf-8")

    fields = {
        "id": "全局唯一 ID：weibo-{微博mid} / bili-{BV号}-{段序} / 其他平台同理",
        "platform": "weibo / bilibili / douyin / wechat / xhs / concert",
        "source_url": "原始公开链接（B站链接由 BV 号推导，标注于 README）",
        "source_type": "采集通道类型",
        "source_owner": "发言者归属（本池全部为 her）",
        "date": "发布/上传日期（YYYY-MM-DD 或空）",
        "title": "视频/微博标题",
        "text": "正文文本（字幕段/微博正文/现场讲话）",
        "verified / verification": "来源核验标记",
        "language / tags / metrics / images / has_video": "元信息",
        "sensitive_period": "敏感期标记（已过滤，本池仅保留公开安全条目）",
        "speaker / speaker_confidence": "B站字幕说话人归属及置信度（≥0.72 才入池）",
        "is_compilation / source_layer / channel": "分层与合集标记",
        "use_policy": "使用策略：public_safe_generation（公开安全生成）/ generation_allowed_paraphrase_only（仅限转述，41 条 2018 艺考剪辑）",
        "checksum": "文本校验和",
    }
    (PKG / "docs" / "fields.md").write_text(
        "# 字段说明\n\n" + "\n".join(f"- `{k}`：{v}" for k, v in fields.items()) + "\n", encoding="utf-8")

    lic = """本语料库采用 CC BY-NC-SA 4.0（署名-非商业性使用-相同方式共享）。

附加条款：
1. 仅供个人研究、粉丝社区非商业用途；禁止任何商业用途。
2. 全部内容采集自黄霄雲女士及其团队的公开发布（微博/哔哩哔哩/抖音/公众号/小红书/演唱会现场），著作权归原作者所有；本库仅为粉丝整理的机器可读索引。
3. 不含任何歌词原文；不含私生活、健康、争议相关内容（入池前已过滤）。
4. 若艺人本人、其团队或官方提出要求，将无条件下架或调整。
"""
    (PKG / "LICENSE").write_text(lic, encoding="utf-8")

    print("package built:", PKG)
    print("data files:", [p.name for p in DATA.iterdir()])
    print("index files:", [p.name for p in INDEX.iterdir()])


if __name__ == "__main__":
    main()
