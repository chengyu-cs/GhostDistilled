"""歌词 RAG（私有层）：歌曲话题轮检索背景，输出端防整段背词。

- 数据：private_data/lyrics_private.jsonl（106 首，仅本地，永不进开源库）
- 注入：[歌曲背景] 块 = 元数据 + 简介 + 主题意象 + 歌词全文（供理解），硬约束"引用最多两行"
- 防背词：回复含 ≥2 段 ≥10 字歌词连续串 → 判定过度引用，触发重生成
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIVATE = ROOT / "private_data" / "lyrics_private.jsonl"
SONGS = ROOT / "open_package" / "huangxiaoyun-public-corpus" / "data" / "songs.json"

_STORE: list[dict] | None = None


def _norm(s: str) -> str:
    # 2026-10-27 用户报障：英文歌名小写猜（reoracle）规范化后 != ReOracle——补 lower
    return re.sub(r"[，,。.·！!？?\s～~《》（）()：:;；'’]", "", s or "").lower()


def _load() -> list[dict]:
    global _STORE
    if _STORE is not None:
        return _STORE
    _STORE = []
    if PRIVATE.exists():
        meta = {}
        if SONGS.exists():
            for r in json.load(open(SONGS, encoding="utf-8")).get("entries", []):
                meta[_norm(r.get("title", ""))] = r
        # songs.json 标题常带"（先行单曲）/（专辑曲）"后缀，联查用包含式兜底
        def _meta_lookup(nm: str) -> dict:
            if nm in meta:
                return meta[nm]
            for k, v in meta.items():
                if nm and (nm in k or k in nm):
                    return v
            return {}
        for line in open(PRIVATE, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = _meta_lookup(_norm(d.get("title", "")))
            _STORE.append({
                "title": d.get("title", ""),
                "lyrics": d.get("lyrics", ""),
                "url": d.get("source_url", ""),
                "date": m.get("date", ""),
                "intro": m.get("intro", ""),
                "composer": m.get("composer", ""),
                "lyricist": m.get("lyricist", ""),
                "type": m.get("type", ""),
                "her": m.get("her_composition", False),
            })
    return _STORE


def detect_song(user_text: str) -> dict | None:
    """用户文本提到歌名（书名号或 ≥6 字歌词原句命中）。"""
    store = _load()
    if not store:
        return None
    for m in re.findall(r"《([^》]{1,15})》", user_text):
        nm = _norm(m)
        for s in store:
            if nm and (nm == _norm(s["title"]) or nm in _norm(s["title"]) or _norm(s["title"]) in nm):
                return s
    # 裸歌名+歌词语境（无书名号）："你最喜欢盲选的哪句歌词"类（2026-10-24 用户实测补）
    if re.search(r"歌词|哪句|那句|这段|哼一句|唱一段|最喜欢.{0,8}歌|成名曲|代表作", user_text):
        un = _norm(user_text)
        for s in store:
            t = _norm(s["title"])
            if len(t) >= 2 and t in un and t not in "AI小云":
                return s
    # 口语句式提取："那句X""这句X""X是什么歌/哪首"——把被指代的句子剥出来优先匹配
    m = re.search(r"(?:那|这)(?:句|段)(.{3,25}?)(?=是什么|是哪|哪首|出自|什么歌|$)", user_text)
    if m:
        quoted = _norm(m.group(1))
        if len(quoted) >= 4:
            for s in store:
                if quoted[:6] in _norm(s["lyrics"]) or quoted[-6:] in _norm(s["lyrics"]):
                    return s
    # 歌词原句命中：用户话语混着感想/标点/口误，整串匹配必败——滑窗片段匹配。
    clean = _norm(user_text)
    if len(clean) >= 6:
        ncd = [(_norm(s["title"]), _norm(s["lyrics"])) for s in store]
        for w in (10, 8, 6):
            step = max(1, w // 2)
            for i in range(0, max(1, len(clean) - w + 1), step):
                frag = clean[i:i + w]
                if len(frag) < w:
                    continue
                for j, (tn, ln) in enumerate(ncd):
                    if frag in ln:
                        return store[j]
    return None


def build_block(song: dict) -> str:
    role = "她自己写的" if song.get("her") in (True, "部分", "合作") else "她唱的"
    credits = []
    if song.get("composer"):
        credits.append(f"作曲 {song['composer']}")
    if song.get("lyricist"):
        credits.append(f"作词 {song['lyricist']}")
    head = f"[歌曲背景：《{song['title']}》{'（' + song['date'] + '）' if song.get('date') else ''}，{role}{('，' + '，'.join(credits)) if credits else ''}]"
    if song.get("intro"):
        head += f"\n官方简介：{song['intro'][:80]}"
    head += "\n（歌词全文如下，供你理解这首歌；回复里引用最多两行，禁止整段复述）\n" + song["lyrics"][:600]
    return head


def detect_song_quoted(text: str) -> dict | None:
    """文本中以书名号提到的库内歌曲——推荐场景反向检测用（2026-10-27 P1 盲区修复）。
    只认《》严格匹配：模型回复里裸词（如'你'）不会误绑《你》。"""
    store = _load()
    if not store:
        return None
    for m in re.findall(r"《([^》]{1,15})》", text):
        nm = _norm(m)
        for s in store:
            if nm and (nm == _norm(s["title"]) or nm in _norm(s["title"]) or _norm(s["title"]) in nm):
                return s
    return None


def is_song_title(text: str) -> bool:
    """文本（剥空白/书名号后）是否恰为库内某首歌名——猜歌局内短句意图判定快路用
    （2026-10-27 用户报障：猜『娃儿莫怕』被 LLM 判成 chat——真歌名猜错/猜对都不该走闲聊）。"""
    s = _norm(text)
    if not s or len(s) > 20:
        return False
    store = _load()
    return any(s == _norm(x["title"]) for x in store) if store else False


def quote_fidelity_check(reply: str, song: dict) -> bool:
    """回复疑似引用歌词原句（引号内 4-30 字）但不逐字存在于该歌歌词 → 编造，需重生成。"""
    ln = _norm(song["lyrics"])
    pairs = [("\u300c", "\u300d"), ("\u300e", "\u300f"), ("\u201c", "\u201d"), ('"', '"'), ("'", "'")]
    for lo, hi in pairs:
        for m in re.finditer(re.escape(lo) + "(.{4,30}?)" + re.escape(hi), reply):
            if _norm(m.group(1)) not in ln:
                return True
    return False


def overquote_check(reply: str, song: dict) -> bool:
    """≥2 段 ≥10 字的连续歌词串 → 过度引用。"""
    rn = _norm(reply)
    ln = _norm(song["lyrics"])
    hits, i = 0, 0
    while i < len(rn) - 9:
        if rn[i:i + 10] in ln:
            hits += 1
            i += 10
        else:
            i += 1
    return hits >= 2
