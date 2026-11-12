"""L3 · 四层记忆（依据 memory_systems 调研，HaluMem 式 schema + 护栏）：

层：core(核心块：稳定偏好/身份) / fact(事件：带日期可过期) / episode(情节摘要) / relation(关系总述)
护栏：只抽显式陈述（正则）+ 每 10 轮 LLM 情节抽取带证据指针；不指望抽得准，靠留痕可改可删。
注入：Begin-Turn JIT 检索 top-4，以[背景事实]背景化注入（禁"我记得你说过"式汇报）。
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    # robustness 与 memory.py 对齐——多连接共享库必须开 WAL，否则写期间锁读（50 槽 Agent 共库）
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""CREATE TABLE IF NOT EXISTS mem_points(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, layer TEXT, mtype TEXT, content TEXT, owner TEXT DEFAULT 'user',
        importance REAL DEFAULT 0.5, status TEXT DEFAULT 'active',
        source TEXT, created_at TEXT, updated_at TEXT, prev_id INTEGER)""")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(mem_points)")]
    if cols and "owner" not in cols:
        conn.execute("ALTER TABLE mem_points ADD COLUMN owner TEXT DEFAULT 'user'")
    return conn


class Memory2:
    def __init__(self, path: str) -> None:
        self.conn = _connect(path)

    def close(self) -> None:
        self.conn.close()

    # ---- 写入 ----
    def add(self, user_id: str, layer: str, mtype: str, content: str,
            importance: float = 0.5, source: str = "") -> int:
        now = datetime.now().isoformat(timespec="seconds")
        cur = self.conn.execute(
            "INSERT INTO mem_points(user_id,layer,mtype,content,importance,status,source,created_at,updated_at)"
            " VALUES(?,?,?,?,?,'active',?,?,?)", (user_id, layer, mtype, content, importance, source, now, now))
        self.conn.commit()
        return cur.lastrowid

    def update(self, point_id: int, content: str) -> None:
        row = self.conn.execute("SELECT content FROM mem_points WHERE id=?", (point_id,)).fetchone()
        now = datetime.now().isoformat(timespec="seconds")
        if row:
            self.conn.execute("INSERT INTO mem_points(user_id,layer,mtype,content,importance,status,source,created_at,updated_at,prev_id)"
                              " SELECT user_id,layer,mtype,?,importance,'active',source,created_at,?,id FROM mem_points WHERE id=?",
                              (content, now, point_id))
            self.conn.execute("UPDATE mem_points SET status='superseded' WHERE id=?", (point_id,))
            self.conn.commit()

    def suppress(self, point_id: int) -> None:
        self.conn.execute("UPDATE mem_points SET status='suppressed' WHERE id=?", (point_id,))
        self.conn.commit()

    # ---- 检索（Begin-Turn JIT）----
    def retrieve(self, user_id: str, text: str, k: int = 4) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id,layer,mtype,content,importance,created_at,owner FROM mem_points"
            " WHERE user_id=? AND status='active' AND (owner IS NULL OR owner='user') ORDER BY id DESC LIMIT 200", (user_id,)).fetchall()
        if not rows:
            return []
        tg = set(re.findall(r"[\u4e00-\u9fff]{2}", text))

        def score(r):
            cg = set(re.findall(r"[\u4e00-\u9fff]{2}", r[3]))
            overlap = len(tg & cg) * 2 if cg else 0
            imp = float(r[4]) * 3
            try:
                age_h = (datetime.now() - datetime.fromisoformat(r[5])).total_seconds() / 3600
            except ValueError:
                age_h = 0
            recency = max(0.0, 1.5 - age_h / 168)  # 一周内轻微加权
            layer_bonus = 1.5 if r[1] == "core" else (1.0 if r[1] == "fact" else 0.5)
            return overlap + imp + recency + layer_bonus

        scored = sorted(rows, key=score, reverse=True)
        return [{"id": r[0], "layer": r[1], "mtype": r[2], "content": r[3]} for r in scored[:k]]

    def panel(self, user_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id,layer,mtype,content,status FROM mem_points WHERE user_id=? ORDER BY id DESC LIMIT 50",
            (user_id,)).fetchall()
        return [{"id": r[0], "layer": r[1], "type": r[2], "content": r[3], "status": r[4]} for r in rows]

    # ---- 显式陈述抽取（End-Turn，正则护栏：只收明说的）----
    EXPLICIT = [
        (re.compile(r"我(?:最喜欢|超喜欢|特别喜欢)(?:吃|听|看|的)(?:是)?([\u4e00-\u9fffA-Za-z0-9]{1,12})"), "core", "preference", 0.7),
        (re.compile(r"我(?:不喜欢|讨厌|不吃)([\u4e00-\u9fffA-Za-z0-9]{1,12})"), "core", "preference", 0.6),
        (re.compile(r"我(?:是个?|在做|当)(学生|上班族|老师|医生|程序员|设计师|高中生|大学生|研究生|宝妈|自由职业)"), "core", "persona", 0.8),
        (re.compile(r"我(?:最近)?(?:在|正在)(学|备考|准备|练)([\u4e00-\u9fffA-Za-z0-9]{2,12})"), "fact", "ongoing", 0.8),
        (re.compile(r"我(?:考过了|上岸了|过啦|成功了|拿证了)"), "fact", "milestone", 0.9),
        (re.compile(r"我(?:家)?(?:的)?(?:猫|狗|仓鼠|兔子|鸟)(?:叫|名字是)([\u4e00-\u9fff]{1,4})"), "core", "pet_name", 0.7),
        (re.compile(r"我(?:养了|家有)(猫|狗|仓鼠|兔子|鸟)"), "core", "preference", 0.6),
    ]

    def extract_explicit(self, user_id: str, text: str, turn: int) -> int:
        n = 0
        for rx, layer, mtype, imp in self.EXPLICIT:
            m = rx.search(text)
            if not m:
                continue
            content = (m.group(0).replace("我", "用户", 1))[:40]
            dup = self.conn.execute(
                "SELECT id FROM mem_points WHERE user_id=? AND content=? AND status='active'",
                (user_id, content)).fetchone()
            if dup:
                continue
            self.add(user_id, layer, mtype, content, imp, f"turn{turn}")
            n += 1
        return n

    def ingest_episode_points(self, user_id: str, points: list[dict], turn: int) -> int:
        """LLM 情节抽取结果入库（带护栏：限条数、限长度）。"""
        n = 0
        for pt in points[:4]:
            c = str(pt.get("content", ""))[:60]
            if len(c) < 6:
                continue
            layer = pt.get("layer", "episode") if pt.get("layer") in ("core", "fact", "episode", "relation") else "episode"
            self.add(user_id, layer, pt.get("type", "episode"), "用户" + c.lstrip("用户") if not c.startswith("用户") else c,
                     float(pt.get("importance", 0.5)), f"turn{turn}:llm")
            n += 1
        return n
