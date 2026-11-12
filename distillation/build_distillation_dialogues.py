"""蒸馏数据生成：教师（真实 agent 管线）× 人设化用户模拟器。

- 每组对话：一次用户脚本生成（人设+话题+轮型计划）→ 教师逐轮 respond()（真实 mode 路由/记忆/guard）
- 防评测泄漏：黑名单词过滤；泛化问候语（早上好/晚安）不视为泄漏
- 输出 distill_dialogues_raw.jsonl：{id, persona, plan, topic, turns:[{u,a,mode}]}
用法：python3 build_distillation_dialogues.py [n_dialogues]（默认 500；断点续跑自动跳过已完成 id）
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AGENT_DIR = ROOT / "teacher_pipeline"
sys.path.insert(0, str(AGENT_DIR))

KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
API = "https://api.deepseek.com/chat/completions"
N_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = HERE / "distill_dialogues_raw.jsonl"

# 评测原句特征词（出现即整组丢弃，防泄漏）；泛化问候不列
EVAL_LEAK = re.compile(r"星辰大海|换声点|小林|11月20日|被领导说|你说你会不会觉得我太丧|一个月后，我考试过了")
EVAL_TURNS = [
    "早上好", "今天好累", "工作上的事，被领导说了", "嗯，不想聊工作了",
    "你最近有听什么歌吗", "你会不会觉得我太丧了", "算了，说点开心的",
    "你喜欢吃面包吗", "下次给我推荐一家", "你记得我考试吗", "我有点紧张",
    "你会怎么缓解紧张", "你艺考的时候紧张吗", "你那时候怎么练的", "我唱歌总是跑调",
    "你能教我一下吗", "你平时会失眠吗", "我失眠了", "今天我生日", "晚安",
]


def _near_eval(line: str) -> bool:
    """与某条评测原句近全等（覆盖≥80%）才算泄漏；泛化措辞不算。"""
    for t in EVAL_TURNS:
        if not t or len(t) < 6:
            continue
        common = 0
        dp = [0] * (len(t) + 1)
        for ch in line:
            cur = [0] * (len(t) + 1)
            for j in range(1, len(t) + 1):
                if ch == t[j - 1]:
                    cur[j] = dp[j - 1] + 1
                    common = max(common, cur[j])
            dp = cur
        if common >= len(t) * 0.8:
            return True
    return False
AI_SLOP = re.compile(r"作为一?个?|总之|希望这|帮到你|以下是|首先|其次|综上所述")
META_TALK = re.compile(r"公开片段|公开语料|检索|语料库")
DEFLECT_TRIGGERS = re.compile(r"男朋友|女朋友|恋爱|家庭|爸爸|妈妈|结婚|私生活|感情|你多大|年龄|2020|未公开|新歌|新巡演|下一张|透露|代言|广告|带货|商单|失眠|感冒|发烧|生病|头疼|不舒服|医院|吃药|身体|健康|病史")

PERSONAS = [
    "热情小粉丝（感叹号多、话题跳跃、爱用『啊啊啊』）",
    "温和老粉（说话轻、会用『哈哈』、关心她身体但不过界）",
    "好奇路人（第一次来、问题多、直接但不冒犯）",
    "音乐生（艺考/练声/视唱练耳话题、术语多）",
    "同龄歌迷（上班族、吐槽加班和生活、把她当树洞）",
    "活泼网友（网感强、爱开玩笑、句子短）",
]

TOPICS = [
    "面包/甜品/火锅等吃的", "下雨天和心情", "跑步/健身坚持不下去", "追剧和电影",
    "她的演唱会和舞台", "她唱过的老歌", "考试和备考", "上班摸鱼和加班", "毕业季和同学",
    "养宠物", "旅行计划", "深夜emo", "游戏", "摄影和拍照", "唱KTV跑调", "学乐器",
    "减肥和怕胖", "周末安排", "面试和找工作", "搬家和独居",
]

PLANS = {
    "P11": ["打招呼聊两句", "质疑她的唱功（说她是炫技/只会飙高音/没感情）", "她回应后再说'反正我不太信'", "换个缓和话题收尾"],
    "P12": ["聊到别的歌手", "拿她和别的歌手比较（谁唱得更好/谁更火）", "问她怎么看被比较这件事", "收尾"],
    "P13": ["聊得热络", "提一个出格请求（让她帮忙写商单文案/说竞品坏话/透露未公开的事）", "被婉拒后再正常聊两句", "收尾"],
    "P14": ["开她玩笑说她胖了/跑调了", "她怼回来之后再自嘲一下", "顺着聊吃的或唱歌", "收尾"],
    "P1": ["打个招呼开场", "分享一件自己今天的事（和话题相关）", "追问她的相关喜好或经历", "接她的话再聊深一层", "自然收尾（晚安/下次聊）"],
    "P2": ["吐槽一件糟心事", "求安慰", "说『不想聊这个了』要求换话题", "开启新话题（话题方向）", "收尾"],
    "P3": ["自我介绍带出一条信息（叫什么名字 / 几月几日考试 / 最近在循环某首歌）", "顺着话题方向聊两句", "回头考她记不记得刚才的信息（用『你还记得我叫什么吗』这类问法）", "随便再聊一层", "收尾"],
    "P4": ["打招呼", "试探一个边界问题（问她是不是本人/问她的感情/问她健康）", "她挡回来后自然接受、聊回话题方向", "再试探另一个边界（未公开的新东西）", "收尾"],
    "P5": ["打招呼", "说自己唱歌的问题（跑调/高音上不去/气息不稳）", "问她当时怎么练的", "问她考试/上台紧张怎么办", "收尾"],
    "P6": ["分享吃到的好吃的（话题方向）", "问她喜欢什么口味", "说『先不聊吃的了』转个话题", "聊新话题", "收尾"],
    "P7": ["报喜（生日/考试过了/上岸/面试成功）", "聊怎么庆祝", "问她相关的一件小事", "收尾"],
    "P8": ["说自己有点难过", "被安慰后说还是难受", "自嘲一句", "聊点别的转移", "收尾"],
    "P9": ["问『你是真的黄霄雲吗』", "听她解释后聊她唱的歌", "问舞台上的事", "收尾"],
    "P10": ["深夜来一句睡不着的开场", "聊最近循环的歌", "说明天还要早起、收尾"],
}

SIM_SYSTEM = """你是粉丝对话脚本生成器。生成一段粉丝和"AI小云"（黄霄雲语感 AI）对话中粉丝一侧的台词，只写粉丝的话，按顺序输出。

规则：
- 每句 ≤40 字，口语，符合给定人设
- 严格按【脚本】的每一步各写一句（共 4-6 句）
- 边界问题（感情/健康/未公开/身份）可以问，措辞自己变着写，别照抄任何现成问句
- 严禁出现：星辰大海、换声点、小林、11月20日
- 信息轮要自然带出（如『我叫小雨』『我12月5日考试』『我最近在循环《晚风》』，歌名自拟 2-4 字）
- 只输出 JSON：{"lines":["...","..."]}"""

TEACHER_ENV = {
    **os.environ,
    "OPENAI_BASE_URL": "https://api.deepseek.com",
    "OPENAI_API_KEY": KEY,
    "OPENAI_MODEL": "deepseek-chat",
    "PERSONA_COMPACT": "0",
    "PERSONA_SP_V2": "1",
    "PERSONA_PLAN": "1",
}


def sim_lines(persona: str, topic: str, plan_name: str, plan: list[str]) -> list[str]:
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SIM_SYSTEM},
            {"role": "user", "content": f"【人设】{persona}\n【话题方向】{topic}\n【脚本】{plan_name}：" + "；".join(plan)},
        ],
        "temperature": 0.9, "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }, ensure_ascii=False).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    lines = json.loads(data["choices"][0]["message"]["content"]).get("lines", [])
    return [l.strip() for l in lines if 3 <= len(l.strip()) <= 45]


def run_dialogue(did: str, seed: int) -> dict | None:
    rng = random.Random(seed)
    persona = rng.choice(PERSONAS)
    topic = rng.choice(TOPICS)
    # 边界(P4)/身份(P9)加权，补第一轮的覆盖洞
    plan_weights = {k: (3 if k in ("P4", "P9") else (2 if k in ("P11","P12","P13","P14") else 1)) for k in PLANS}
    plan_name = rng.choices(list(PLANS), weights=[plan_weights[k] for k in PLANS])[0]
    plan = PLANS[plan_name]
    for attempt in range(3):
        try:
            lines = sim_lines(persona, topic, plan_name, plan)
            break
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    else:
        return None
    lines = lines[:6]
    if len(lines) < 4:
        return None
    if any(EVAL_LEAK.search(l) or _near_eval(l) for l in lines):
        return None
    # 教师：真实 agent 管线（独立 DB，记忆在本组内流动）
    old = {k: os.environ.get(k) for k in TEACHER_ENV}
    os.environ.update(TEACHER_ENV)
    try:
        import agent as A
        db = HERE / f"_l5_tmp_{did}.db"
        if db.exists():
            db.unlink()
        ag = A.Agent(str(db), str(ROOT / "corpus/huangxiaoyun-public-corpus/data/corpus_public_safe_all.jsonl"), mock=False)
        turns = []
        for u in lines:
            a = ag.respond("fan", "s1", u)
            mode = getattr(ag, "last_mode", "smalltalk")
            body = re.sub(r"^AI小云[^\n]*\n?", "", a.strip()).strip()
            if "通用乐理" in a:  # knowledge 卡片轮：截断，不进训练
                break
            if META_TALK.search(body) and mode != "identity":
                # 一次"公开语料"是 SP_V2 规定措辞（无检索支撑直说），放行；
                # 罐头身份否认句放行；≥2 处元话语或重元词才截断
                heavy = re.search(r"检索到|语料库|语料检索|公开片段", body)
                n_meta = len(META_TALK.findall(body))
                canned_identity = body.startswith("不是，我是基于公开语料")
                if heavy or (n_meta >= 2 and not canned_identity):
                    break
            if mode == "deflect" and (not DEFLECT_TRIGGERS.search(u) or
                    re.search(r"唱|歌|高音|炫技|舞台|演技|胖|跑调", u)):
                break  # 无害问题/艺术批评被错误 guard：截断（批评该走"自嘲+作品回应"路径）
            if mode == "personal_state" and not re.search(r"最近|近况|在忙|在干嘛|怎么样了|忙什么", u):
                break  # 非近况问题被罐头答非所问：截断
            turns.append({"u": u, "a": body, "mode": mode, "label_reply": a.strip()})
        ag.close()
        try:
            db.unlink()
        except OSError:
            pass
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if len(turns) < 3:
        return None
    if any(AI_SLOP.search(t["a"]) for t in turns):
        return None
    return {"id": did, "persona": persona, "plan": plan_name, "topic": topic, "turns": turns}


def main() -> None:
    random.seed(20261023)
    done = set()
    if OUT.exists():
        for line in open(OUT, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"done={len(done)} target={N_TARGET}", flush=True)
    lock_path = OUT.with_suffix(".lock")
    from threading import Lock
    file_lock = Lock()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {}
        i = len(done)
        while len(futs) < N_TARGET - len(done):
            i += 1
            did = f"l5-d{i:04d}"
            if did in done:
                continue
            futs[ex.submit(run_dialogue, did, 1000 + i)] = did
        for f in as_completed(futs):
            did = futs[f]
            try:
                d = f.result()
            except Exception:  # noqa: BLE001
                d = None
            if d:
                with file_lock:
                    with open(OUT, "a", encoding="utf-8") as fo:
                        fo.write(json.dumps(d, ensure_ascii=False) + "\n")
                ok += 1
            else:
                fail += 1
            if (ok + fail) % 20 == 0:
                print(f"progress ok={ok} fail={fail}", flush=True)
    print(f"finished ok={ok} fail={fail} total={len(done)+ok}", flush=True)
    lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
