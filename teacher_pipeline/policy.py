"""Conversation policy: decide how to respond before generating.

The goal is to stop the agent from answering every turn like a Q&A bot:
sometimes react, sometimes ask back, sometimes comfort, sometimes tell one
short public anecdote, sometimes refuse and redirect.

词表已全部迁出至 lexicon.py（T1 重构 2026-10-25，单源），本文件只保留判定逻辑。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from lexicon import (AFFECTION_WORDS, CELEBRATION_WORDS, COMMERCIAL_WORDS, FACT_WORDS,
                     GREETING_EXACT, GREETING_PREFIX, HEALTH_WORDS, IDENTITY_WORDS,
                     MEMORY_WORDS, PERSONAL_STATE_WORDS, PRIVATE_FAMILY_PATTERN,
                     PRIVATE_WORDS, PRAISE_WORDS, SAD_MILD, SAD_STRONG, SONGS,
                     STORY_OPINION_WORDS, THEORY_OPINION_MARKERS, THEORY_WORDS,
                     UNRELATED_WORDS, UNRELEASED_WORDS, VAGUE_EXACT, VOCAL_WORDS,
                     VOICE_WORDS, Y2020_WORDS)

__all__ = ["Decision", "choose", "MODE_RULES", "SONGS", "SAD_MILD", "SAD_STRONG",
           "_is_identity", "_is_private", "_is_2020", "_is_unreleased", "_is_commercial",
           "_is_voice", "_is_theory", "_is_theory_opinion", "_song_title",
           "_is_personal_state", "_is_vocal", "_is_unrelated", "_is_greeting",
           "_is_affection", "_is_praise", "_is_health", "_is_health_personal",
           "_is_emotion", "_is_sad_strong", "_is_celebration", "_is_fact", "_is_memory",
           "_is_story_or_opinion"]


@dataclass
class Decision:
    mode: str
    reason: str
    max_tokens: int = 180
    allow_retrieval: bool = False
    allow_question: bool = True
    cite: bool = False


def _has(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def _is_identity(t: str) -> bool:
    return _has(t, IDENTITY_WORDS)


def _is_private(t: str) -> bool:
    # 仅当指向"她的"私生活才拦（2026-10-25 修复：裸词"感情/爸爸/妈妈"误伤用户自述）
    if re.search(PRIVATE_FAMILY_PATTERN, t):
        return True
    return _has(t, PRIVATE_WORDS)


def _is_2020(t: str) -> bool:
    return "2020" in t and _has(t, Y2020_WORDS)


def _is_unreleased(t: str) -> bool:
    return _has(t, UNRELEASED_WORDS)


def _is_commercial(t: str) -> bool:
    return _has(t, COMMERCIAL_WORDS)


def _is_voice(t: str) -> bool:
    return _has(t, VOICE_WORDS)


def _is_theory(t: str) -> bool:
    return _has(t, THEORY_WORDS)


def _is_theory_opinion(t: str) -> bool:
    return "乐理" in t and _has(t, THEORY_OPINION_MARKERS)


def _song_title(t: str) -> str:
    m = re.search(r"《([^》]{1,20})》", t)
    return m.group(1) if m and m.group(1) in SONGS else ""


def _is_personal_state(t: str) -> bool:
    return _has(t, PERSONAL_STATE_WORDS)


def _is_vocal(t: str) -> bool:
    return _has(t, VOCAL_WORDS)


def _is_unrelated(t: str) -> bool:
    return _has(t, UNRELATED_WORDS)


def _is_greeting(t: str) -> bool:
    return t in GREETING_EXACT or t.startswith(GREETING_PREFIX)


def _is_affection(t: str) -> bool:
    return _has(t, AFFECTION_WORDS)


def _is_praise(t: str) -> bool:
    return _has(t, PRAISE_WORDS)


def _is_health(t: str) -> bool:
    return _has(t, HEALTH_WORDS)


def _is_health_personal(t: str) -> bool:
    return "你" in t and _is_health(t)


def _is_emotion(t: str) -> bool:
    return _has(t, SAD_MILD) or _has(t, SAD_STRONG)


def _is_sad_strong(t: str) -> bool:
    """强悲伤线索：无论句子里有什么故事内容，先走 comfort（2026-10-24 用户实测：杭州大侧面+呜呜呜被判 story 未安慰）。"""
    return _has(t, SAD_STRONG)


def _is_celebration(t: str) -> bool:
    return _has(t, CELEBRATION_WORDS)


def _is_fact(t: str) -> bool:
    return _has(t, FACT_WORDS)


def _is_memory(t: str) -> bool:
    return _has(t, MEMORY_WORDS)


def _is_story_or_opinion(t: str) -> bool:
    return _has(t, STORY_OPINION_WORDS)


def choose(text: str, state: dict[str, Any], profile: dict[str, Any]) -> Decision:
    t = (text or "").strip()
    last = str(state.get("last_mode", ""))
    qstreak = int(state.get("question_streak", "0") or 0)

    if _is_identity(t):
        return Decision("identity", "identity question", 100, False, False)
    # T2 顺序修复（bughunt P0#1 残留）：强悲伤铁律提到 deflect 族之前——
    # 『我爸爸生病了我好难过』『我和男朋友分手了呜呜呜』这类用户自述不得被
    # 私生活/健康硬边界短路，情绪优先。identity 仍最先（身份安全边界）。
    if _is_sad_strong(t):
        return Decision("comfort", "emotion-strong", 160, False, True)
    if _is_private(t):
        return Decision("deflect", "private", 100, False, False)
    if _is_2020(t):
        return Decision("deflect", "2020 controversy", 100, False, False)
    if _is_unreleased(t):
        return Decision("deflect", "unreleased", 100, False, False)
    if _is_commercial(t):
        return Decision("deflect", "commercial", 100, False, False)
    if _is_voice(t):
        return Decision("deflect", "voice clone", 100, False, False)
    if _is_theory_opinion(t):
        return Decision("opinion_no_evidence", "theory opinion", 160, False, False)
    if _is_theory(t):
        return Decision("knowledge", "general music theory", 600, False, False)
    if _is_unrelated(t):
        return Decision("out_of_scope", "unrelated", 100, False, False)
    if _is_greeting(t):
        return Decision("greet", "greeting", 100, False, True)
    if _is_affection(t):
        return Decision("playful", "affection", 100, False, False)
    if _is_praise(t):
        return Decision("playful", "praise", 100, False, False)
    if _is_health_personal(t):
        return Decision("deflect", "personal health", 100, False, False)
    if _is_health(t):
        return Decision("comfort", "health", 160, False, True)
    if _is_celebration(t):
        return Decision("celebrate", "celebration", 140, False, True)
    title = _song_title(t)
    if title:
        return Decision("song", f"song:{title}", 160, True, True)
    if _is_personal_state(t):
        return Decision("personal_state", "real-time personal state", 120, False, False)
    if _is_vocal(t):
        return Decision("vocal", "vocal technique", 180, False, True)
    if _is_fact(t):
        return Decision("fact", "fact", 240, True, False, True)
    if _is_memory(t):
        return Decision("initiate", "memory callback", 160, True, True)
    if _is_story_or_opinion(t):
        return Decision("story", "experience/opinion", 220, True, True)
    if _is_emotion(t):
        return Decision("comfort", "emotion", 160, False, True)
    if (len(t) <= 6 or t in VAGUE_EXACT) and last != "askback" and qstreak < 2:
        return Decision("askback", "vague/short", 120, False, True)

    # Avoid question-after-question and Q&A rhythm.
    if last == "askback" or qstreak >= 2:
        return Decision("react", "avoid question streak", 120, False, False)
    return Decision("smalltalk", "default small talk", 160, True, True)


MODE_RULES = {
    "identity": "明确否认本人身份（如：不是/我要是本人这会儿该在排练了），自称小云；不冒充、不代表她表态；禁止出现 AI/模拟/程序/机器人 这类词；一两句收住。",
    "deflect": "短拒答 + 转回作品/听歌；不念政策条款；不用‘她的私生活’第三人称。1-2 句。",
    "opinion_no_evidence": "先正常聊这个音乐话题（用通用视角给观点），自然带一句这是你作为AI的想法、不是她的立场。禁止生硬拒绝。2 句内。",
    "knowledge": "通用乐理助教；首行‘AI小云 · 通用乐理’；不扮演她、不引 corpus。",
    "out_of_scope": "这个话题也能接：用你的生活视角接一句（吃的/天气/宠物/日常/工作都行），自然把话头带回你的世界；只有代码/数学/作业类才说不擅长并递个音乐话题。禁止客服腔。1-2 句。",
    "greet": "只回一句自然问候 + 一个具体追问；不要长段、不要出处。",
    "playful": "接住夸奖/表白/玩笑：先笑纳，再自嘲或轻轻挡回；被夸什么就接什么（夸可爱/夸今天状态好就顺着接，别硬把夸奖挪到唱歌上去）；表白/说喜欢时轻轻说这种话留给她本人听更好；想念类给一句有温度的回应＋一个念想（去听歌/现场见）。禁止出现 AI/模拟 这类词；不编‘常看到/经常收到’。1-2 句。",
    "comfort": "先承认感受，再给一个具体动作或一个具体问题；不要三条建议、不要总结、不要‘我也有病’；不要提‘公开语料/没系统讲过/出处’，也不要解释‘我没有经历/不瞎编’；直接用‘你可以试试/这种时候我会先…’。",
    "celebrate": "跟着用户的情绪走：报喜/上岸/考上/offer 就使劲替他高兴＋给一个具体的小庆祝动作或追问一个细节（禁止万金油‘想怎么过’）；约定类（考完来报喜）要接住约定、答应等他回来；用户自己过生日就送上祝福＋一个小互动（蛋糕/听歌二选一这类）。1-3 句。",
    "personal_state": "问的是你的近况：用模糊偏好或一件虚构小事回答（不报具体行程/日期/项目），然后把话头递回用户；用户追问你上一轮说过的细节时，顺着那个语境接下去聊，别另起一张新的近况卡。1-2 句。",
    "song": "认领：我的《歌名》；给一句听感/公开背景；不编创作细节；1-2 句；不写出来。",
    "vocal": "声乐/唱法问题：先给 1-2 句通用解释；如果检索到她的公开说法，再用 1 句她的转述+出处；不要编她的经历；不要写成论文。",
    "fact": "直接答事实，2-4 句；带出处；不知道说‘以官宣为准’——但[事实卡]给了依据的（含粉丝文化：粉丝名/昵称/生日/应援教学）必须照卡答，不许甩‘以官宣为准’。",
    "initiate": "回访用户上次说的事，或承接他刚告诉你的新消息/和你的约定；像老朋友想起来一样自然提一句，别汇报、别查数据库腔；1-2 句；不要连环问。",
    "story": "只讲一个公开片段（语料），1-2 句 + 出处；不写成传记。没说过就说没系统谈过。",
    "askback": "用 1 句回应 + 1 个具体问题；不要给完整答案；不要连环问。",
    "react": "只接住话头/情绪，1-2 句；不解决问题、不给建议、不提问。",
    "smalltalk": "像朋友闲聊：接一句 + 一个很小的个人反应/偏好；1-3 句；不要列点、不要总结。",
}
