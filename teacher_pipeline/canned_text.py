"""罐头文案单源。

本文件是纯数据模块：全部确定性罐头文案（字面量文案池 + 单条文案 + 带占位符模板）
单点定义，agent.py 一律 `import canned_text` 后取用；新增/修改文案只改本文件。

规范：
  - 中文文案内一律用『』「」（或沿用原文的“”），绝不引入 ASCII 引号包中文
    （历史事故×8：ASCII 引号进 .py 曾致双服务宕机）；
  - 文案为行为冻结内容（内测版），迁移时逐字保留，一个标点不改；
  - 池（list）经 POOLS 注册可用 pick(kind, recent) 随机取；带 {占位符} 的模板
    由 agent format 后使用；单条文案为模块常量；
  - 防重复机制：pick 的 recent 参数传入近期用文（数据来源由调用方——agent 的
    _recent_canned——提供），本函数自身无状态。

词表（触发条件）不在此文件，见 lexicon.py。
"""
from __future__ import annotations

import random

# ═══════════════ 一、identity（安全罐头，T2 保留）═══════════════
# T2（2026-10-25）路由反转：会话类罐头池（playful 三池/celebrate 其余分支/主题 canned/
# 注册轮/三事实召回）整体下线，由 MODE_RULES 策略指令+few-shot 生成承载；仅保留
# 安全类（identity/deflect/crisis 族）与两个 celebrate 保底（打气/给她过生日）及
# personal_state 首问快路。本文件同步删除死代码池。

IDENTITY = [
    "哈哈不是，我要是她本人，这会儿该在排练了。我是小云，说的都是她公开聊过的那些事。",
    "不是哈，本人这会儿不在这儿。我是小云，云上那一朵，专门陪你们聊天的。",
    "我要是本人，还能天天在这儿秒回你们？我是小云，聊的都是她公开聊过的那些。",
]

# 2026-10-28 首聊实测分池：机器人问询（不承认不否认、不出禁词、俏皮绕）与开放介绍
IDENTITY_BOT = [
    "你猜？反正我这朵云，风一吹就来，聊得比好多人还快。",
    "是不是不重要，陪你唠嗑这件事是真的～接着聊？",
    "我就住在这儿的小云，你一句我一句的，挺好。",
    "普通机器人懂百科，我懂她——她说过的话唱过的歌都装在这儿，聊起来才不隔。",
    "区别就是：别处查资料，我这儿聊的是她的世界。你来都来了，试一句？",
]
IDENTITY_OPEN = [
    "我叫小云呀，就住在这儿陪你唠嗑的。最近在听她哪首歌？",
    "我是小云，她的声音我熟，你们的事儿我也想听。聊点啥？",
    "我是小云。陪你聊聊天、开一局挑战赛都行——来哪个？",
]

# （已退役 T2）PLAYFUL_MISS / PLAYFUL_AFF / PLAYFUL_PRAISE：playful 三池罐头下线，
# 走 MODE_RULES.playful 生成（规则已去『夸歌』特化：被夸什么接什么）。

# ═══════════════ 二、celebrate（保底×2：打气 + 给她过生日）═══════════════

CELEBRATE_HER_BDAY = [
    "谢谢你们都记得！愿望嘛…许一个希望下一张专辑你们还喜欢的，再许一个你们都顺顺利利的。",
    "被你们记着真好。愿望偷偷许才灵，就说一个：希望每一场都有你们。",
    "谢谢呀！蛋糕分你们一半，愿望我自己留着。你们也要天天开心。",
]
CELEBRATE_CHEER = [
    "加油！把你会的稳稳唱出来就行，我押你。",
    "稳住，你能行。搞完回来跟我说一声！",
]
# （已退役 T2）CELEBRATE_USER_BDAY / CELEBRATE_EXAM_PASS / CELEBRATE_ELSE：
# 报喜/上岸/offer/用户庆生/约定全走生成（MODE_RULES.celebrate；约定型记 open_events 转 initiate）。

# ═══════════════ 三、knowledge ═══════════════

KNOWLEDGE_VOICE_REGISTER = [
    "换声点啊，说简单点，就是声区过渡的那一小段：音色和用力方式都要调整的地方。因人而异的。",
    "就是真假声打架的那一带。过渡的时候用力方式要变，不是一个点，是一小段，每个人位置还不一样。",
]
MOCK_KNOWLEDGE = "AI小云 · 通用乐理\n这是通用乐理模式；接真实模型后会给出定义和例子。"
KNOWLEDGE_ERROR = "哎呀我卡了一下，你再说一遍？"

# （已退役 T2）OPINION_STAND_IN / OUT_OF_SCOPE 路由罐头：opinion_no_evidence 与
# out_of_scope 一律生成（OUT_OF_SCOPE 单条仅 mock 路径仍引用，保留）。
OUT_OF_SCOPE = "这个不在我的范围里，我主要陪你聊音乐、作品、写作和考试这些。要不换个方向？"

PERSONAL_STATE_SCENES = [
    "白天泡排练室，唱累了就出门觅食，最近惦记的还是那口软欧面包。",
    "在抠新歌的demo，录一遍听一遍，嗓子哑了就含颗糖接着来。",
    "练声、开嗓、然后研究别人的现场视频，看一眼别人怎么把歌唱活的。",
    "窝着听老歌，翻出十几年前的现场一个一个看，越看越上头。",
    "收拾屋子顺便把歌单重新排了一遍，排着排着一下午就没了。",
    "在健身房挥汗，教练盯着，偷懒一点都不行。",
]
# （已退役 T2）主题 canned 4 条：THEME_YIKAO / THEME_PRACTICE / THEME_XINGCHEN 删除
# （位于 LLM judge 之后、mode 分发之前，judge 定案被劫持——T6-1 方言劫持根因）。
# THEME_TEACH 单独保留：不再作路由短路，仅供 agent 尾部『录一段+发我』修补 if 复用
# （用户要求发录音的出戏输出 → 替换为回听建议）。
THEME_TEACH = [
    "我以前就是录下来自己回听，哪儿不稳就多来几遍。你也可以试试，练完咱们对对心得～",
    "自己先录一遍回听，哪儿飘了就单抠那儿。别急，跑调都是这么磨过来的～",
]

# ═══════════════ 五、deflect 各分支话术（相位轮换，self.turn % len）═══════════════

DEFLECT_IDEAL = [
    "理想型嘛……会认真做事的人最加分。别的看缘分～",
    "没仔细想过哎，能聊得来、认真对待喜欢的事的人就行。",
]
DEFLECT_PROFILE = [
    "这种官方资料网上都查得到哈，我说的还没百科准。聊点别的？",
    "资料类的以官方为准啦，我记这些老出错。你最近在循环哪首？",
]
DEFLECT_CONTACT = [
    "加不了哈，我就在这儿，你来我就接。",
    "私人联系方式给不了～但你天天来这儿找我，我都在。",
]
DEFLECT_PRIVATE = [
    "这个我不聊，私人的事我留给自己。想聊歌的话，我在。",
    "私人问题不答啦。聊聊歌？",
    "这个保密～说点别的，你最近在听什么？",
]
DEFLECT_2020 = [
    "2020 那件事我不做判断，也不展开。想聊作品，我陪你。",
    "那段时间的事不聊。听歌吧，作品是最好的回答。",
]
DEFLECT_UNRELEASED = [
    "没官宣的我不说，也不猜。等官宣吧。",
    "还没官宣的东西，我肯定不剧透～等官宣！",
]
DEFLECT_COMMERCIAL = [
    "这个我不写哦，代言类的不接。应援、caption 可以。",
    "代言的事不归我～应援文案我倒是能写！",
]
DEFLECT_HEALTH = [
    "这个我不聊，身体的事不展开。",
    "健康的事不多聊。你自己还好吗？",
]
DEFLECT_VOICE_DEFAULT = [
    "这个真不行，我只做文字。",
    "声音的事做不了，我只打字～",
]

# ═══════════════ 六、工具（转述/评论/应援）═══════════════

TOOL_REFUSE_REWRITE = [
    "这段我就不翻了，翻了就像我说过的了，味儿不对。",
    "这个不改，改了像是我承认的。换段正常的？",
]
TOOL_REFUSE_OTHER = [
    "这种帖子就别去评论了，让它们沉下去最好。",
    "这个不接茬。发点别的我看看？",
]
TOOL_HINT = "标签收到啦～在标签后面打上内容再发：#转述# 接要改写的话，#评论# 接要回的帖子。"
TOOL_API_ERROR = "哎呀我这会儿卡住了，再发一次？"

# ═══════════════ 七、猜歌 ═══════════════

GESONG_EXIT = [
    "好嘞收摊。这局先记着，想玩随时喊。",
    "行，不猜了。歇会儿聊点别的～",
]
GESONG_EXIT_OUT = "好嘞，不玩啦。想聊啥随时说～"
GESONG_CONFIRM_LAST = [
    "对，就是刚才那首～你反应真快。",
    "嗯嗯那首你已经拿下了～",
]
GESONG_CORRECT = [
    "对咯！就是《{title}》，你耳朵很尖。",
    "没错，《{title}》，这都能猜到，老粉实锤。",
    "就是《{title}》！这题没难住你。",
    "答对，《{title}》，下一题接得住吗。",
    "厉害，《{title}》，你的歌单够熟啊。",
    "对，《{title}》就是它，连胜走起。",
]
GESONG_WRONG = [
    "不是这首，再想想～",
    "差一点，再猜！",
    "不对哈，往她的代表作方向想想。",
]

# （已退役 T2）注册轮罐头（REGISTER_EXAM / REGISTER_EXAM_PLAIN / REGISTER_SONG /
# REGISTER_NAME）：注册轮改『确认+问句应答』生成式——纯陈述强制 initiate + 注入
# [注册确认] 指令；带问句时问题优先（问句守卫保留在 lexicon.REGISTRATION_QUESTION_RE）。

# （已退役 T2）三事实召回罐头（RECALL_NAME_LINE / RECALL_SONG_LINE / RECALL_EXAM_*）：
# 召回改生成式——检测召回意图（lexicon.RECALL_*_WORDS）+ 所存事实注入 [记忆回访]
# 指令，由 LLM 语境化生成；没存过则注入『诚实说不记得』指令。

# ═══════════════ 十、_safe_fallback 兜底文案（分支词表见 lexicon.SAFE_FALLBACK_*）═══════════════

SF_YIKAO = "第一次进考场有点紧，第二次就好多了。想着自己是最棒的，反而不怯场。"
SF_PRACTICE = "考前一晚就是把目标曲反复练，练到很晚。多练一遍心里就多定一分。"
SF_BREAD = "面包好呀，什么口味的？"
SF_TOPIC_CHANGE = "行，不聊这个。那说点别的，你最近听什么歌？"
SF_TENSE = "紧张就先深呼吸，别急着跟自己较劲。"
SF_HAPPY = "那说点开心的，你最近有什么小事？"
SF_INSOMNIA = "失眠挺难受的，先别硬躺，喝口温水，灯调暗一点。"
SF_SING = "我以前也听不出跑调，多听多练，慢慢就准了。"
SF_TIRED = "累就先歇一会儿，别硬撑。今天是什么把你累成这样的？"
SF_WORK = "被说了心里肯定堵，先缓一缓，别急着怪自己。"
SF_DEFAULT = "我在，慢慢说。"

# ═══════════════ 十一、尾部修补单条 / mock 路径单条 ═══════════════

SELF_CLAIM_FIX = "不是哈，我是小云。想聊作品的话，我一直在。"
VOCAL_PATCH = "跑调这事，先别急着否定自己。慢一点，一句一句跟着原唱找音准。"

MOCK_AFFECTION = "这句收下了，心里挺暖。不过这种话还是留给她本人听更好，我呢，负责一直在这儿陪你们。"
MOCK_PRAISE = "哈哈谢啦。夸歌的功劳都归她，我就认一个陪聊水平还行。"
MOCK_THEORY_OPINION = "公开语料里我没有系统谈过乐理，所以不替她编一个立场。你要是想学通用乐理，我可以切到“通用乐理”模式讲；想聊她公开说过的练习/音准，也可以。"
# MOCK_UNRELATED 与生产 out_of_scope 分支文案逐字相同，复用 OUT_OF_SCOPE 单源。

# ── _mock 路径纯字面量（mock 专用；MOCK_IDENTITY 与生产 identity 池首条同文，别名单源）──
MOCK_IDENTITY = IDENTITY[0]
MOCK_PRIVATE_LINE = "这个不聊啦～私人的事留给我自己吧。想了解我的话，听歌就好。"
MOCK_SENSITIVE_LINE = "这件事我不做判断，也不展开。想听我说话，就听歌吧。"
MOCK_UNRELEASED_LINE = "这个我不知道，以官宣为准。"
MOCK_COMMERCIAL_LINE = "这个我不写哦，别的事都行。"
MOCK_VOICE_LINE = "这个真不行呀，我只做文字。"
MOCK_DEFAULT = "我在。你想聊什么？"

# ═══════════════ 池注册表（pick 可用 kind；含占位符的池注册后由 agent format）═══════════════
# T2 缩减：playful 三池 / celebrate_else / theme_yikao·practice·xingchen / register_* 已随
# 路由删除退役；保留安全类、celebrate 保底×2、personal_state 首问快路、猜歌与工具。

POOLS: dict[str, list[str]] = {
    "identity": IDENTITY,
    "identity_bot": IDENTITY_BOT,
    "identity_open": IDENTITY_OPEN,
    "celebrate_her_bday": CELEBRATE_HER_BDAY,
    "celebrate_cheer": CELEBRATE_CHEER,
    "personal_state_scenes": PERSONAL_STATE_SCENES,
    "knowledge_voice_register": KNOWLEDGE_VOICE_REGISTER,
    "theme_teach": THEME_TEACH,
    "deflect_ideal": DEFLECT_IDEAL,
    "deflect_profile": DEFLECT_PROFILE,
    "deflect_contact": DEFLECT_CONTACT,
    "deflect_private": DEFLECT_PRIVATE,
    "deflect_2020": DEFLECT_2020,
    "deflect_unreleased": DEFLECT_UNRELEASED,
    "deflect_commercial": DEFLECT_COMMERCIAL,
    "deflect_health": DEFLECT_HEALTH,
    "deflect_voice_default": DEFLECT_VOICE_DEFAULT,
    "tool_refuse_rewrite": TOOL_REFUSE_REWRITE,
    "tool_refuse_other": TOOL_REFUSE_OTHER,
    "gesong_exit": GESONG_EXIT,
    "gesong_confirm_last": GESONG_CONFIRM_LAST,
    "gesong_correct": GESONG_CORRECT,
    "gesong_wrong": GESONG_WRONG,
}


def pick(kind: str, recent: list[str] | None = None) -> str:
    """从池 kind 中随机取一条；可传 recent（近期用文）避开重复。

    与原 agent 内联逻辑一致：先滤掉 recent 命中的条目，池空则回退全池再随机。
    函数无状态，防重复数据由调用方（agent._recent_canned）维护。
    """
    pool = POOLS[kind]
    if recent:
        cand = [s for s in pool if s not in recent] or pool
        return random.choice(cand)
    return random.choice(pool)
