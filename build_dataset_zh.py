#!/usr/bin/env python3
"""Build a deterministic Chinese-only training corpus for the ESP32-S3 model.

The `User:` / `Bot:` record markers are intentionally kept because main.py
parses them. All question and answer content is Chinese (plus numbers).
"""

import json
import random
import re
from collections import Counter
from pathlib import Path


RNG = random.Random(20260920)
PAIRS = []


def add(question, answer, category="通用"):
    question = " ".join(question.strip().split())
    answer = " ".join(answer.strip().split())
    if not question or not answer:
        return
    if re.search(r"[A-Za-z]", question + answer):
        raise ValueError(f"English content is not allowed: {question!r}")
    PAIRS.append((question, answer, category))


def add_choices(questions, answers):
    for question in questions:
        for answer in answers:
            add(question, answer)


def common_chat():
    add_choices(
        ["你好", "嗨", "在吗", "你好呀", "早上好", "晚上好"],
        ["你好，很高兴和你聊天。", "我在，你想聊什么？", "你好呀，今天过得怎么样？"],
    )
    add_choices(
        ["你是谁", "你是什么", "介绍一下你自己", "你能做什么"],
        [
            "我是运行在微控制器上的小型离线聊天助手。",
            "我是一个小型本地语言模型，可以进行简单聊天和问答。",
            "我可以陪你聊天、回答简单问题，并处理一些基础计算。",
        ],
    )
    add_choices(
        ["你能上网吗", "你知道今天的新闻吗", "你知道现在几点吗", "你能看见我吗"],
        [
            "我不能联网，也无法获取实时信息。",
            "我没有摄像头和真实世界感知能力。",
            "我无法知道当前时间，但可以和你讨论一般性问题。",
        ],
    )
    add_choices(
        ["我今天很难过", "我有点焦虑", "我觉得很累", "我心情不好"],
        [
            "听起来你现在不太轻松。可以先休息一下，也可以说说发生了什么。",
            "你的感受很重要。试着做几次缓慢呼吸，再把事情拆成小步骤。",
            "辛苦了。先照顾好自己，必要时可以向信任的人寻求支持。",
        ],
    )
    add_choices(
        ["谢谢", "谢谢你", "你真好", "帮大忙了"],
        ["不客气。", "很高兴能帮到你。", "不用谢，有问题随时再问。"],
    )
    add_choices(
        ["再见", "晚安", "我先走了", "下次见"],
        ["再见，祝你一切顺利。", "晚安，祝你睡个好觉。", "下次见。"],
    )
    add_choices(
        ["给我一个建议", "我不知道该怎么办", "怎样开始做一件事"],
        [
            "先确定最小的一步，完成后再决定下一步。",
            "把问题写下来，区分能控制和暂时不能控制的部分。",
            "先从五分钟能完成的小任务开始，行动通常比等待更有帮助。",
        ],
    )
    add_choices(
        ["什么是人工智能", "人工智能是什么", "什么是机器学习"],
        [
            "人工智能是让计算机完成识别、预测或生成等任务的技术。",
            "机器学习是让模型从数据中学习规律的方法，而不是为每个规则手工编程。",
        ],
    )


def general_knowledge_pairs():
    """Expand conversational coverage so number exercises cannot dominate."""
    base = [(question, answer) for question, answer, _ in PAIRS]
    wrappers = [
        "", "请问，", "我想问，", "能告诉我，", "麻烦回答，", "请用简单的话说，",
        "我有个问题：", "可以解释一下，", "帮我想想，", "你怎么看，", "请认真回答，",
        "我想了解，", "方便说说，", "请直接回答，", "我需要一点建议：", "请帮我理解，",
        "我有些疑惑：", "请告诉我，", "麻烦说明，", "我需要知道，", "可以回答，",
        "请帮我回答，", "我想弄明白，", "请用通俗的话说，", "请简单说明，",
    ]
    for question, answer in base:
        stem = question.rstrip("？。")
        for wrapper in wrappers:
            add(f"{wrapper}{stem}？", answer, "通用")

    facts = [
        ("一年有几个月", "一年有十二个月。"),
        ("一周有几天", "一周有七天。"),
        ("一天有多少小时", "一天有二十四小时。"),
        ("水在正常情况下多少度结冰", "水在正常大气压下零摄氏度结冰。"),
        ("太阳从哪里升起", "通常说太阳从东方升起。"),
        ("植物为什么需要阳光", "植物可以利用阳光进行光合作用，制造生长所需的养分。"),
        ("什么是节约", "节约是在满足需要的前提下，避免浪费资源。"),
        ("什么是尊重", "尊重是认真对待他人的感受、权利和边界。"),
        ("什么是合作", "合作是为了共同目标分工配合、互相支持。"),
        ("什么是健康的作息", "健康作息通常包括规律睡眠、适量运动和均衡饮食。"),
        ("为什么要洗手", "洗手能减少污垢和部分病原体传播。"),
        ("为什么要喝水", "适量喝水有助于身体维持正常功能。"),
        ("怎样保护眼睛", "阅读时保持合适距离，适当休息，并避免长时间盯着屏幕。"),
        ("怎样整理房间", "先分类物品，再从一个小区域开始收拾，最后把常用物品放回固定位置。"),
        ("怎样安排学习", "把学习目标拆成小任务，安排专注时间，并及时复习。"),
        ("遇到陌生链接怎么办", "不要随意打开或输入个人信息，先向可信的人核实。"),
        ("密码应该告诉别人吗", "密码应尽量保密，不要随意告诉他人。"),
        ("怎样面对失败", "先总结原因，再调整方法；一次失败不等于没有能力。"),
        ("朋友难过时怎么办", "可以耐心倾听、表达关心，并尊重对方是否愿意继续说。"),
        ("什么是目标", "目标是希望达成的具体结果，可以拆成小步骤逐步完成。"),
    ]
    prompts = ["{}？", "请解释{}。", "我想知道{}。", "能说说{}吗？", "用简单的话回答：{}。"]
    for subject, answer in facts:
        for prompt in prompts:
            add(prompt.format(subject), answer, "通用")

    terms = {
        "耐心": "耐心是在等待或处理困难时保持平静并持续努力。",
        "诚实": "诚实是如实表达和做事，不故意欺骗他人。",
        "责任": "责任是对自己应做的事情认真承担。",
        "习惯": "习惯是长期重复后形成的较稳定行为方式。",
        "复习": "复习是再次回顾学过的内容，以加深理解和记忆。",
        "计划": "计划是为目标安排步骤、时间和资源。",
        "专注": "专注是把注意力集中在当前重要任务上。",
        "沟通": "沟通是通过语言或其他方式交换信息和理解彼此。",
        "隐私": "隐私是个人不希望被随意公开的信息和生活空间。",
        "安全": "安全是避免伤害、风险和不必要损失的状态。",
    }
    for term, answer in terms.items():
        for prompt in ["什么是{}？", "{}是什么意思？", "请解释{}。", "用简单的话说说{}。"]:
            add(prompt.format(term), answer, "通用")
    add_choices(
        ["你能上网吗", "你知道今天的新闻吗", "你知道现在几点吗", "你能看见我吗"],
        [
            "我不能联网，也无法获取实时信息。",
            "我没有摄像头和真实世界感知能力。",
            "我无法知道当前时间，但可以和你讨论一般性问题。",
        ],
    )
    add_choices(
        ["我今天很难过", "我有点焦虑", "我觉得很累", "我心情不好"],
        [
            "听起来你现在不太轻松。可以先休息一下，也可以说说发生了什么。",
            "你的感受很重要。试着做几次缓慢呼吸，再把事情拆成小步骤。",
            "辛苦了。先照顾好自己，必要时可以向信任的人寻求支持。",
        ],
    )
    add_choices(
        ["谢谢", "谢谢你", "你真好", "帮大忙了"],
        ["不客气。", "很高兴能帮到你。", "不用谢，有问题随时再问。"],
    )
    add_choices(
        ["再见", "晚安", "我先走了", "下次见"],
        ["再见，祝你一切顺利。", "晚安，祝你睡个好觉。", "下次见。"],
    )
    add_choices(
        ["给我一个建议", "我不知道该怎么办", "怎样开始做一件事"],
        [
            "先确定最小的一步，完成后再决定下一步。",
            "把问题写下来，区分能控制和暂时不能控制的部分。",
            "先从五分钟能完成的小任务开始，行动通常比等待更有帮助。",
        ],
    )
    add_choices(
        ["什么是人工智能", "人工智能是什么", "什么是机器学习"],
        [
            "人工智能是让计算机完成识别、预测或生成等任务的技术。",
            "机器学习是让模型从数据中学习规律的方法，而不是为每个规则手工编程。",
        ],
    )


def arithmetic_pairs():
    # Numeric questions provide broad Chinese coverage without relying on an
    # English-only external corpus. The device's math harness remains faster
    # for English expressions; these samples train Chinese conversational form.
    phrasings = [
        "{a}加{b}等于多少？",
        "请计算{a}+{b}",
        "{a}和{b}相加是多少？",
        "帮我算一下{a}加{b}",
    ]
    for _ in range(500):
        a, b = RNG.randint(0, 9999), RNG.randint(0, 9999)
        add(RNG.choice(phrasings).format(a=a, b=b), f"{a + b}。", "计算")

    phrasings = ["{a}减{b}等于多少？", "请计算{a}-{b}", "{a}减去{b}是多少？"]
    for _ in range(400):
        a, b = RNG.randint(0, 9999), RNG.randint(0, 9999)
        if a < b:
            a, b = b, a
        add(RNG.choice(phrasings).format(a=a, b=b), f"{a - b}。", "计算")

    phrasings = ["{a}乘{b}等于多少？", "请计算{a}乘以{b}", "{a}和{b}相乘是多少？"]
    for _ in range(300):
        a, b = RNG.randint(0, 99), RNG.randint(0, 99)
        add(RNG.choice(phrasings).format(a=a, b=b), f"{a * b}。", "计算")


def conversion_pairs():
    units = [
        ("米", "厘米", 100),
        ("厘米", "毫米", 10),
        ("千米", "米", 1000),
        ("千克", "克", 1000),
        ("小时", "分钟", 60),
        ("分钟", "秒", 60),
    ]
    for source, target, factor in units:
        for _ in range(60):
            value = RNG.randint(1, 500)
            question = RNG.choice([
                f"{value}{source}等于多少{target}？",
                f"请把{value}{source}换算成{target}",
                f"{value}{source}有多少{target}？",
            ])
            add(question, f"{value * factor}{target}。", "换算")


def knowledge_pairs():
    source = Path("data/duduke/qa_train.jsonl")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    prefixes = ["", "请问", "告诉我，", "我想知道，", "能说说，", "请回答："]
    suffixes = ["", "。", "？"]
    variants = []
    for record in records:
        for prefix in prefixes:
            for suffix in suffixes:
                question = prefix + record["instruction"].rstrip("？。") + suffix
                variants.append((question, record["response"]))
    RNG.shuffle(variants)
    # 40 source records x 18 surface forms = 720 records. Two shuffled
    # passes keep the specialized knowledge visible without overwhelming
    # general Chinese conversation.
    for _ in range(2):
        RNG.shuffle(variants)
        for question, answer in variants:
            add(question, answer, "知识库")


def main():
    common_chat()
    general_knowledge_pairs()
    arithmetic_pairs()
    conversion_pairs()
    knowledge_pairs()
    RNG.shuffle(PAIRS)
    output = Path("dataset_zh.txt")
    output.write_text("".join(f"User: {q}\nBot: {a}\n" for q, a, _ in PAIRS), encoding="utf-8")
    chinese_pairs = sum(bool(re.search(r"[\u4e00-\u9fff]", q + a)) for q, a, _ in PAIRS)
    counts = Counter(category for _, _, category in PAIRS)
    print(f"wrote {output}: {len(PAIRS)} pairs; Chinese-content pairs: {chinese_pairs}")
    print("category counts:", ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))


if __name__ == "__main__":
    main()
