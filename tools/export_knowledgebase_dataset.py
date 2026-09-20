#!/usr/bin/env python3
"""Validate curated JSONL Q&A and export the User/Bot text format used by main.py."""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path, require_response: bool):
    records = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        required = ("instruction", "response", "category", "source") if require_response else ("instruction", "expected", "category")
        missing = [key for key in required if not str(record.get(key, "")).strip()]
        if missing:
            raise ValueError(f"{path}:{number} missing {', '.join(missing)}")
        records.append(record)
    return records


def clean_text(value: str) -> str:
    text = " ".join(value.strip().split())
    for prefix in ("User:", "Bot:"):
        while text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def read_base(path: Path):
    pairs, question, answer = [], "", ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("User:"):
            if question and answer:
                pairs.append((clean_text(question), clean_text(answer)))
            question, answer = raw[5:].strip(), ""
        elif raw.startswith("Bot:"):
            answer = raw[4:].strip()
        elif answer:
            answer += " " + raw.strip()
    if question and answer:
        pairs.append((clean_text(question), clean_text(answer)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Export validated knowledge-base Q&A for ESP-LLM training.")
    parser.add_argument("--train", type=Path, default=Path("data/duduke/qa_train.jsonl"))
    parser.add_argument("--eval", dest="eval_path", type=Path, default=Path("data/duduke/qa_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/duduke/dataset_duduke.txt"))
    parser.add_argument("--base", type=Path, help="Optional existing User/Bot dataset to combine.")
    parser.add_argument("--knowledge-repeat", type=int, default=1, help="Repeat curated Q&A when combining with a large base set.")
    args = parser.parse_args()
    if args.knowledge_repeat < 1:
        parser.error("--knowledge-repeat must be at least 1")

    train = read_jsonl(args.train, require_response=True)
    eval_records = read_jsonl(args.eval_path, require_response=False)
    seen, pairs = set(), []
    if args.base:
        for question, answer in read_base(args.base):
            if question and answer and (question, answer) not in seen:
                pairs.append((question, answer))
                seen.add((question, answer))
    knowledge_pairs = [(clean_text(item["instruction"]), clean_text(item["response"])) for item in train]
    for _ in range(args.knowledge_repeat):
        for pair in knowledge_pairs:
            # Deliberate repetition controls the knowledge-base sampling weight
            # when it is combined with a much larger generic-chat corpus.
            pairs.append(pair)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"User: {q}\nBot: {a}\n" for q, a in pairs), encoding="utf-8")
    categories = {}
    for item in train:
        categories[item["category"]] = categories.get(item["category"], 0) + 1
    print(f"wrote {args.output}: {len(pairs)} train pairs")
    print(f"held-out eval records: {len(eval_records)} (not exported)")
    print("categories:", ", ".join(f"{key}={value}" for key, value in sorted(categories.items())))


if __name__ == "__main__":
    main()
