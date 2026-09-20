#!/usr/bin/env python3
"""Run deterministic local keyword evaluation against an ESP-LLM checkpoint."""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Chinese ESP-LLM prompts locally.")
    parser.add_argument("--target", default="esp32s3", choices=["esp8266", "esp32", "esp32s3"])
    parser.add_argument("--eval", default="data/chinese/qa_eval_zh.jsonl", type=Path)
    parser.add_argument("--checkpoint", help="Optional checkpoint base path, without .best/.quantized suffix.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--allow-failures", action="store_true", help="Always exit 0 after reporting failures.")
    return parser.parse_args()


def expected_terms(record):
    # This is intentionally a lightweight regression check, not a semantic
    # evaluator. Either comma-separated expected phrase is accepted.
    return [term.strip() for term in record["expected"].replace("或", "、").split("、") if term.strip()]


def main():
    args = parse_args()
    if args.checkpoint:
        os.environ["ESPGPT_CHECKPOINT"] = args.checkpoint
    # main.py chooses its profile at import time; hide evaluator-only flags.
    sys.argv = ["main.py", f"--target={args.target}"]
    import main as llm

    llm.model = llm.Transformer(group_size=llm.qat_group_size).to(llm.device)
    llm.load_model()
    records = [json.loads(line) for line in args.eval.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = 0
    for number, record in enumerate(records, 1):
        prompt = f"User: {record['instruction'].strip()}\nBot:"
        context = llm.encode(prompt).unsqueeze(0).to(llm.device)
        prompt_len = len(llm.decode(context[0].tolist()))
        output_ids = llm.model.generate(context, max_new_tokens=args.max_new_tokens, echo=False)
        reply = llm.decode(output_ids[0].tolist())[prompt_len:].replace("\ufffd", "").strip()
        terms = expected_terms(record)
        ok = any(term in reply for term in terms)
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {number:02d}/{len(records)} {record['category']}: {record['instruction']}")
        print(f"  reply: {reply or '(empty)'}")
        print(f"  expected one of: {', '.join(terms)}")
    print(f"\nSummary: {passed}/{len(records)} keyword checks passed")
    return 0 if passed == len(records) or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
