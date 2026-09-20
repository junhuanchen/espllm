#!/usr/bin/env python3
"""Trace greedy quantized-model tokens for comparison with ESP32 logs."""

import argparse
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@torch.no_grad()
def prompt_last_token_experts(llm, context):
    """Trace MoE selections and residual summaries for the prompt-tail token."""
    model = llm.model
    x = model.tok_emb(context)
    cos = model.rope_cos[: context.size(1)].to(x.device)
    sin = model.rope_sin[: context.size(1)].to(x.device)
    experts = []
    activations = []
    for block in model.blocks:
        x = x + block.attn(block.ln1(x), cos, sin)
        activations.append(("attn", activation_summary(x)))
        mlp_input = block.ln2(x)
        router_logits = block.mlp.router(mlp_input)
        experts.append(router_logits[0, -1].argmax().item())
        mlp_out, _ = block.mlp(mlp_input)
        x = x + mlp_out
        activations.append(("mlp", activation_summary(x)))
    return experts, activations


def activation_summary(x):
    values = x[0, -1].detach().float().cpu().tolist()
    return math.fsum(values), math.fsum(value * value for value in values), max(abs(value) for value in values)


def main():
    parser = argparse.ArgumentParser(description="Trace greedy ESP-LLM tokens for ESP32 comparison.")
    parser.add_argument("--target", default="esp32s3", choices=["esp8266", "esp32", "esp32s3"])
    parser.add_argument("--question", default="嘟嘟可是谁的物品？")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    # main.py selects its architecture at import time.
    sys.argv = ["main.py", f"--target={args.target}"]
    import main as llm

    llm.model = llm.Transformer(group_size=llm.qat_group_size).to(llm.device)
    llm.load_model()

    prompt = f"User: {args.question.strip()}\nBot:"
    context = llm.encode(prompt).unsqueeze(0).to(llm.device)

    print(f"prompt: {prompt}")
    print("encoded token ids:", " ".join(str(token) for token in context[0].tolist()))
    experts, activations = prompt_last_token_experts(llm, context)
    print("prompt-tail MoE experts:", " ".join(str(expert) for expert in experts))
    for index in range(llm.n_layer):
        for stage, (total, sumsq, maxabs) in activations[index * 2 : index * 2 + 2]:
            print(
                f"prompt-tail layer {index + 1} {stage} "
                f"sum={total:.7g} sumsq={sumsq:.7g} maxabs={maxabs:.7g}"
            )
    generated = []
    with torch.no_grad():
        for step in range(1, args.max_new_tokens + 1):
            logits, _ = llm.model(context[:, -llm.block_size:])
            next_logits = logits[0, -1, :]
            values, ids = torch.topk(next_logits, k=args.top_k)
            next_id = ids[0].item()
            top = " ".join(
                f"#{rank}=id{token_id}:{logit:.5f}"
                for rank, (token_id, logit) in enumerate(zip(ids.tolist(), values.tolist()), 1)
            )
            print(f"step {step:02d}: selected=id{next_id} {top}")
            if next_id == 0:
                break
            generated.append(next_id)
            context = torch.cat((context, torch.tensor([[next_id]], device=llm.device)), dim=1)
            decoded = llm.decode(generated)
            if "\n" in decoded:
                break
    print("generated token ids:", " ".join(str(token) for token in generated))
    print("decoded reply:", llm.decode(generated).replace("\ufffd", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
