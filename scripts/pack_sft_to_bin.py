from __future__ import annotations

import argparse
import gzip
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from transformers import AutoTokenizer

ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""


def _open_jsonl(path: str | os.PathLike):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")


def _load_examples(path: str | os.PathLike, shuffle: bool, seed: int = 0, max_examples: int | None = None):
    examples = []
    with _open_jsonl(path) as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            examples.append(json.loads(line))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(examples)
    return examples


def _batched(iterable: list, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_input", required=True, help="HF model/tokenizer path")
    ap.add_argument("--input_jsonl", required=True, help="train.jsonl or train.jsonl.gz")
    ap.add_argument("--output_bin", required=True, help="output .bin path (uint32)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle examples with seed=0 (same as dataset)")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed (default 0 to match dataset)")
    ap.add_argument("--max_examples", type=int, default=None, help="optional cap for debugging")
    ap.add_argument("--batch_size", type=int, default=1024, help="tokenizer batch size")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_input, use_fast=True)

    # Same token-id logic as your dataset
    eos_token_id = tok.eos_token_id
    if eos_token_id is None:
        eos_token_id = tok.convert_tokens_to_ids("<|end_of_text|>")

    bos_token_id = tok.bos_token_id
    if bos_token_id is None:
        bos_token_id = tok.convert_tokens_to_ids("<|begin_of_text|>")

    # Load + optional shuffle (exactly like your Dataset init)
    examples = _load_examples(
        path=args.input_jsonl,
        shuffle=args.shuffle,
        seed=args.seed,
        max_examples=args.max_examples,
    )

    out_path = Path(args.output_bin)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream-write uint32 tokens to disk (fast + low memory)
    num_examples = 0
    num_tokens = 0

    with open(out_path, "wb") as fout:
        for batch in _batched(examples, args.batch_size):
            texts = [
                ALPACA_TEMPLATE.format(prompt=ex["prompt"], response=ex["response"])
                for ex in batch
            ]

            # Batch tokenize (massive speedup)
            enc = tok(
                texts,
                add_special_tokens=True,          # matches tokenizer.encode(..., add_special_tokens=True)
                return_attention_mask=False,
                return_token_type_ids=False,
                padding=False,
                truncation=False,
            )

            # enc["input_ids"] is a list[list[int]]
            # We append EOS manually per-example, exactly like your dataset.
            # Then write to .bin as uint32.
            for ids in enc["input_ids"]:
                # ids already includes BOS (because add_special_tokens=True), matching your dataset
                if ids and ids[0] != bos_token_id:
                    # Not strictly necessary, but a safety check if tokenizer changes behavior
                    pass

                arr = np.asarray(ids + [eos_token_id], dtype=np.uint32)
                fout.write(arr.tobytes())
                num_tokens += int(arr.size)
                num_examples += 1

    meta = {
        "input_jsonl": str(args.input_jsonl),
        "num_examples": num_examples,
        "num_tokens": num_tokens,
        "dtype": "uint32",
        "bos_token_id": int(bos_token_id),
        "eos_token_id": int(eos_token_id),
        "tokenizer_name_or_path": str(args.model_input),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "tokenizer_batch_size": int(args.batch_size),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote: {out_path} ({num_tokens} tokens, {num_examples} examples)")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
