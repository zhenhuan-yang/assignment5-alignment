from __future__ import annotations

import argparse
import gzip
import json
import os
import random
from pathlib import Path

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


def _load_hh_examples(path: str | os.PathLike, shuffle: bool, seed: int = 0, max_examples: int | None = None):
    """Load HH dataset from a single jsonl.gz file."""
    examples = []
    
    with _open_jsonl(path) as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            
            item = json.loads(line)
            chosen = item["chosen"]
            rejected = item["rejected"]
            
            # Filter multi-turn: Human speaks more than once
            if chosen.count("\n\nHuman:") > 1:
                continue
            
            # Parse Instruction (everything before the first Assistant response)
            try:
                parts = chosen.split("\n\nAssistant:", 1)
                instruction = parts[0].replace("\n\nHuman:", "").strip()
                chosen_response = parts[1].strip()
                rejected_response = rejected.split("\n\nAssistant:", 1)[1].strip()
                
                examples.append({
                    "instruction": instruction,
                    "chosen_response": chosen_response,
                    "rejected_response": rejected_response
                })
            except IndexError:
                continue
    
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
    ap.add_argument("--input_jsonl", required=True, help="HH train.jsonl or train.jsonl.gz")
    ap.add_argument("--output_bin_prefix", required=True, help="output directory prefix (e.g., data/dpo_packed/harmless-base/)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle examples")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed")
    ap.add_argument("--max_examples", type=int, default=None, help="optional cap")
    ap.add_argument("--batch_size", type=int, default=1024, help="tokenizer batch size")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_input, use_fast=True)

    eos_token_id = tok.eos_token_id
    if eos_token_id is None:
        eos_token_id = tok.convert_tokens_to_ids("<|end_of_text|>")

    bos_token_id = tok.bos_token_id
    if bos_token_id is None:
        bos_token_id = tok.convert_tokens_to_ids("<|begin_of_text|>")

    # Load HH dataset
    examples = _load_hh_examples(args.input_jsonl, args.shuffle, args.seed, args.max_examples)
    print(f"Loaded {len(examples)} preference pairs")

    # Create output paths
    prefix = Path(args.output_bin_prefix)
    prefix.mkdir(parents=True, exist_ok=True)
    
    chosen_path = prefix / "train.chosen.bin"
    rejected_path = prefix / "train.rejected.bin"

    num_examples = 0
    num_tokens_chosen = 0
    num_tokens_rejected = 0

    with open(chosen_path, "wb") as f_chosen, open(rejected_path, "wb") as f_rejected:
        for batch in _batched(examples, args.batch_size):
            chosen_texts = [
                ALPACA_TEMPLATE.format(prompt=ex["instruction"], response=ex["chosen_response"])
                for ex in batch
            ]
            rejected_texts = [
                ALPACA_TEMPLATE.format(prompt=ex["instruction"], response=ex["rejected_response"])
                for ex in batch
            ]

            enc_chosen = tok(chosen_texts, add_special_tokens=True, return_attention_mask=False,
                           return_token_type_ids=False, padding=False, truncation=False)
            enc_rejected = tok(rejected_texts, add_special_tokens=True, return_attention_mask=False,
                             return_token_type_ids=False, padding=False, truncation=False)

            # Write pairs in aligned order
            for chosen_ids, rejected_ids in zip(enc_chosen["input_ids"], enc_rejected["input_ids"]):
                # Write chosen + EOS
                chosen_arr = np.asarray(chosen_ids + [eos_token_id], dtype=np.uint32)
                f_chosen.write(chosen_arr.tobytes())
                num_tokens_chosen += len(chosen_arr)
                
                # Write rejected + EOS
                rejected_arr = np.asarray(rejected_ids + [eos_token_id], dtype=np.uint32)
                f_rejected.write(rejected_arr.tobytes())
                num_tokens_rejected += len(rejected_arr)
                
                num_examples += 1

    meta = {
        "input_jsonl": str(args.input_jsonl),
        "num_examples": num_examples,
        "num_chosen_tokens": num_tokens_chosen,
        "num_rejected_tokens": num_tokens_rejected,
        "dtype": "uint32",
        "bos_token_id": int(bos_token_id),
        "eos_token_id": int(eos_token_id),
        "tokenizer_name_or_path": str(args.model_input),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "tokenizer_batch_size": int(args.batch_size),
    }
    
    meta_path = prefix / "train.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    
    print(f"Wrote: {chosen_path} ({num_tokens_chosen} tokens)")
    print(f"Wrote: {rejected_path} ({num_tokens_rejected} tokens)")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
