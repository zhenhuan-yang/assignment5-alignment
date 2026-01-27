from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import gzip
import json
import random
import os
import numpy as np


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""

class PackedSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # 1. Load Data
        examples = []
        # Handle both .gz and plain .jsonl
        if str(dataset_path).endswith('.gz'):
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'
            
        with open_func(dataset_path, mode, encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
                
        # 2. Shuffle if requested
        if shuffle:
            rng = random.Random(0)
            rng.shuffle(examples)

        # 3. Get Special Token IDs
        # Llama 3 specific: <|end_of_text|> is usually the EOS
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")

        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")

        # 4. Format and Tokenize
        all_token_ids = []
        
        for ex in examples:
            text = ALPACA_TEMPLATE.format(prompt=ex["prompt"], response=ex["response"])
            # Adds the [BOS]
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            all_token_ids.extend(tokens)
            # Mannual adds [EOS]
            all_token_ids.append(eos_token_id)
            
        # 5. Pack into chunks
        total_tokens = len(all_token_ids)
        # The very last token in your stream can be a label, but it can never be an input because there is no token after it to serve as its label.
        num_chunks = max(0, (total_tokens - 1) // seq_length)

        # Drop the remainder
        cutoff = num_chunks * seq_length
        self.input_ids = torch.tensor(all_token_ids[:cutoff], dtype=torch.long)
        self.labels = torch.tensor(all_token_ids[1:cutoff+1], dtype=torch.long)
        
        self.input_ids = self.input_ids.view(num_chunks, seq_length)
        self.labels = self.labels.view(num_chunks, seq_length)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx].clone(),
            "labels": self.labels[idx].clone()
        }


class MemmapPackedDataset(Dataset):
    """
    Reads a contiguous uint32 token stream from .bin and returns fixed-length chunks:
      input_ids: tokens[t : t+seq_length]
      labels:    tokens[t+1 : t+1+seq_length]
    """
    def __init__(self, bin_path: str, seq_length: int):
        self.seq_length = seq_length
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")

        total_tokens = len(self.data)
        self.num_chunks = max(0, (total_tokens - 1) // seq_length)  # must have label for last input
        self.cutoff = self.num_chunks * seq_length  # last input index is cutoff-1, label uses cutoff

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx: int):
        start = idx * self.seq_length
        x_np = np.asarray(self.data[start : start + self.seq_length], dtype=np.int64)
        y_np = np.asarray(self.data[start + 1 : start + 1 + self.seq_length], dtype=np.int64)
        return {"input_ids": torch.from_numpy(x_np), "labels": torch.from_numpy(y_np)}
