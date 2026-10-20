#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📦 RATISS VEGA — Dataset
Streams text corpus, tokenizes, packs to 4096 blocks.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset

class PackedDataset(Dataset):
    def __init__(self, tokenized_file: str, block_size=4096):
        # tokenized_file: .bin of int32 tokens (pre-tokenized for speed)
        # For starter, we load raw text and tokenize on fly if needed.
        self.data = torch.from_file(tokenized_file, dtype=torch.int32) if Path(tokenized_file).exists() and tokenized_file.endswith(".bin") else None
        self.block_size = block_size
        if self.data is not None:
            self.n_blocks = len(self.data) // block_size

    def __len__(self):
        # Placeholder: will be overridden by streaming loader
        return 10000

    def __getitem__(self, idx):
        # Dummy for template — real loader uses iterable dataset
        raise NotImplementedError("Use training/train.py streaming loader")


def streaming_dataloader(tokenizer, corpus_path, block_size=4096, batch_size=4):
    """
    Simple streaming: reads lines, tokenizes, buffers, yields packed batches.
    CPU-friendly, no HF datasets required for minimal run.
    """
    from pathlib import Path
    buf = []
    batch = []
    for line in open(corpus_path, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        ids = tokenizer.encode(line, add_special_tokens=False)
        buf.extend(ids.ids if hasattr(ids,"ids") else ids)
        # emit blocks
        while len(buf) >= block_size:
            block = buf[:block_size]
            buf = buf[block_size:]
            batch.append(torch.tensor(block, dtype=torch.long))
            if len(batch) == batch_size:
                yield torch.stack(batch)  # (B,T)
                batch=[]
    # flush
    if batch:
        yield torch.stack(batch)
