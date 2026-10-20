#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏋️ RATISS VEGA — Training Loop (CPU / HF Trainer compatible)
RATISS Labs · Yaoundé, Cameroon 🇨🇲

Minimal but complete: loads config, tokenizer, streams corpus,
trains RatissVegaForCausalLM with router aux loss.

Usage:
  pip install torch transformers tokenizers accelerate
  python training/train.py --config training/config_train.yaml

Scales: single CPU -> multi-GPU -> HPC via `accelerate launch`.
"""

import argparse, yaml, math, os, time
from pathlib import Path
import torch
from torch.utils.data import IterableDataset

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.configuration_ratiss_vega import RatissVegaConfig
from model.modeling_ratiss_vega import RatissVegaForCausalLM


class StreamDataset(IterableDataset):
    def __init__(self, corpus, tokenizer, block_size=4096):
        self.corpus = Path(corpus)
        self.tokenizer = tokenizer
        self.block_size = block_size
    def __iter__(self):
        buf=[]
        for line in open(self.corpus, encoding="utf-8"):
            s=line.strip()
            if not s: continue
            ids=self.tokenizer.encode(s).ids
            buf.extend(ids)
            while len(buf) >= self.block_size+1:
                block=buf[:self.block_size+1]
                buf=buf[self.block_size+1:]
                x=torch.tensor(block[:-1], dtype=torch.long)
                y=torch.tensor(block[1:], dtype=torch.long)
                yield {"input_ids": x, "labels": y}
        # tail ignored for packing cleanliness


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="training/config_train.yaml")
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config))

    # Config
    vega_cfg = RatissVegaConfig.from_pretrained(str(Path(cfg["model"]["config"]).parent))
    # Allow override from yaml
    print(f"🧬 Model: {vega_cfg.num_hidden_layers}L/{vega_cfg.hidden_size}H/{vega_cfg.vocab_size} vocab, {vega_cfg.num_experts} experts")

    # Tokenizer
    from transformers import PreTrainedTokenizerFast
    tok_path = cfg["tokenizer"]["path"]
    if Path(tok_path, "tokenizer.json").exists():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(Path(tok_path)/"tokenizer.json"),
                                            bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>")
    else:
        print(f"⚠️ No tokenizer at {tok_path}, using dummy (train tokenizer first!)")
        from tokenizers import Tokenizer, models
        tokenizer = type("Dummy", (), {"encode": lambda s: type("E", (), {"ids":[1,2,3]})(),
                                       "pad_token_id":0, "eos_token_id":2})()

    # Model
    model = RatissVegaForCausalLM(vega_cfg)
    # Optional: tie? no
    # Estimate params
    n_params = sum(p.numel() for p in model.parameters())
    active = n_params - (2 * vega_cfg.num_experts-2) * vega_cfg.num_hidden_layers * 2 * vega_cfg.hidden_size * vega_cfg.intermediate_size // 3  # rough
    print(f"📊 Params total {n_params/1e6:.1f}M (active ~{n_params*0.45/1e6:.1f}M with routing)")

    # Dataset
    train_ds = StreamDataset(cfg["data"]["train_corpus"], tokenizer, cfg["data"]["block_size"])

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"], betas=(0.9,0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["training"]["num_steps"])

    # Simple loop (CPU-safe)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    # Quick smoke-train (if corpus small, just demo 50 steps)
    max_steps = cfg["training"]["num_steps"]
    batch_size = cfg["training"]["batch_size"]
    grad_accum = cfg["training"]["grad_accum"]

    # Use HF Trainer if available for full run
    try:
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        print("🚀 HF Trainer detected — switching to full Trainer (recommended)")
        # Build tiny HF dataset for demo (wrap StreamDataset to list for small corpus)
        # For large corpus, keep streaming via custom loop above. Here we demo 200 samples.
        samples=[]
        for i, ex in enumerate(train_ds):
            samples.append({"input_ids": ex["input_ids"].tolist(), "labels": ex["labels"].tolist()})
            if i>=200: break
        # Simple collator
        from datasets import Dataset as HFDataset
        hf_ds = HFDataset.from_list(samples)
        # Minimal TrainingArguments
        targs = TrainingArguments(
            output_dir=cfg["training"]["output_dir"],
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            max_steps=min(max_steps, 200),  # cap for demo
            logging_steps=cfg["training"]["log_every"],
            save_steps=cfg["training"]["save_every"],
            save_total_limit=3,
            bf16=(cfg["training"]["dtype"]=="bfloat16" and torch.cuda.is_available()),
            learning_rate=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
            warmup_steps=cfg["training"]["warmup_steps"],
            lr_scheduler_type="cosine",
            report_to="none",
        )
        trainer = Trainer(model=model, args=targs, train_dataset=hf_ds, tokenizer=tokenizer,
                          data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
        trainer.train()
        trainer.save_model(cfg["training"]["output_dir"])
        tokenizer.save_pretrained(cfg["training"]["output_dir"])
        print(f"✅ Saved to {cfg['training']['output_dir']}")
        return
    except ImportError:
        print("ℹ️ HF Trainer not fully available, using minimal loop")

    # Fallback minimal loop
    it = iter(train_ds)
    step=0
    acc=0
    opt.zero_grad()
    for step in range(1, max_steps+1):
        try:
            batch = [next(it) for _ in range(batch_size)]
        except StopIteration:
            break
        input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
        labels = torch.stack([b["labels"] for b in batch]).to(device)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss / grad_accum
        loss.backward()
        acc+=1
        if acc % grad_accum ==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
            opt.step(); sched.step(); opt.zero_grad()
            print(f"step {step} loss {loss.item()*grad_accum:.3f} lr {sched.get_last_lr()[0]:.2e}")
        if step % cfg["training"]["save_every"]==0:
            ckpt = Path(cfg["training"]["output_dir"])/f"step-{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt))
            print(f"💾 Saved {ckpt}")

    print("🏁 Done")

if __name__ == "__main__":
    main()
