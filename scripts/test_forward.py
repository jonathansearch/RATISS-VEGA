#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 RATISS VEGA — Smoke Test (CPU, <1GB)
Checks: config loads, model forward, generation, router traces.
Run: python scripts/test_forward.py
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.configuration_ratiss_vega import RatissVegaConfig
from model.modeling_ratiss_vega import RatissVegaForCausalLM

def main():
    print("🧬 Loading config...")
    cfg = RatissVegaConfig(
        vocab_size=32000,
        hidden_size=960,
        intermediate_size=2560,
        num_hidden_layers=4,  # tiny for test (28 in production)
        num_attention_heads=15,
        num_key_value_heads=5,
        num_experts=3,
    )
    print(f"  {cfg.num_hidden_layers}L/{cfg.hidden_size}H/{cfg.num_experts} experts, {cfg.max_position_embeddings} ctx")
    model = RatissVegaForCausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Params (4L test): {n_params/1e6:.2f}M — prod 28L ~ {n_params*7/1e6:.1f}M")

    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
        out = model(input_ids=input_ids)
        print(f"✅ Forward logits: {out.logits.shape} (2,16,{cfg.vocab_size}) loss n/a")

        # Generation-style: check KV cache
        out2 = model.model(input_ids, use_cache=True, output_router=True)
        print(f"✅ KV cache: {len(out2['past_key_values'])} layers, router traces sample: {out2['router_traces'][0][:2]}")

        # Tiny generate
        ids = torch.tensor([[1, 423, 789]], dtype=torch.long)  # bos + dummy
        for step in range(5):
            logits = model(ids).logits[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        print(f"✅ Greedy 5 tokens: {ids[0].tolist()}")

    print("\n🎉 Smoke test passed — model is train-ready!")
    print("Next: python training/train.py --config training/config_train.yaml")

if __name__ == "__main__":
    main()
