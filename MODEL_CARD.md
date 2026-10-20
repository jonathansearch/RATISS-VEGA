# RATISS VEGA — Model Card (Base Architecture)

**Version:** 0.1 — Design + Code, not yet trained  
**Lab:** RATISS Labs, Yaoundé, Cameroon 🇨🇲  
**License:** MIT  
**Base:** Decoder-only Transformer, GQA + RoPE, SwiGLU, Constellation MoE

## Architecture at a Glance

| Hyperparameter | VEGA-360M (prod) | VEGA-test (smoke) |
|---|---|---:|
| Layers | 28 | 4 |
| Hidden | 960 | 960 |
| Heads / KV | 15 / 5 | 15 / 5 |
| Head dim | 64 | 64 |
| Intermediate (per expert) | 2560 | 2560 |
| Experts / layer | 3 (Q/A/O) | 3 |
| Active per token | ~1 expert (soft mix) | same |
| Vocab | 32k BPE | 32k |
| Context | 4096 | 4096 |
| RoPE theta | 100k | 100k |
| Params total | ~410M | ~58M |
| Params active | ~185M | ~26M |

**Why 410M total?** 3 experts × 28 layers × 2×960×2560 ≈ 3×138M = dense 360M equivalent with extra capacity, but routed.

## From RATISS-X Lessons

- **X:** Post-hoc slicing + magnitude elite failed at 95% in generation (looping).
- **VEGA:** Sparse *by design* — router trained jointly, no post-hoc surgery. Activation-guided elite is baked into router's auxiliary loss.
- **X:** Single GPS centroid collapsed (93% Q vs 30% A).
- **VEGA:** 3 native experts, each specialized by data mix, router learns to blend — verified per token.

## Tokenizer

- **Type:** Byte-level BPE, 32k
- **Script:** `tokenizer/train_tokenizer.py`
- **Corpus:** FR/EN bilingual + code + Cameroon pidgin samples (`corpus_sample.txt` is starter; replace with 10–100B tokens)
- **Specials:** `<pad> <s> </s> <unk> <mask>`
- **Chat:** No chat template enforced in base; add in instruct finetune.

## Training Pipeline

1. **Tokenizer:** `python tokenizer/train_tokenizer.py --corpus data/big.txt --out tokenizer/`
2. **Test model:** `python scripts/test_forward.py` (CPU, <1GB)
3. **Train:** `python training/train.py --config training/config_train.yaml`
4. **Convert:** `python gguf/convert_to_gguf.py --model checkpoints/vega-360M --out gguf/vega-360M-f16.gguf`
5. **Quantize:** `llama-quantize gguf/vega-360M-f16.gguf gguf/vega-q4_k_m.gguf q4_k_m`

## File Map for Training

```
model/
  configuration_ratiss_vega.py   # Config class
  modeling_ratiss_vega.py        # Full transformer (attention+GQA, RoPE, 3-expert MLP)
  config.json                    # 28L/960H/32k production config

tokenizer/
  train_tokenizer.py             # BPE trainer
  corpus_sample.txt               # Starter (replace)
  tokenizer_config.json

training/
  config_train.yaml              # LR, batch, steps, router aux coef
  train.py                       # HF Trainer + fallback loop
  dataset.py                     # Streaming loader

gguf/
  convert_to_gguf.py             # HF→GGUF wrapper (llama.cpp compatible)
scripts/
  test_forward.py                # Smoke test
```

> **GGUF Next-Gen:** Output is standard GGUF v3, loadable in `llama.cpp`, `ollama`, `koboldcpp`. Experts are stored as `blk.0.ffn_gate_expert0.weight` etc., router as `blk.0.ffn_gate_inp.weight` — already supported by llama.cpp MoE path.
