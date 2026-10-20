#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔤 RATISS VEGA — Tokenizer Training (BPE)
RATISS Labs · Yaoundé, Cameroon 🇨🇲

Bilingual FR/EN + code + Cameroon languages (basaa, ewondo pidgin samples)
32k vocab, byte-level BPE, compatible HF tokenizers.
Input: text corpus (one sentence per line). Output: tokenizer.json + HF files.
"""

import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers

def train(corpus: str, out_dir: str, vocab_size=32000):
    corpus = Path(corpus)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"📚 Training BPE {vocab_size} on {corpus} ...")
    tok.train([str(corpus)], trainer=trainer)

    # Save
    tok.save(str(out_dir / "tokenizer.json"))
    print(f"✅ Saved to {out_dir}/tokenizer.json")

    # Also save HF wrapper
    try:
        from transformers import PreTrainedTokenizerFast
        hf = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>", mask_token="<mask>",
            model_max_length=4096,
        )
        hf.save_pretrained(str(out_dir))
        print(f"✅ HF tokenizer saved to {out_dir}/")
    except Exception as e:
        print(f"⚠️ HF save skipped: {e}")

    # Quick test
    test = tok.encode("Bonjour le monde! The capital of France is Paris. Yaoundé 🇨🇲")
    print(f"🧪 Test encode: {test.tokens[:20]} ... ids {test.ids[:12]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="path to .txt corpus (one doc per line)")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--vocab_size", type=int, default=32000)
    args = ap.parse_args()
    train(args.corpus, args.out, args.vocab_size)
