#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧬 RATISS VEGA — Configuration
RATISS Labs · Yaoundé, Cameroon 🇨🇲 · MIT

Calibrated like the star Vega. Sparse by design, not post-hoc.
"""

from transformers.configuration_utils import PretrainedConfig


class RatissVegaConfig(PretrainedConfig):
    model_type = "ratiss_vega"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=960,
        intermediate_size=2560,
        num_hidden_layers=28,
        num_attention_heads=15,
        num_key_value_heads=5,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=100000.0,
        rope_scaling=None,
        # --- VEGA specifics ---
        num_experts=3,
        expert_routing="constellation",
        router_temperature=0.3,
        router_aux_loss_coef=0.01,
        active_expert_capacity="elastic",
        elastic_min=0.60,
        elastic_max=0.85,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # VEGA
        self.num_experts = num_experts
        self.expert_routing = expert_routing
        self.router_temperature = router_temperature
        self.router_aux_loss_coef = router_aux_loss_coef
        self.active_expert_capacity = active_expert_capacity
        self.elastic_min = elastic_min
        self.elastic_max = elastic_max
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
