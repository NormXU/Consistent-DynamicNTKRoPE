# -*- coding:utf-8 -*-
# create: @time: 8/7/23 15:49
"""
This script is used to evaluate the execution time of apply_rotary_pos_emb in the context of consistent DynamicNTKScale RoPE on LLaMA-7B (32 layers).
Unlike the huggingface implementation, where RoPE is applied to a single key_state resulting in the "inconsistent problem",
the consistent DynamicNTKScale RoPE in this repo caches all keys before applying RoPE to a length-increasing key_states list, which, therefore, takes more time on RoPE

"""

import time, torch
from transformers.models.llama.modeling_llama import rotate_half, LlamaDynamicNTKScalingRotaryEmbedding


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    handler_start_time = time.time()
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    request_cost = round((time.time() - handler_start_time) * 1000, 2)
    print("seq_length:{}; exec time: {} ms".format(k_embed.shape[2], request_cost * 32))
    return q_embed, k_embed


if __name__ == '__main__':
    rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
        128, max_position_embeddings=1024, scaling_factor=1.0
    )
    query_states = torch.rand((1, 32, 1, 128))
    for seq_len in range(16, 1024, 16):
        key_states = torch.rand((1, 32, seq_len, 128))
        cos, sin = rotary_emb(query_states, seq_len=seq_len)
        position_ids = torch.tensor([[seq_len - 1]]).to(dtype=torch.long)
        apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
