import math

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)
import torch
import torch.nn.functional as F

def calculate_attention_from_qk(
    model,
    all_hidden_states,
    all_position_ids=None,
    all_attention_mask=None,
    query_indices=None,
):
    qwen_decoder = model.model
    num_layers = len(qwen_decoder.layers)
    all_timesteps_attention = []

    for t, hs_per_layer in enumerate(all_hidden_states):
        bsz, seq_len, _ = hs_per_layer[0].shape

        if query_indices is None:
            q_idx = [seq_len - 1]
        else:
            q_idx = query_indices

        if all_position_ids is not None:
            if torch.is_tensor(all_position_ids):
                position_ids = all_position_ids
            else:
                position_ids = all_position_ids[t]
        cos, sin = qwen_decoder.rotary_emb(hs_per_layer[0], position_ids)  # cos/sin: (3, bsz, seq_len, head_dim_part) 展开后会对齐

        # mask
        if all_attention_mask is not None:
            if torch.is_tensor(all_attention_mask):
                attn_mask_2d = all_attention_mask
            else:
                attn_mask_2d = all_attention_mask[t]
            orig_impl = qwen_decoder.config._attn_implementation
            qwen_decoder.config._attn_implementation = "eager"
            causal_mask = qwen_decoder._update_causal_mask(
                attn_mask_2d,
                hs_per_layer[0],
                cache_position=torch.arange(seq_len, device=hs_per_layer[0].device),
                past_key_values=None,
                output_attentions=True,
            )
            qwen_decoder.config._attn_implementation = orig_impl
        else:
            causal_mask = None

        timestep_attns = []

        for layer_idx in range(num_layers):
            layer = qwen_decoder.layers[layer_idx]
            self_attn = layer.self_attn

            layer_input = hs_per_layer[layer_idx]
            layer_input = layer.input_layernorm(layer_input)
            layer_input_q = layer_input[:, q_idx, :]           # (bsz, Q, hidden)

            k_proj = self_attn.k_proj(layer_input)
            q_proj = self_attn.q_proj(layer_input_q)           # (bsz, Q, num_heads*head_dim)
            # q = q_proj.view(bsz, len(q_idx), -1, self_attn.head_dim).transpose(1, 2)

            k = k_proj.view(bsz, seq_len, -1, self_attn.head_dim).transpose(1, 2)  # (bsz, kv_heads, N, d)
            q = q_proj.view(bsz, len(q_idx), -1, self_attn.head_dim).transpose(1, 2)  # (bsz, n_heads, Q, d)

            k = repeat_kv(k, self_attn.num_key_value_groups)  

            # RoPE
            k, _ = apply_multimodal_rotary_pos_emb(
                k, k.clone(), cos, sin, self_attn.rope_scaling["mrope_section"]
            )
            cos_q = cos[:, :, q_idx, :]
            sin_q = sin[:, :, q_idx, :]

            q, _ = apply_multimodal_rotary_pos_emb(
                q, q.clone(), cos_q, sin_q, self_attn.rope_scaling["mrope_section"])

            #  q: (bsz, n_heads, Q, d), k: (bsz, n_heads, N, d)

            attn_scores = torch.matmul(
                q, k.transpose(-2, -1)
            ) / math.sqrt(self_attn.head_dim)  # (bsz, n_heads, Q, N)

            if causal_mask is not None:
                attn_scores = attn_scores + causal_mask[:, :, q_idx, :].to(attn_scores.dtype)

            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            p = getattr(self_attn, "attention_dropout", None)
            if p is None:
                p = getattr(qwen_decoder.config, "attention_dropout", 0.0)
            attn_weights = F.dropout(attn_weights, p=p, training=model.training)
            timestep_attns.append(attn_weights)

        all_timesteps_attention.append(timestep_attns)

    return all_timesteps_attention