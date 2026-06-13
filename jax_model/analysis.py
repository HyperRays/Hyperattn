import math

import jax
import jax.numpy as jnp

from .language_model import forward_backbone, loss
from .layers import _compress_blocks, _mask_fill_value, block_forward
from .ops import layer_norm, linear, linear_cross_entropy_tokens
from .rope import apply_rope, apply_rope_at_positions, precompute_rope_cache


def block_kind(block):
    if "attn" in block:
        return "attn"
    if "edge_proj" in block:
        return "span"
    if "pool_score" in block:
        return "hca"
    return "?"


def _ablate_block(params, i):
    """Return a params copy with block i's mixer write zeroed (leaves shared elsewhere)."""
    blocks = list(params["blocks"])
    b = dict(blocks[i])
    if "gate" in b:  # span / hca: sigmoid(gate) -> 0, the block writes nothing
        b["gate"] = jnp.full_like(b["gate"], -30.0)
    elif "attn" in b:  # local attention: zero the output projection
        attn = dict(b["attn"])
        attn["out"] = {**attn["out"], "weight": jnp.zeros_like(attn["out"]["weight"])}
        b["attn"] = attn
    blocks[i] = b
    return {**params, "blocks": blocks}


def block_ablation_deltas(params, idx, targets, cfg, *, attention_backend="chunked", span_backend="materialized"):
    """Per-block importance = rise in loss when each block's mixer write is zeroed.

    Returns (baseline_loss, [(index, kind, delta_loss), ...]). This is the honest
    importance signal — a block's gate magnitude is not (a fully-open gate can still
    contribute nothing). One forward per block, so call it on an eval cadence (e.g.
    every eval_interval), not every step. Only the mixer sublayer is ablated; each
    block's MLP is left in place.
    """
    kw = dict(attention_backend=attention_backend, span_backend=span_backend)
    base = float(loss(params, idx, targets, cfg, **kw))
    deltas = []
    for i, b in enumerate(params["blocks"]):
        d = float(loss(_ablate_block(params, i), idx, targets, cfg, **kw)) - base
        deltas.append((i, block_kind(b), d))
    return base, deltas


def _hca_null_mass(block, x, cfg, cos, sin):
    """Average softmax mass an HCA block puts on its null sink (≈1 = bypassing memory)."""
    B, T, C = x.shape
    H = cfg.n_head
    D = cfg.n_embd // cfg.n_head
    y = layer_norm(x, block["ln1"])
    q = linear(y, block["q_proj"]).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    q = apply_rope(q, cos, sin)
    mem, nb = _compress_blocks(block, y, cfg.compression_block)
    k = linear(mem, block["k_proj"]).reshape(B, nb, H, D).transpose(0, 2, 1, 3)
    block_ends = jnp.minimum((jnp.arange(nb) + 1) * cfg.compression_block - 1, T - 1)
    k = apply_rope_at_positions(k, cos, sin, block_ends)
    null_k = jnp.broadcast_to(block["null_k"], (B, H, 1, D))
    k = jnp.concatenate([null_k, k], axis=2)
    scores = jnp.einsum("bhtd,bhmd->bhtm", q, k) / math.sqrt(D)
    token_pos = jnp.arange(T)[:, None]
    allow = jnp.concatenate([jnp.ones((T, 1), dtype=bool), block_ends[None, :] <= token_pos], axis=1)
    scores = jnp.where(allow[None, None, :, :], scores, _mask_fill_value(scores.dtype))
    weights = jax.nn.softmax(scores, axis=-1)
    return float(jnp.mean(weights[..., 0]))


def backbone_diagnostics(params, idx, cfg, *, attention_backend="chunked", span_backend="materialized"):
    """Residual-stream RMS after each block, plus each HCA block's null-sink mass.

    Residual RMS rising/exploding across depth flags a poorly conditioned deep stack;
    HCA null mass ≈ 1 means the compressed-memory block is bypassed (the dead-block
    failure, seen directly rather than inferred from the gate).
    """
    _, T = idx.shape
    cos, sin = precompute_rope_cache(
        cfg.n_embd // cfg.n_head, T, dtype=params["token_embedding"]["weight"].dtype
    )
    x = params["token_embedding"]["weight"][idx]
    res_rms, null_mass = [], {}
    for i, block in enumerate(params["blocks"]):
        if "pool_score" in block:
            null_mass[i] = _hca_null_mass(block, x, cfg, cos, sin)
        x = block_forward(block, x, cfg, cos, sin, attention_backend=attention_backend, span_backend=span_backend)
        res_rms.append(float(jnp.sqrt(jnp.mean(jnp.square(x)))))
    return {"residual_rms": res_rms, "hca_null_mass": null_mass}


def position_bucketed_loss(
    params, idx, targets, cfg, n_buckets=4, *, attention_backend="chunked", span_backend="materialized"
):
    """Mean next-token loss split into n_buckets along the sequence (early -> late).

    A falling early->late curve means the model exploits longer context; a flat curve
    means its effective context is capped (the long-range capability has not emerged).
    """
    hidden = forward_backbone(params, idx, cfg, attention_backend=attention_backend, span_backend=span_backend)
    tok = linear_cross_entropy_tokens(hidden, params["token_embedding"]["weight"], targets)
    tok = jnp.mean(tok, axis=0)
    T = tok.shape[0]
    edges = [round(b * T / n_buckets) for b in range(n_buckets + 1)]
    return [(edges[b], edges[b + 1], float(jnp.mean(tok[edges[b] : edges[b + 1]]))) for b in range(n_buckets)]
