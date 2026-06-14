import math

import jax
import jax.numpy as jnp

from jax_model.layers import local_attention_forward, mlp_forward
from jax_model.ops import gelu, layer_norm, linear

from .config import span_specs_for_kind
from .span import delayed_span_hypergraph, fused_delayed_span_edge_projection


def _select_layer_sources(states, max_sources):
    if max_sources is None or len(states) <= max_sources:
        return tuple(states)
    if max_sources <= 1:
        return (states[-1],)
    return (states[0],) + tuple(states[-(max_sources - 1) :])


def tokenwise_layer_attention(params, x_prev, states, cfg):
    sources = _select_layer_sources(states, cfg.layer_attn_max_sources)
    bank = jnp.stack(sources, axis=1)  # [B, S, T, C]
    q = linear(layer_norm(x_prev, params["ln"]), params["q_proj"])
    bank_norm = layer_norm(bank, params["ln"])
    k = linear(bank_norm, params["k_proj"])
    v = linear(bank, params["v_proj"])
    scores = jnp.einsum("btc,bstc->bst", q, k) / math.sqrt(cfg.n_embd)
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=1).astype(x_prev.dtype)
    routed = jnp.einsum("bst,bstc->btc", weights, v)
    return linear(routed, params["out_proj"])


def route_block_input(params, x_prev, states, cfg, *, block_index):
    if not cfg.use_layer_attention or block_index == 0:
        return x_prev
    routed = tokenwise_layer_attention(params["route"], x_prev, states, cfg)
    route_gate = jax.nn.sigmoid(params["route"]["gate"])
    return route_gate * routed + (1.0 - route_gate) * x_prev


def attention_block_forward(params, x, cfg, cos, sin, attention_backend="windowed"):
    x = x + local_attention_forward(params["attn"], layer_norm(x, params["ln1"]), cfg, cos, sin, attention_backend)
    x = x + mlp_forward(params["mlp"], layer_norm(x, params["ln2"]))
    return x


def span_block_forward(params, x, cfg, kind, span_backend="fused"):
    residual = x
    h = linear(layer_norm(x, params["ln1"]), params["in_proj"])
    specs = span_specs_for_kind(cfg, kind)
    if span_backend == "materialized":
        z = delayed_span_hypergraph(h, specs)
        z = linear(z, params["edge_proj"])
    elif span_backend == "fused":
        z = fused_delayed_span_edge_projection(h, specs, params["edge_proj"])
    else:
        raise ValueError(f"unknown JAX model2 span backend: {span_backend}")
    z = gelu(z)
    z = linear(z, params["out_proj"])
    x = residual + jax.nn.sigmoid(params["gate"]) * z
    x = x + mlp_forward(params["mlp"], layer_norm(x, params["ln2"]))
    return x


def horizontal_block_forward(params, x, cfg, kind, cos, sin, attention_backend="windowed", span_backend="fused"):
    if kind == "attn":
        return attention_block_forward(params, x, cfg, cos, sin, attention_backend)
    if kind in {"far_span", "mid_span", "local_span"}:
        return span_block_forward(params, x, cfg, kind, span_backend)
    raise ValueError(f"unknown model2 block kind: {kind}")


def routed_block_forward(
    params,
    x_prev,
    states,
    cfg,
    kind,
    cos,
    sin,
    *,
    block_index,
    attention_backend="windowed",
    span_backend="fused",
):
    x = route_block_input(params, x_prev, states, cfg, block_index=block_index)
    return horizontal_block_forward(params, x, cfg, kind, cos, sin, attention_backend, span_backend)
