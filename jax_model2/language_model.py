import jax
import jax.numpy as jnp

from jax_model.ops import layer_norm, linear_cross_entropy
from jax_model.rope import precompute_rope_cache

from .config import resolve_block_layout
from .layers import routed_block_forward, tokenwise_layer_attention


def _forward_one_block(
    block,
    x,
    states,
    cfg,
    kind,
    cos,
    sin,
    *,
    block_index,
    attention_backend,
    span_backend,
    remat_blocks,
):
    def apply_block(b, h, ss):
        return routed_block_forward(
            b,
            h,
            ss,
            cfg,
            kind,
            cos,
            sin,
            block_index=block_index,
            attention_backend=attention_backend,
            span_backend=span_backend,
        )

    return jax.checkpoint(apply_block)(block, x, states) if remat_blocks else apply_block(block, x, states)


def forward_backbone(
    params,
    idx,
    cfg,
    *,
    attention_backend="windowed",
    span_backend="fused",
    remat_blocks=False,
):
    _, T = idx.shape
    assert T <= cfg.block_size
    layout = resolve_block_layout(cfg)
    cos, sin = precompute_rope_cache(cfg.n_embd // cfg.n_head, T, dtype=params["token_embedding"]["weight"].dtype)
    x = params["token_embedding"]["weight"][idx]
    states = (x,)
    for i, (kind, block) in enumerate(zip(layout, params["blocks"])):
        x = _forward_one_block(
            block,
            x,
            states,
            cfg,
            kind,
            cos,
            sin,
            block_index=i,
            attention_backend=attention_backend,
            span_backend=span_backend,
            remat_blocks=remat_blocks,
        )
        states = states + (x,)
    return layer_norm(x, params["ln_f"])


def forward(
    params,
    idx,
    cfg,
    *,
    attention_backend="windowed",
    span_backend="fused",
    remat_blocks=False,
):
    hidden = forward_backbone(
        params,
        idx,
        cfg,
        attention_backend=attention_backend,
        span_backend=span_backend,
        remat_blocks=remat_blocks,
    )
    return hidden @ params["token_embedding"]["weight"].T


def loss(
    params,
    idx,
    targets,
    cfg,
    *,
    attention_backend="windowed",
    span_backend="fused",
    remat_blocks=False,
):
    hidden = forward_backbone(
        params,
        idx,
        cfg,
        attention_backend=attention_backend,
        span_backend=span_backend,
        remat_blocks=remat_blocks,
    )
    return linear_cross_entropy(hidden, params["token_embedding"]["weight"], targets)


def layer_attention_weights(
    params,
    idx,
    cfg,
    *,
    attention_backend="windowed",
    span_backend="fused",
):
    """Return tokenwise depth-routing weights for inspection.

    Each entry has shape [B, num_sources, T] and corresponds to the layer attention
    before that block. The first block has no entry because it only sees embeddings.
    """

    _, T = idx.shape
    layout = resolve_block_layout(cfg)
    cos, sin = precompute_rope_cache(cfg.n_embd // cfg.n_head, T, dtype=params["token_embedding"]["weight"].dtype)
    x = params["token_embedding"]["weight"][idx]
    states = (x,)
    weights = []
    for i, (kind, block) in enumerate(zip(layout, params["blocks"])):
        if cfg.use_layer_attention and i > 0:
            # Inline the score calculation so diagnostics can inspect weights
            # without changing the main forward path.
            from .layers import _select_layer_sources
            from jax_model.ops import layer_norm, linear
            import math

            sources = _select_layer_sources(states, cfg.layer_attn_max_sources)
            bank = jnp.stack(sources, axis=1)
            q = linear(layer_norm(x, block["route"]["ln"]), block["route"]["q_proj"])
            k = linear(layer_norm(bank, block["route"]["ln"]), block["route"]["k_proj"])
            scores = jnp.einsum("btc,bstc->bst", q, k) / math.sqrt(cfg.n_embd)
            weights.append(jax.nn.softmax(scores.astype(jnp.float32), axis=1))
            route = tokenwise_layer_attention(block["route"], x, states, cfg)
            gate = jax.nn.sigmoid(block["route"]["gate"])
            x_in = gate * route + (1.0 - gate) * x
        else:
            x_in = x
        from .layers import horizontal_block_forward

        x = horizontal_block_forward(block, x_in, cfg, kind, cos, sin, attention_backend, span_backend)
        states = states + (x,)
    return weights


def count_parameters(params):
    leaves = []

    def collect(value):
        if isinstance(value, dict):
            for v in value.values():
                collect(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                collect(v)
        elif hasattr(value, "size"):
            leaves.append(value)

    collect(params)
    return sum(int(jnp.size(x)) for x in leaves)
