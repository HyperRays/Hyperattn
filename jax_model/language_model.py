import jax.numpy as jnp
import jax

from .layers import block_forward, span_stack_forward
from .ops import layer_norm, linear_cross_entropy
from .rope import precompute_rope_cache


def _block_kind(block):
    if "attn" in block:
        return "attn"
    if "edge_proj" in block:
        return "span"
    if "pool_score" in block:
        return "hca"
    return "?"


def _stack_same_structure(blocks):
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *blocks)


def _forward_one_block(block, x, cfg, cos, sin, *, attention_backend, span_backend, remat_blocks):
    def apply_block(b, h):
        return block_forward(b, h, cfg, cos, sin, attention_backend=attention_backend, span_backend=span_backend)

    return jax.checkpoint(apply_block)(block, x) if remat_blocks else apply_block(block, x)


def forward_backbone(
    params,
    idx,
    cfg,
    *,
    attention_backend="windowed",
    span_backend="materialized",
    remat_blocks=False,
    scan_span_runs=False,
):
    _, T = idx.shape
    assert T <= cfg.block_size
    cos, sin = precompute_rope_cache(cfg.n_embd // cfg.n_head, T, dtype=params["token_embedding"]["weight"].dtype)
    x = params["token_embedding"]["weight"][idx]
    i = 0
    while i < len(params["blocks"]):
        block = params["blocks"][i]
        if scan_span_runs and _block_kind(block) == "span":
            j = i + 1
            while j < len(params["blocks"]) and _block_kind(params["blocks"][j]) == "span":
                j += 1
            if j - i > 1:
                x = span_stack_forward(
                    _stack_same_structure(params["blocks"][i:j]),
                    x,
                    cfg,
                    span_backend=span_backend,
                    remat_blocks=remat_blocks,
                )
                i = j
                continue
        x = _forward_one_block(
            block,
            x,
            cfg,
            cos,
            sin,
            attention_backend=attention_backend,
            span_backend=span_backend,
            remat_blocks=remat_blocks,
        )
        i += 1
    return layer_norm(x, params["ln_f"])


def forward(
    params,
    idx,
    cfg,
    *,
    attention_backend="windowed",
    span_backend="materialized",
    remat_blocks=False,
    scan_span_runs=False,
):
    hidden = forward_backbone(
        params,
        idx,
        cfg,
        attention_backend=attention_backend,
        span_backend=span_backend,
        remat_blocks=remat_blocks,
        scan_span_runs=scan_span_runs,
    )
    return hidden @ params["token_embedding"]["weight"].T


def loss(
    params,
    idx,
    targets,
    cfg,
    *,
    attention_backend="windowed",
    span_backend="materialized",
    remat_blocks=False,
    scan_span_runs=False,
):
    hidden = forward_backbone(
        params,
        idx,
        cfg,
        attention_backend=attention_backend,
        span_backend=span_backend,
        remat_blocks=remat_blocks,
        scan_span_runs=scan_span_runs,
    )
    return linear_cross_entropy(hidden, params["token_embedding"]["weight"], targets)


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
