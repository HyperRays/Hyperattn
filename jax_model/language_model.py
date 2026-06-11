import jax.numpy as jnp

from .layers import block_forward
from .ops import layer_norm, softmax_cross_entropy
from .rope import precompute_rope_cache


def forward_backbone(params, idx, cfg, *, attention_backend="windowed", span_backend="materialized"):
    _, T = idx.shape
    assert T <= cfg.block_size
    cos, sin = precompute_rope_cache(cfg.n_embd // cfg.n_head, T, dtype=params["token_embedding"]["weight"].dtype)
    x = params["token_embedding"]["weight"][idx]
    for block in params["blocks"]:
        x = block_forward(block, x, cfg, cos, sin, attention_backend=attention_backend, span_backend=span_backend)
    return layer_norm(x, params["ln_f"])


def forward(params, idx, cfg, *, attention_backend="windowed", span_backend="materialized"):
    hidden = forward_backbone(params, idx, cfg, attention_backend=attention_backend, span_backend=span_backend)
    return hidden @ params["token_embedding"]["weight"].T


def loss(params, idx, targets, cfg, *, attention_backend="windowed", span_backend="materialized"):
    logits = forward(params, idx, cfg, attention_backend=attention_backend, span_backend=span_backend)
    return softmax_cross_entropy(logits, targets)


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
