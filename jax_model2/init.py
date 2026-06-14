import jax
import jax.numpy as jnp

from .config import resolve_block_layout, span_specs_for_kind


def _split(key, n):
    return jax.random.split(key, n)


def _linear(key, out_dim, in_dim, std, bias=True):
    params = {"weight": std * jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32)}
    if bias:
        params["bias"] = jnp.zeros((out_dim,), dtype=jnp.float32)
    return params


def _identity_linear(dim):
    return {"weight": jnp.eye(dim, dtype=jnp.float32)}


def _layer_norm(dim):
    return {
        "weight": jnp.ones((dim,), dtype=jnp.float32),
        "bias": jnp.zeros((dim,), dtype=jnp.float32),
    }


def _mlp(key, cfg):
    k1, k2 = _split(key, 2)
    return {
        "fc": _linear(k1, 4 * cfg.n_embd, cfg.n_embd, cfg.initializer_std),
        "proj": _linear(k2, cfg.n_embd, 4 * cfg.n_embd, cfg.initializer_std),
    }


def _route(key, cfg):
    kq, kk, kw = _split(key, 3)
    route = {
        "ln": _layer_norm(cfg.n_embd),
        "q_proj": _linear(kq, cfg.n_embd, cfg.n_embd, cfg.initializer_std, bias=False),
        "k_proj": _linear(kk, cfg.n_embd, cfg.n_embd, cfg.initializer_std, bias=False),
        # Identity value/output projections make the initial router a scale-preserving
        # average over prior layer states instead of a near-zero random projection.
        "v_proj": _identity_linear(cfg.n_embd),
        "out_proj": _identity_linear(cfg.n_embd),
        "gate": jnp.asarray(cfg.route_gate_init, dtype=jnp.float32),
    }
    if cfg.use_memory_router:
        # Learned softmax write that folds this block's output into the streaming memory
        # (slots 1..layer_memory_slots-1; slot 0 is the embedding).
        route["write_score"] = _linear(kw, cfg.layer_memory_slots - 1, cfg.n_embd, cfg.initializer_std)
    return route


def _attn_block(key, cfg):
    kr, ka, ko, km = _split(key, 4)
    return {
        "route": _route(kr, cfg),
        "ln1": _layer_norm(cfg.n_embd),
        "attn": {
            "qkv": _linear(ka, 3 * cfg.n_embd, cfg.n_embd, cfg.initializer_std, bias=False),
            "out": _linear(ko, cfg.n_embd, cfg.n_embd, cfg.initializer_std, bias=False),
        },
        "ln2": _layer_norm(cfg.n_embd),
        "mlp": _mlp(km, cfg),
    }


def _span_block(key, cfg, kind):
    kr, ki, ke, ko, km = _split(key, 5)
    specs = span_specs_for_kind(cfg, kind)
    return {
        "route": _route(kr, cfg),
        "ln1": _layer_norm(cfg.n_embd),
        "in_proj": _linear(ki, cfg.n_embd, cfg.n_embd, cfg.initializer_std),
        "edge_proj": _linear(ke, cfg.n_embd, cfg.n_embd * len(specs), cfg.initializer_std),
        "out_proj": _linear(ko, cfg.n_embd, cfg.n_embd, cfg.initializer_std),
        "gate": jnp.asarray(-3.0, dtype=jnp.float32),
        "ln2": _layer_norm(cfg.n_embd),
        "mlp": _mlp(km, cfg),
    }


def init_params(key, cfg):
    layout = resolve_block_layout(cfg)
    keys = _split(key, len(layout) + 2)
    params = {
        "token_embedding": {
            "weight": cfg.initializer_std
            * jax.random.normal(keys[0], (cfg.vocab_size, cfg.n_embd), dtype=jnp.float32)
        },
        "blocks": [],
        "ln_f": _layer_norm(cfg.n_embd),
    }
    if cfg.use_memory_router:
        assert cfg.layer_memory_slots >= 2, "layer_memory_slots must be >= 2 (embedding + >=1 memory slot)"
        # Per-slot init for the streaming memory (slots 1..n-1); slot 0 is the embedding at runtime.
        params["mem_init"] = cfg.initializer_std * jax.random.normal(
            keys[-1], (cfg.layer_memory_slots - 1, cfg.n_embd), dtype=jnp.float32
        )
    for kind, block_key in zip(layout, keys[1:-1]):
        if kind == "attn":
            params["blocks"].append(_attn_block(block_key, cfg))
        else:
            params["blocks"].append(_span_block(block_key, cfg, kind))
    return params
