import math

import jax
import jax.numpy as jnp

from .ops import gelu, layer_norm, linear
from .rope import apply_rope, apply_rope_at_positions, apply_rope_bthd
from .span import einsum_fused_span_edge_projection, fused_span_edge_projection, span_hypergraph


def _mask_fill_value(dtype):
    return jnp.asarray(jnp.finfo(dtype).min, dtype=dtype)


def mlp_forward(params, x):
    x = linear(x, params["fc"])
    x = gelu(x)
    return linear(x, params["proj"])


def _project_qkv(params, x, cfg, cos, sin):
    B, T, C = x.shape
    H = cfg.n_head
    D = cfg.n_embd // cfg.n_head
    qkv = linear(x, params["qkv"]).reshape(B, T, 3, H, D)
    q = apply_rope_bthd(qkv[:, :, 0], cos, sin)
    k = apply_rope_bthd(qkv[:, :, 1], cos, sin)
    v = qkv[:, :, 2]
    return q, k, v


def manual_local_attention_forward(params, x, cfg, cos, sin):
    B, T, C = x.shape
    D = cfg.n_embd // cfg.n_head
    q, k, v = _project_qkv(params, x, cfg, cos, sin)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(D)
    pos = jnp.arange(T)
    i = pos[:, None]
    j = pos[None, :]
    mask = (j <= i) & ((i - j) < cfg.local_window)
    scores = jnp.where(mask[None, None, :, :], scores, _mask_fill_value(scores.dtype))
    weights = jax.nn.softmax(scores, axis=-1)
    y = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
    return linear(y, params["out"])


def windowed_local_attention_forward(params, x, cfg, cos, sin):
    B, T, C = x.shape
    q, k, v = _project_qkv(params, x, cfg, cos, sin)

    output_dtype = q.dtype
    if output_dtype == jnp.float16 and jax.default_backend() == "METAL":
        q = q.astype(jnp.float32)
        k = k.astype(jnp.float32)
        v = v.astype(jnp.float32)

    y = jax.nn.dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        local_window_size=(cfg.local_window - 1, 0),
        implementation=None,
    )
    y = y.astype(output_dtype)
    y = y.reshape(B, T, C)
    return linear(y, params["out"])


def chunked_local_attention_forward(params, x, cfg, cos, sin):
    B, T, C = x.shape
    H = cfg.n_head
    D = cfg.n_embd // cfg.n_head
    w = cfg.local_window
    q, k, v = _project_qkv(params, x, cfg, cos, sin)

    pad = (-T) % w
    if pad:
        q = jnp.pad(q, ((0, 0), (0, pad), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, pad), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, pad), (0, 0), (0, 0)))
    n = (T + pad) // w

    # Flatten (B, n, H) into one leading batch axis so the attention matmuls
    # (and their gradients) are plain 3D batched matmuls; jax-metal fails to
    # lower dot_general with multiple or non-leading batch dimensions.
    qc = q.reshape(B, n, w, H, D).transpose(0, 1, 3, 2, 4)
    kc = k.reshape(B, n, w, H, D).transpose(0, 1, 3, 2, 4)
    vc = v.reshape(B, n, w, H, D).transpose(0, 1, 3, 2, 4)
    k_prev = jnp.pad(kc[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
    v_prev = jnp.pad(vc[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
    kk = jnp.concatenate([k_prev, kc], axis=3)
    vv = jnp.concatenate([v_prev, vc], axis=3)

    qc = qc.reshape(B * n * H, w, D)
    kk = kk.reshape(B * n * H, 2 * w, D)
    vv = vv.reshape(B * n * H, 2 * w, D)
    scores = (qc @ kk.swapaxes(-1, -2) / math.sqrt(D)).reshape(B, n, H, w, 2 * w)

    # key offset m in [0, 2w) maps to global position chunk*w + m - w; a query at
    # offset a sees exactly the keys with 0 <= i - j < w, i.e. a < m <= a + w.
    a = jnp.arange(w)[:, None]
    m = jnp.arange(2 * w)[None, :]
    rel = (m > a) & (m <= a + w)
    j_global = jnp.arange(n)[:, None, None] * w + (m[None] - w)
    key_valid = (j_global >= 0) & (j_global < T)
    mask = rel[None] & key_valid
    scores = jnp.where(mask[None, :, None], scores, _mask_fill_value(scores.dtype))

    weights = jax.nn.softmax(scores, axis=-1).reshape(B * n * H, w, 2 * w)
    y = (weights @ vv).reshape(B, n, H, w, D)
    y = y.transpose(0, 1, 3, 2, 4).reshape(B, n * w, C)[:, :T]
    return linear(y, params["out"])


def local_attention_forward(params, x, cfg, cos, sin, attention_backend="windowed"):
    if attention_backend == "manual":
        return manual_local_attention_forward(params, x, cfg, cos, sin)
    if attention_backend == "windowed":
        return windowed_local_attention_forward(params, x, cfg, cos, sin)
    if attention_backend == "chunked":
        return chunked_local_attention_forward(params, x, cfg, cos, sin)
    raise ValueError(f"unknown JAX attention backend: {attention_backend}")


def local_attention_block_forward(params, x, cfg, cos, sin, attention_backend="windowed"):
    x = x + local_attention_forward(params["attn"], layer_norm(x, params["ln1"]), cfg, cos, sin, attention_backend)
    x = x + mlp_forward(params["mlp"], layer_norm(x, params["ln2"]))
    return x


def span_block_forward(params, x, cfg, span_backend="materialized"):
    residual = x
    h = linear(layer_norm(x, params["ln1"]), params["in_proj"])
    if span_backend == "materialized":
        z = span_hypergraph(h, cfg.span_widths)
        z = linear(z, params["edge_proj"])
    elif span_backend == "fused":
        z = fused_span_edge_projection(h, cfg.span_widths, params["edge_proj"])
    elif span_backend == "einsum_fused":
        z = einsum_fused_span_edge_projection(h, cfg.span_widths, params["edge_proj"])
    else:
        raise ValueError(f"unknown JAX span backend: {span_backend}")
    z = gelu(z)
    z = linear(z, params["out_proj"])
    x = residual + jax.nn.sigmoid(params["gate"]) * z
    x = x + mlp_forward(params["mlp"], layer_norm(x, params["ln2"]))
    return x


def span_stack_forward(params, x, cfg, span_backend="materialized", remat_blocks=False):
    def step(h, block_params):
        def forward_one(p, y):
            return span_block_forward(p, y, cfg, span_backend)

        h = jax.checkpoint(forward_one)(block_params, h) if remat_blocks else forward_one(block_params, h)
        return h, None

    x, _ = jax.lax.scan(step, x, params)
    return x


def _compress_blocks(params, y, compression_block):
    B, T, C = y.shape
    cb = compression_block
    pad_len = (-T) % cb
    if pad_len:
        y_pad = jnp.pad(y, ((0, 0), (0, pad_len), (0, 0)))
    else:
        y_pad = y
    Tp = y_pad.shape[1]
    nb = Tp // cb
    chunks = y_pad.reshape(B, nb, cb, C)
    valid = jnp.arange(Tp).reshape(nb, cb) < T
    score = linear(chunks, params["pool_score"]).squeeze(-1)
    score = jnp.where(valid[None, :, :], score, _mask_fill_value(score.dtype))
    weight = jax.nn.softmax(score, axis=-1)
    mem = jnp.sum(weight[..., None] * chunks, axis=2)
    return mem, nb


def compressed_memory_block_forward(params, x, cfg, cos, sin):
    B, T, C = x.shape
    H = cfg.n_head
    D = cfg.n_embd // cfg.n_head
    residual = x
    y = layer_norm(x, params["ln1"])

    q = linear(y, params["q_proj"]).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    q = apply_rope(q, cos, sin)

    mem, nb = _compress_blocks(params, y, cfg.compression_block)
    k = linear(mem, params["k_proj"]).reshape(B, nb, H, D).transpose(0, 2, 1, 3)
    v = linear(mem, params["v_proj"]).reshape(B, nb, H, D).transpose(0, 2, 1, 3)

    block_ends = (jnp.arange(nb) + 1) * cfg.compression_block - 1
    block_ends = jnp.minimum(block_ends, T - 1)
    k = apply_rope_at_positions(k, cos, sin, block_ends)

    null_k = jnp.broadcast_to(params["null_k"], (B, H, 1, D))
    null_v = jnp.broadcast_to(params["null_v"], (B, H, 1, D))
    k = jnp.concatenate([null_k, k], axis=2)
    v = jnp.concatenate([null_v, v], axis=2)

    scores = jnp.einsum("bhtd,bhmd->bhtm", q, k) / math.sqrt(D)
    token_pos = jnp.arange(T)[:, None]
    allow_blocks = block_ends[None, :] <= token_pos
    allow = jnp.concatenate([jnp.ones((T, 1), dtype=bool), allow_blocks], axis=1)
    scores = jnp.where(allow[None, None, :, :], scores, _mask_fill_value(scores.dtype))
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhtm,bhmd->bhtd", weights, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    out = linear(out, params["out"])
    x = residual + jax.nn.sigmoid(params["gate"]) * out
    x = x + mlp_forward(params["mlp"], layer_norm(x, params["ln2"]))
    return x


def block_forward(params, x, cfg, cos, sin, attention_backend="windowed", span_backend="materialized"):
    if "attn" in params:
        return local_attention_block_forward(params, x, cfg, cos, sin, attention_backend)
    if "edge_proj" in params:
        return span_block_forward(params, x, cfg, span_backend)
    if "pool_score" in params:
        return compressed_memory_block_forward(params, x, cfg, cos, sin)
    raise ValueError(f"unknown block params: {sorted(params.keys())}")
