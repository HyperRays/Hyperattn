import math

import jax
import jax.numpy as jnp

from .config import EngramLayerSpec


def rms_norm(x, weight, eps):
    scale = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x * scale * weight


def multi_head_embedding(params, hash_ids):
    """hash_ids: (B, T, n_heads) int32 -> (B, T, n_heads * head_dim).

    One fused gather from the concatenated per-head tables; per-head offsets are
    baked into the ids instead of indexing n_heads separate tables.
    """
    ids = hash_ids + params["offsets"]
    emb = params["weight"][ids]
    B, T = ids.shape[:2]
    return emb.reshape(B, T, -1)


def short_conv(params, x, spec: EngramLayerSpec):
    """Causal depthwise dilated conv over (B, T, G, C), with per-group RMSNorm + SiLU.

    The grouped Conv1d is unrolled into kernel_size shifted scaled adds (pad+slice).
    For K=4 this is just as memory-bound as a conv kernel, differentiates cleanly,
    and avoids grouped-convolution lowering entirely (unsupported/slow on jax-metal).
    """
    T = x.shape[1]
    K = spec.kernel_size
    dil = spec.dilation

    xn = rms_norm(x, params["norm"], spec.conv_norm_eps)
    w = params["conv"]  # (G, C, K), torch Conv1d layout (GC, 1, K) reshaped

    # torch cross-correlation with left padding (K-1)*dil: y[t] = sum_m w[K-1-m] * x[t - m*dil]
    y = xn * w[..., K - 1]
    for m in range(1, K):
        shift = m * dil
        lagged = jnp.pad(xn[:, : T - shift], ((0, 0), (shift, 0), (0, 0), (0, 0)))
        y = y + lagged * w[..., K - 1 - m]
    return jax.nn.silu(y)


def engram_forward(params, hidden_states, hash_ids, spec: EngramLayerSpec):
    """hidden_states: (B, T, hc_mult, hidden); hash_ids: (B, T, n_heads) int32.

    Everything is vectorized over the hc_mult groups: the per-group key
    projections run as one stacked einsum and the per-group RMSNorms use stacked
    (G, hidden) weights, so the whole layer jits to a handful of fused kernels.
    """
    D = spec.hidden_size
    embeddings = multi_head_embedding(params["embedding"], hash_ids)  # (B, T, E)

    keys = jnp.einsum("bte,gde->btgd", embeddings, params["key_proj"]["weight"])
    keys = keys + params["key_proj"]["bias"]
    normed_key = rms_norm(keys, params["norm1"], spec.norm_eps)  # (G, D) weights
    normed_query = rms_norm(hidden_states, params["norm2"], spec.norm_eps)

    gate = jnp.sum(normed_key * normed_query, axis=-1) / math.sqrt(D)  # (B, T, G)
    gate = jnp.sqrt(jnp.maximum(jnp.abs(gate), spec.gate_eps)) * jnp.sign(gate)
    gate = jax.nn.sigmoid(gate)[..., None]

    value = embeddings @ params["value_proj"]["weight"].T + params["value_proj"]["bias"]
    value = gate * value[:, :, None, :]  # (B, T, G, D)
    return value + short_conv(params["short_conv"], value, spec)


def init_engram_params(key, head_vocab_sizes, spec: EngramLayerSpec, dtype=jnp.float32):
    """Random init mirroring torch defaults. head_vocab_sizes: flat list of per-head N."""
    assert len(head_vocab_sizes) == spec.n_heads
    E = spec.engram_hidden_size
    D = spec.hidden_size
    G = spec.hc_mult
    K = spec.kernel_size
    k_emb, k_val_w, k_val_b, k_key_w, k_key_b, k_conv = jax.random.split(key, 6)

    offsets = jnp.asarray([0] + list(jnp.cumsum(jnp.asarray(head_vocab_sizes))[:-1]), dtype=jnp.int32)
    lin_bound = 1.0 / math.sqrt(E)
    return {
        "embedding": {
            "weight": jax.random.normal(k_emb, (int(sum(head_vocab_sizes)), spec.head_dim), dtype),
            "offsets": offsets,
        },
        "value_proj": {
            "weight": jax.random.uniform(k_val_w, (D, E), dtype, -lin_bound, lin_bound),
            "bias": jax.random.uniform(k_val_b, (D,), dtype, -lin_bound, lin_bound),
        },
        "key_proj": {
            "weight": jax.random.uniform(k_key_w, (G, D, E), dtype, -lin_bound, lin_bound),
            "bias": jax.random.uniform(k_key_b, (G, D), dtype, -lin_bound, lin_bound),
        },
        "norm1": jnp.ones((G, D), dtype),
        "norm2": jnp.ones((G, D), dtype),
        "short_conv": {
            "norm": jnp.ones((G, D), dtype),
            "conv": jax.random.uniform(k_conv, (G, D, K), dtype, -1.0 / math.sqrt(K), 1.0 / math.sqrt(K)),
        },
    }
