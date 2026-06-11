import jax
import jax.numpy as jnp


def precompute_rope_cache(head_dim: int, max_seq_len: int, base: float = 10_000.0, dtype=jnp.float32):
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("t,d->td", t, inv_freq)
    cos = jnp.cos(freqs)[None, None, :, :].astype(dtype)
    sin = jnp.sin(freqs)[None, None, :, :].astype(dtype)
    return cos, sin


def apply_rope(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos
    return jnp.stack((y_even, y_odd), axis=-1).reshape(*x.shape)


def apply_rope_bthd(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    c = jnp.swapaxes(cos[:, :, : x.shape[1], :], 1, 2)
    s = jnp.swapaxes(sin[:, :, : x.shape[1], :], 1, 2)
    y_even = x_even * c - x_odd * s
    y_odd = x_even * s + x_odd * c
    return jnp.stack((y_even, y_odd), axis=-1).reshape(*x.shape)


def apply_rope_at_positions(x, cos, sin, positions):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    c = jnp.take(cos, positions, axis=2)
    s = jnp.take(sin, positions, axis=2)
    y_even = x_even * c - x_odd * s
    y_odd = x_even * s + x_odd * c
    return jnp.stack((y_even, y_odd), axis=-1).reshape(*x.shape)


def block_until_ready(tree):
    return jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)
