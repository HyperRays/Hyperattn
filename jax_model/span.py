import jax.numpy as jnp


def _span_mean_from_prefix(prefix, width, dtype):
    # ends are 1..T and starts are a clamped shift, so the two gathers reduce to
    # a slice and a pad+slice (slices avoid gather/scatter kernels in fwd/bwd).
    T = prefix.shape[1] - 1
    w = min(int(width), T)
    lagged = jnp.pad(prefix[:, 1 : T + 1 - w, :], ((0, 0), (w, 0), (0, 0)))
    span_sum = prefix[:, 1:, :] - lagged
    span_len = jnp.minimum(jnp.arange(1, T + 1, dtype=jnp.int32), w).astype(dtype).reshape(1, T, 1)
    return span_sum / span_len


def span_hypergraph(h, widths):
    prefix = jnp.concatenate([jnp.zeros_like(h[:, :1, :]), jnp.cumsum(h, axis=1)], axis=1)
    spans = []
    for width in tuple(widths):
        spans.append(_span_mean_from_prefix(prefix, width, h.dtype))
    return jnp.concatenate(spans, axis=-1)


def fused_span_edge_projection(h, widths, edge_proj_params):
    prefix = jnp.concatenate([jnp.zeros_like(h[:, :1, :]), jnp.cumsum(h, axis=1)], axis=1)
    weight = edge_proj_params["weight"]
    bias = edge_proj_params.get("bias")
    C = h.shape[-1]
    out = jnp.zeros((*h.shape[:2], weight.shape[0]), dtype=h.dtype)

    for i, width in enumerate(tuple(widths)):
        span = _span_mean_from_prefix(prefix, width, h.dtype)
        weight_slice = weight[:, i * C : (i + 1) * C]
        out = out + span @ weight_slice.T

    if bias is not None:
        out = out + bias
    return out


def einsum_fused_span_edge_projection(h, widths, edge_proj_params):
    prefix = jnp.concatenate([jnp.zeros_like(h[:, :1, :]), jnp.cumsum(h, axis=1)], axis=1)
    spans = []
    for width in tuple(widths):
        spans.append(_span_mean_from_prefix(prefix, width, h.dtype))
    stacked = jnp.stack(spans, axis=1)
    weight = edge_proj_params["weight"].reshape(edge_proj_params["weight"].shape[0], len(tuple(widths)), h.shape[-1])
    out = jnp.einsum("bwtc,owc->bto", stacked, weight)
    bias = edge_proj_params.get("bias")
    if bias is not None:
        out = out + bias
    return out
