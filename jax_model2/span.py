import jax.numpy as jnp


def _prefix_at_delay(prefix, delay):
    """prefix[:, max(t + 1 - delay, 0)] for every token t."""

    T = prefix.shape[1] - 1
    delay = int(delay)
    if delay <= 0:
        return prefix[:, 1:, :]
    if delay >= T:
        return jnp.zeros_like(prefix[:, 1:, :])
    values = prefix[:, 1 : T + 1 - delay, :]
    return jnp.pad(values, ((0, 0), (delay, 0), (0, 0)))


def delayed_span_mean_from_prefix(prefix, width, lag, dtype):
    width = int(width)
    lag = int(lag)
    end = _prefix_at_delay(prefix, lag)
    start = _prefix_at_delay(prefix, lag + width)
    span_sum = end - start
    available = jnp.maximum(jnp.arange(1, prefix.shape[1], dtype=jnp.int32) - lag, 0)
    span_len = jnp.minimum(available, width).astype(dtype).reshape(1, -1, 1)
    return jnp.where(span_len > 0, span_sum / jnp.maximum(span_len, 1), 0)


def delayed_span_hypergraph(h, specs):
    prefix = jnp.concatenate([jnp.zeros_like(h[:, :1, :]), jnp.cumsum(h, axis=1)], axis=1)
    spans = [delayed_span_mean_from_prefix(prefix, width, lag, h.dtype) for width, lag in tuple(specs)]
    return jnp.concatenate(spans, axis=-1)


def fused_delayed_span_edge_projection(h, specs, edge_proj_params):
    prefix = jnp.concatenate([jnp.zeros_like(h[:, :1, :]), jnp.cumsum(h, axis=1)], axis=1)
    weight = edge_proj_params["weight"]
    bias = edge_proj_params.get("bias")
    C = h.shape[-1]
    out = jnp.zeros((*h.shape[:2], weight.shape[0]), dtype=h.dtype)

    for i, (width, lag) in enumerate(tuple(specs)):
        span = delayed_span_mean_from_prefix(prefix, width, lag, h.dtype)
        weight_slice = weight[:, i * C : (i + 1) * C]
        out = out + span @ weight_slice.T

    if bias is not None:
        out = out + bias
    return out
