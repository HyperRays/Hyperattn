import jax
import jax.numpy as jnp


LOGIT_CHUNK_SIZE = 2048


def linear(x, params):
    y = x @ params["weight"].T
    bias = params.get("bias")
    if bias is not None:
        y = y + bias
    return y


def layer_norm(x, params, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) * jax.lax.rsqrt(var + eps)
    return y * params["weight"] + params["bias"]


def gelu(x):
    return jax.nn.gelu(x, approximate=False)


def softmax_cross_entropy(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_loss = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return jnp.mean(token_loss)


def _pad_vocab(weight):
    vocab_size = weight.shape[0]
    pad = (-vocab_size) % LOGIT_CHUNK_SIZE
    if pad:
        weight = jnp.pad(weight, ((0, pad), (0, 0)))
    return weight, vocab_size


def _linear_cross_entropy_token_loss_and_lse(hidden, weight, targets):
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    weight_pad, vocab_size = _pad_vocab(weight)
    num_chunks = weight_pad.shape[0] // LOGIT_CHUNK_SIZE

    def scan_chunk(carry, chunk_id):
        max_score, exp_sum = carry
        start = chunk_id * LOGIT_CHUNK_SIZE
        w = jax.lax.dynamic_slice_in_dim(weight_pad, start, LOGIT_CHUNK_SIZE, axis=0)
        logits = (h @ w.T).astype(jnp.float32)
        valid = start + jnp.arange(LOGIT_CHUNK_SIZE) < vocab_size
        logits = jnp.where(valid[None, :], logits, -jnp.inf)
        chunk_max = jnp.max(logits, axis=-1)
        new_max = jnp.maximum(max_score, chunk_max)
        exp_sum = exp_sum * jnp.exp(max_score - new_max)
        exp_sum = exp_sum + jnp.sum(jnp.exp(logits - new_max[:, None]), axis=-1)
        return (new_max, exp_sum), None

    init = (
        jnp.full((h.shape[0],), -jnp.inf, dtype=jnp.float32),
        jnp.zeros((h.shape[0],), dtype=jnp.float32),
    )
    (max_score, exp_sum), _ = jax.lax.scan(scan_chunk, init, jnp.arange(num_chunks))
    lse = max_score + jnp.log(exp_sum)
    target_logits = jnp.sum(h * weight[t], axis=-1).astype(jnp.float32)
    return lse - target_logits, lse


def linear_cross_entropy_tokens(hidden, weight, targets):
    """Per-token cross entropy for a tied output projection, without materializing logits."""
    token_loss, _ = _linear_cross_entropy_token_loss_and_lse(hidden, weight, targets)
    return token_loss.reshape(targets.shape)


@jax.custom_vjp
def linear_cross_entropy(hidden, weight, targets):
    """Mean cross entropy for hidden @ weight.T, chunked over vocab.

    This is mathematically equivalent to softmax_cross_entropy(hidden @ weight.T, targets),
    but avoids a [batch, sequence, vocab] logits allocation in forward/backward.
    """
    return jnp.mean(linear_cross_entropy_tokens(hidden, weight, targets))


def _linear_cross_entropy_fwd(hidden, weight, targets):
    token_loss, lse = _linear_cross_entropy_token_loss_and_lse(hidden, weight, targets)
    return jnp.mean(token_loss), (hidden, weight, targets, lse)


def _linear_cross_entropy_bwd(res, g):
    hidden, weight, targets, lse = res
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    weight_pad, vocab_size = _pad_vocab(weight)
    num_chunks = weight_pad.shape[0] // LOGIT_CHUNK_SIZE
    scale = (g / h.shape[0]).astype(jnp.float32)
    h_f32 = h.astype(jnp.float32)

    def scan_chunk(grad_h, chunk_id):
        start = chunk_id * LOGIT_CHUNK_SIZE
        w = jax.lax.dynamic_slice_in_dim(weight_pad, start, LOGIT_CHUNK_SIZE, axis=0)
        logits = (h @ w.T).astype(jnp.float32)
        valid = start + jnp.arange(LOGIT_CHUNK_SIZE) < vocab_size
        probs = jnp.exp(logits - lse[:, None]) * scale
        probs = jnp.where(valid[None, :], probs, 0.0)
        grad_h = grad_h + probs @ w.astype(jnp.float32)
        grad_w_chunk = probs.T @ h_f32
        return grad_h, grad_w_chunk

    grad_h, grad_w_chunks = jax.lax.scan(
        scan_chunk,
        jnp.zeros(h.shape, dtype=jnp.float32),
        jnp.arange(num_chunks),
    )
    grad_h = grad_h - weight[t].astype(jnp.float32) * scale
    grad_w = grad_w_chunks.reshape(weight_pad.shape)[:vocab_size]
    grad_w = grad_w.at[t].add(-h_f32 * scale)
    return grad_h.reshape(hidden.shape).astype(hidden.dtype), grad_w.astype(weight.dtype), None


linear_cross_entropy.defvjp(_linear_cross_entropy_fwd, _linear_cross_entropy_bwd)
