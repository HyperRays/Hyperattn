import jax
import jax.numpy as jnp


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
