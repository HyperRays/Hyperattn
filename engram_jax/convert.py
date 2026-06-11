import jax.numpy as jnp

from .config import EngramLayerSpec


def _array(tensor):
    return jnp.asarray(tensor.detach().cpu().numpy())


def from_torch_engram(module):
    """Convert a torch reference Engram module to (params, EngramLayerSpec)."""
    G = module.backbone_cfg.hc_mult
    D = module.backbone_cfg.hidden_size
    n_heads = module.multi_head_embedding.num_heads
    head_dim = module.multi_head_embedding.embedding_dim
    K = module.short_conv.conv.kernel_size[0]

    norm_eps = module.norm1[0].eps
    if norm_eps is None:
        norm_eps = float(jnp.finfo(jnp.float32).eps)

    spec = EngramLayerSpec(
        hidden_size=D,
        hc_mult=G,
        n_heads=n_heads,
        head_dim=head_dim,
        kernel_size=K,
        dilation=module.short_conv.conv.dilation[0],
        norm_eps=norm_eps,
        conv_norm_eps=module.short_conv.norms[0].eps,
    )

    params = {
        "embedding": {
            "weight": _array(module.multi_head_embedding.embedding.weight),
            "offsets": _array(module.multi_head_embedding.offsets).astype(jnp.int32),
        },
        "value_proj": {
            "weight": _array(module.value_proj.weight),
            "bias": _array(module.value_proj.bias),
        },
        "key_proj": {
            "weight": jnp.stack([_array(p.weight) for p in module.key_projs]),
            "bias": jnp.stack([_array(p.bias) for p in module.key_projs]),
        },
        "norm1": jnp.stack([_array(n.weight) for n in module.norm1]),
        "norm2": jnp.stack([_array(n.weight) for n in module.norm2]),
        "short_conv": {
            "norm": jnp.stack([_array(n.weight) for n in module.short_conv.norms]),
            "conv": _array(module.short_conv.conv.weight).reshape(G, D, K),
        },
    }
    return params, spec
