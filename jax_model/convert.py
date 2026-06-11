import jax.numpy as jnp

from .config import from_torch_config


def _array(tensor):
    return jnp.asarray(tensor.detach().cpu().numpy())


def _linear(state, prefix, bias=True):
    params = {"weight": _array(state[f"{prefix}.weight"])}
    key = f"{prefix}.bias"
    if bias and key in state:
        params["bias"] = _array(state[key])
    return params


def _layer_norm(state, prefix):
    return {
        "weight": _array(state[f"{prefix}.weight"]),
        "bias": _array(state[f"{prefix}.bias"]),
    }


def _mlp(state, prefix):
    return {
        "fc": _linear(state, f"{prefix}.net.0"),
        "proj": _linear(state, f"{prefix}.net.2"),
    }


def _local_block(state, prefix):
    return {
        "ln1": _layer_norm(state, f"{prefix}.ln1"),
        "attn": {
            "qkv": _linear(state, f"{prefix}.attn.qkv", bias=False),
            "out": _linear(state, f"{prefix}.attn.out", bias=False),
        },
        "ln2": _layer_norm(state, f"{prefix}.ln2"),
        "mlp": _mlp(state, f"{prefix}.mlp"),
    }


def _span_block(state, prefix):
    return {
        "ln1": _layer_norm(state, f"{prefix}.ln1"),
        "in_proj": _linear(state, f"{prefix}.in_proj"),
        "edge_proj": _linear(state, f"{prefix}.edge_proj"),
        "out_proj": _linear(state, f"{prefix}.out_proj"),
        "gate": _array(state[f"{prefix}.gate"]),
        "ln2": _layer_norm(state, f"{prefix}.ln2"),
        "mlp": _mlp(state, f"{prefix}.mlp"),
    }


def _memory_block(state, prefix):
    return {
        "ln1": _layer_norm(state, f"{prefix}.ln1"),
        "q_proj": _linear(state, f"{prefix}.q_proj", bias=False),
        "k_proj": _linear(state, f"{prefix}.k_proj", bias=False),
        "v_proj": _linear(state, f"{prefix}.v_proj", bias=False),
        "pool_score": _linear(state, f"{prefix}.pool_score", bias=False),
        "out": _linear(state, f"{prefix}.out", bias=False),
        "gate": _array(state[f"{prefix}.gate"]),
        "null_k": _array(state[f"{prefix}.null_k"]),
        "null_v": _array(state[f"{prefix}.null_v"]),
        "ln2": _layer_norm(state, f"{prefix}.ln2"),
        "mlp": _mlp(state, f"{prefix}.mlp"),
    }


def from_torch_model(torch_model):
    state = torch_model.state_dict()
    cfg = from_torch_config(torch_model.cfg)
    params = {
        "token_embedding": {"weight": _array(state["token_embedding.weight"])},
        "blocks": [],
        "ln_f": _layer_norm(state, "ln_f"),
    }

    for i, block in enumerate(torch_model.blocks):
        prefix = f"blocks.{i}"
        class_name = block.__class__.__name__
        if class_name == "LocalAttentionBlock":
            params["blocks"].append(_local_block(state, prefix))
        elif class_name == "CausalSpanHypergraphBlock":
            params["blocks"].append(_span_block(state, prefix))
        elif class_name == "CausalCompressedMemoryAttentionBlock":
            params["blocks"].append(_memory_block(state, prefix))
        else:
            raise ValueError(f"unsupported torch block type: {class_name}")
    return params, cfg
