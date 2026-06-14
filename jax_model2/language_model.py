import jax
import jax.numpy as jnp

from jax_model.ops import layer_norm, linear_cross_entropy
from jax_model.rope import precompute_rope_cache

from .config import resolve_block_layout
from .layers import (
    horizontal_block_forward,
    memory_router_write,
    route_block_input_memory,
    routed_block_forward,
    span_block_forward,
    tokenwise_layer_attention,
    tokenwise_layer_attention_from_bank,
)

SPAN_KINDS = ("far_span", "mid_span", "local_span")


def _stack_same_structure(blocks):
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *blocks)


def _scan_routed_span_run(stacked_params, x0, window, cfg, kind, span_backend, remat_blocks):
    """Run a homogeneous, fully-capped span run as a single lax.scan.

    Within the run depth always exceeds layer_attn_max_sources, so every block routes over
    exactly ``embedding (x0) + the last (max_sources - 1) outputs``. That window is a
    fixed-size carry, so the run compiles to one block body instead of being unrolled.
    ``window`` is [B, K, T, C] (K = max_sources - 1), oldest-first, newest == x_prev.
    """

    def body(win, block_params):
        x_prev = win[:, -1]
        bank = jnp.concatenate([x0[:, None], win], axis=1)  # [B, max_sources, T, C]
        routed = tokenwise_layer_attention_from_bank(block_params["route"], x_prev, bank, cfg)
        gate = jax.nn.sigmoid(block_params["route"]["gate"])
        x_in = gate * routed + (1.0 - gate) * x_prev
        x_out = span_block_forward(block_params, x_in, cfg, kind, span_backend)
        return jnp.concatenate([win[:, 1:], x_out[:, None]], axis=1)

    # Remat the WHOLE step (routing + block), matching the unrolled path. Otherwise the scan
    # saves the per-step bank [B, max_sources, T, C] stacked over the run length for backward
    # (e.g. bf16[16, B, 8, T, C] ~= 6G per run on TPU); rematerializing recomputes it instead,
    # so the scan only carries the window.
    body = jax.checkpoint(body) if remat_blocks else body

    def step(win, block_params):
        return body(win, block_params), None

    window, _ = jax.lax.scan(step, window, stacked_params)
    return window  # final window; x_out of the run == window[:, -1]


def _memory_block(block, x, x0, mem, cfg, kind, cos, sin, attention_backend, span_backend, remat_blocks):
    """One unrolled block under the memory router: read over [embedding; mem], then write."""

    def apply(b, h, ms):
        bank = jnp.concatenate([x0[:, None], ms], axis=1)  # [B, layer_memory_slots, T, C]
        x_in = route_block_input_memory(b["route"], h, bank, cfg)
        out = horizontal_block_forward(b, x_in, cfg, kind, cos, sin, attention_backend, span_backend)
        return out, memory_router_write(b["route"], ms, out, cfg)

    return jax.checkpoint(apply)(block, x, mem) if remat_blocks else apply(block, x, mem)


def _scan_memory_span_run(stacked_params, x0, x_init, mem, cfg, kind, span_backend, remat_blocks):
    """A homogeneous span run under the memory router as one lax.scan; carry = (x, mem)."""

    def body(carry, block_params):
        x, ms = carry
        bank = jnp.concatenate([x0[:, None], ms], axis=1)
        x_in = route_block_input_memory(block_params["route"], x, bank, cfg)
        out = span_block_forward(block_params, x_in, cfg, kind, span_backend)
        return out, memory_router_write(block_params["route"], ms, out, cfg)

    body = jax.checkpoint(body) if remat_blocks else body

    def step(carry, block_params):
        return body(carry, block_params), None

    (x, mem), _ = jax.lax.scan(step, (x_init, mem), stacked_params)
    return x, mem


def _forward_backbone_memory(params, x, cfg, layout, cos, sin, *, attention_backend, span_backend, remat_blocks, scan_span_runs):
    B, T, C = x.shape
    blocks = params["blocks"]
    n = len(blocks)
    x0 = x  # embedding == memory slot 0 (explicit skip path), never overwritten
    mem = jnp.broadcast_to(params["mem_init"][None, :, None, :], (B, cfg.layer_memory_slots - 1, T, C))
    i = 0
    while i < n:
        kind = layout[i]
        if scan_span_runs and kind in SPAN_KINDS:
            j = i + 1
            while j < n and layout[j] == kind:
                j += 1
            if j - i > 1:
                x, mem = _scan_memory_span_run(
                    _stack_same_structure(blocks[i:j]), x0, x, mem, cfg, kind, span_backend, remat_blocks
                )
                i = j
                continue
        x, mem = _memory_block(
            blocks[i], x, x0, mem, cfg, kind, cos, sin, attention_backend, span_backend, remat_blocks
        )
        i += 1
    return x


def _forward_one_block(
    block,
    x,
    states,
    cfg,
    kind,
    cos,
    sin,
    *,
    block_index,
    attention_backend,
    span_backend,
    remat_blocks,
):
    def apply_block(b, h, ss):
        return routed_block_forward(
            b,
            h,
            ss,
            cfg,
            kind,
            cos,
            sin,
            block_index=block_index,
            attention_backend=attention_backend,
            span_backend=span_backend,
        )

    return jax.checkpoint(apply_block)(block, x, states) if remat_blocks else apply_block(block, x, states)


def forward_backbone(
    params,
    idx,
    cfg,
    *,
    attention_backend="windowed",
    span_backend="fused",
    remat_blocks=False,
    scan_span_runs=False,
):
    _, T = idx.shape
    assert T <= cfg.block_size
    layout = resolve_block_layout(cfg)
    cos, sin = precompute_rope_cache(cfg.n_embd // cfg.n_head, T, dtype=params["token_embedding"]["weight"].dtype)
    x = params["token_embedding"]["weight"][idx]

    if cfg.use_memory_router:
        x = _forward_backbone_memory(
            params,
            x,
            cfg,
            layout,
            cos,
            sin,
            attention_backend=attention_backend,
            span_backend=span_backend,
            remat_blocks=remat_blocks,
            scan_span_runs=scan_span_runs,
        )
        return layer_norm(x, params["ln_f"])

    blocks = params["blocks"]
    n = len(blocks)
    max_src = cfg.layer_attn_max_sources
    # A homogeneous span run starting at block i is "fully capped" (every block routes over
    # exactly max_src sources) iff i >= max_src; only then can it be scanned as a fixed-size
    # window. max_src must be >= 2 (max_src == 1 drops the embedding source). Earlier/shorter
    # runs and attention blocks stay unrolled.
    can_scan = scan_span_runs and cfg.use_layer_attention and max_src is not None and max_src >= 2

    states = (x,)
    i = 0
    while i < n:
        kind = layout[i]
        if can_scan and kind in SPAN_KINDS and i >= max_src:
            j = i + 1
            while j < n and layout[j] == kind:
                j += 1
            if j - i > 1:
                k = max_src - 1
                window = jnp.stack(states[-k:], axis=1)  # [B, K, T, C], oldest-first
                window = _scan_routed_span_run(
                    _stack_same_structure(blocks[i:j]),
                    states[0],
                    window,
                    cfg,
                    kind,
                    span_backend,
                    remat_blocks,
                )
                x = window[:, -1]
                # Collapse history to (embedding, last K outputs): all that capped routing in
                # the following blocks can ever read, and exactly equal to the true tail.
                states = (states[0],) + tuple(jnp.moveaxis(window, 1, 0))
                i = j
                continue
        x = _forward_one_block(
            blocks[i],
            x,
            states,
            cfg,
            kind,
            cos,
            sin,
            block_index=i,
            attention_backend=attention_backend,
            span_backend=span_backend,
            remat_blocks=remat_blocks,
        )
        states = states + (x,)
        i += 1
    return layer_norm(x, params["ln_f"])


def forward(
    params,
    idx,
    cfg,
    *,
    attention_backend="windowed",
    span_backend="fused",
    remat_blocks=False,
    scan_span_runs=False,
):
    hidden = forward_backbone(
        params,
        idx,
        cfg,
        attention_backend=attention_backend,
        span_backend=span_backend,
        remat_blocks=remat_blocks,
        scan_span_runs=scan_span_runs,
    )
    return hidden @ params["token_embedding"]["weight"].T


def loss(
    params,
    idx,
    targets,
    cfg,
    *,
    attention_backend="windowed",
    span_backend="fused",
    remat_blocks=False,
    scan_span_runs=False,
):
    hidden = forward_backbone(
        params,
        idx,
        cfg,
        attention_backend=attention_backend,
        span_backend=span_backend,
        remat_blocks=remat_blocks,
        scan_span_runs=scan_span_runs,
    )
    return linear_cross_entropy(hidden, params["token_embedding"]["weight"], targets)


def layer_attention_weights(
    params,
    idx,
    cfg,
    *,
    attention_backend="windowed",
    span_backend="fused",
):
    """Return tokenwise depth-routing weights for inspection.

    Each entry has shape [B, num_sources, T] and corresponds to the layer attention
    before that block. The first block has no entry because it only sees embeddings.
    """

    _, T = idx.shape
    layout = resolve_block_layout(cfg)
    cos, sin = precompute_rope_cache(cfg.n_embd // cfg.n_head, T, dtype=params["token_embedding"]["weight"].dtype)
    x = params["token_embedding"]["weight"][idx]
    states = (x,)
    weights = []
    for i, (kind, block) in enumerate(zip(layout, params["blocks"])):
        if cfg.use_layer_attention and i > 0:
            # Inline the score calculation so diagnostics can inspect weights
            # without changing the main forward path.
            from .layers import _select_layer_sources
            from jax_model.ops import layer_norm, linear
            import math

            sources = _select_layer_sources(states, cfg.layer_attn_max_sources)
            bank = jnp.stack(sources, axis=1)
            q = linear(layer_norm(x, block["route"]["ln"]), block["route"]["q_proj"])
            k = linear(layer_norm(bank, block["route"]["ln"]), block["route"]["k_proj"])
            scores = jnp.einsum("btc,bstc->bst", q, k) / math.sqrt(cfg.n_embd)
            weights.append(jax.nn.softmax(scores.astype(jnp.float32), axis=1))
            route = tokenwise_layer_attention(block["route"], x, states, cfg)
            gate = jax.nn.sigmoid(block["route"]["gate"])
            x_in = gate * route + (1.0 - gate) * x
        else:
            x_in = x
        from .layers import horizontal_block_forward

        x = horizontal_block_forward(block, x_in, cfg, kind, cos, sin, attention_backend, span_backend)
        states = states + (x,)
    return weights


def count_parameters(params):
    leaves = []

    def collect(value):
        if isinstance(value, dict):
            for v in value.values():
                collect(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                collect(v)
        elif hasattr(value, "size"):
            leaves.append(value)

    collect(params)
    return sum(int(jnp.size(x)) for x in leaves)
