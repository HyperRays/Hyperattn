from dataclasses import dataclass
from typing import Optional, Tuple


def default_block_layout():
    """Coarse-to-fine span stack with attention refresh points.

    The stack is intentionally explicit: local attention seeds token features, far
    delayed spans force old-context features, attention refreshes content routing,
    medium spans bind mid-range state, and local spans recover fluent next-token
    detail. Tokenwise layer attention is applied before each block after the first.
    """

    return (
        ("attn",)
        + ("far_span",) * 6
        + ("attn",)
        + ("mid_span",) * 10
        + ("attn",)
        + ("local_span",) * 16
    )


@dataclass(frozen=True)
class LayerRoutedHGConfig:
    vocab_size: int
    block_size: int
    n_embd: int = 384
    n_head: int = 6
    local_window: int = 256
    dropout: float = 0.0
    block_layout: Optional[Tuple[str, ...]] = None

    # Span families. A span pair (width, lag) summarizes
    # [position + 1 - lag - width, position + 1 - lag).
    far_span_widths: Tuple[int, ...] = (64, 128, 256, 512)
    far_span_lags: Tuple[int, ...] = (128, 256, 512, 1024)
    mid_span_widths: Tuple[int, ...] = (32, 64, 128, 256)
    mid_span_lags: Tuple[int, ...] = (32, 64, 128, 256)
    local_span_widths: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    local_span_lags: Tuple[int, ...] = (0,)

    # Tokenwise layer attention over previous block states. If max_sources is set,
    # the router keeps the embedding state plus the most recent max_sources - 1
    # states; None means all previous states.
    use_layer_attention: bool = True
    layer_attn_max_sources: Optional[int] = None
    route_gate_init: float = 2.0
    initializer_std: float = 0.02


def resolve_block_layout(cfg: LayerRoutedHGConfig):
    layout = cfg.block_layout if cfg.block_layout is not None else default_block_layout()
    valid = {"attn", "far_span", "mid_span", "local_span"}
    unknown = [name for name in layout if name not in valid]
    if unknown:
        raise ValueError(f"unknown block types {unknown}; valid: {sorted(valid)}")
    return tuple(layout)


def span_specs_for_kind(cfg: LayerRoutedHGConfig, kind: str):
    if kind == "far_span":
        widths, lags = cfg.far_span_widths, cfg.far_span_lags
    elif kind == "mid_span":
        widths, lags = cfg.mid_span_widths, cfg.mid_span_lags
    elif kind == "local_span":
        widths, lags = cfg.local_span_widths, cfg.local_span_lags
    else:
        raise ValueError(f"{kind!r} is not a span block kind")

    if len(lags) == 1:
        lags = lags * len(widths)
    if len(widths) != len(lags):
        raise ValueError(f"{kind} widths/lags must have equal length, or one lag for all widths")
    return tuple((int(w), int(lag)) for w, lag in zip(widths, lags))
