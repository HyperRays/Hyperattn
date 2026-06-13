from dataclasses import dataclass


@dataclass(frozen=True)
class EfficientHGConfig:
    vocab_size: int
    block_size: int
    n_embd: int = 384
    n_head: int = 6
    n_local_attn_layers: int = 1
    n_span_layers: int = 6
    n_compressed_memory_layers: int = 1
    span_widths: tuple = (2, 4, 8, 16, 32, 64)
    local_window: int = 256
    compression_block: int = 64
    dropout: float = 0.1
    # Resolved block stack (tuple of "attn"/"span"/"hca"); informational on the JAX
    # side, where the forward pass dispatches per block on the converted param keys.
    block_layout: tuple = None


def from_torch_config(cfg):
    layout = getattr(cfg, "block_layout", None)
    return EfficientHGConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_local_attn_layers=cfg.n_local_attn_layers,
        n_span_layers=cfg.n_span_layers,
        n_compressed_memory_layers=cfg.n_compressed_memory_layers,
        span_widths=tuple(cfg.span_widths),
        local_window=cfg.local_window,
        compression_block=cfg.compression_block,
        dropout=cfg.dropout,
        block_layout=tuple(layout) if layout is not None else None,
    )
