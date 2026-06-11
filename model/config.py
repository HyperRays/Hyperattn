from dataclasses import dataclass


@dataclass
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
