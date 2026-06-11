from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280 * 5, 129280 * 5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4


@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30


@dataclass(frozen=True)
class EngramLayerSpec:
    """Static (hashable) description of one Engram layer, usable as a jit static arg."""

    hidden_size: int
    hc_mult: int
    n_heads: int  # total hash heads = (max_ngram_size - 1) * n_head_per_ngram
    head_dim: int  # n_embed_per_ngram // n_head_per_ngram
    kernel_size: int
    dilation: int
    gate_eps: float = 1e-6
    norm_eps: float = 1.1920928955078125e-07  # torch.nn.RMSNorm(eps=None) on fp32
    conv_norm_eps: float = 1e-5

    @property
    def engram_hidden_size(self) -> int:
        return self.n_heads * self.head_dim


def layer_spec(engram_cfg: EngramConfig, backbone_cfg: BackBoneConfig) -> EngramLayerSpec:
    return EngramLayerSpec(
        hidden_size=backbone_cfg.hidden_size,
        hc_mult=backbone_cfg.hc_mult,
        n_heads=(engram_cfg.max_ngram_size - 1) * engram_cfg.n_head_per_ngram,
        head_dim=engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        kernel_size=engram_cfg.kernel_size,
        dilation=engram_cfg.max_ngram_size,
    )
