import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    CausalCompressedMemoryAttentionBlock,
    CausalSpanHypergraphBlock,
    LocalAttentionBlock,
)
from .config import EfficientHGConfig
from .rope import precompute_rope_cache


_BLOCK_BUILDERS = {
    "attn": LocalAttentionBlock,
    "span": CausalSpanHypergraphBlock,
    "hca": CausalCompressedMemoryAttentionBlock,
    "mem": CausalCompressedMemoryAttentionBlock,
}


def _apply_top_p(logits, top_p, min_tokens_to_keep=1):
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., :min_tokens_to_keep] = False
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
    return torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)


def _apply_no_repeat_ngram(logits, idx, ngram_size):
    if ngram_size is None or ngram_size <= 0 or idx.size(1) < ngram_size - 1:
        return logits
    prefix_len = ngram_size - 1
    for b in range(idx.size(0)):
        tokens = idx[b].tolist()
        prefix = tuple(tokens[-prefix_len:]) if prefix_len else tuple()
        banned = []
        for i in range(len(tokens) - ngram_size + 1):
            if tuple(tokens[i : i + prefix_len]) == prefix:
                banned.append(tokens[i + prefix_len])
        if banned:
            logits[b, torch.tensor(banned, device=logits.device)] = -float("inf")
    return logits


def _apply_token_penalties(
    logits,
    idx,
    *,
    repetition_penalty=1.0,
    repetition_window=None,
    frequency_penalty=0.0,
    presence_penalty=0.0,
):
    if (
        (repetition_penalty is None or repetition_penalty == 1.0)
        and frequency_penalty == 0.0
        and presence_penalty == 0.0
    ):
        return logits

    recent = idx if repetition_window is None else idx[:, -repetition_window:]
    for b in range(idx.size(0)):
        tokens, counts = torch.unique(recent[b], return_counts=True)
        selected = logits[b, tokens]
        if repetition_penalty is not None and repetition_penalty != 1.0:
            selected = torch.where(selected > 0, selected / repetition_penalty, selected * repetition_penalty)
        if presence_penalty:
            selected = selected - presence_penalty
        if frequency_penalty:
            selected = selected - frequency_penalty * counts.to(selected.dtype)
        logits[b, tokens] = selected
    return logits


def _default_layout(cfg: EfficientHGConfig):
    """Legacy stack: local attention, then span layers with memory blocks interleaved."""
    layout = ["attn"] * cfg.n_local_attn_layers
    mem_insert_every = max(1, cfg.n_span_layers // max(1, cfg.n_compressed_memory_layers))
    mem_inserted = 0
    for i in range(cfg.n_span_layers):
        layout.append("span")
        if mem_inserted < cfg.n_compressed_memory_layers and ((i + 1) % mem_insert_every == 0):
            layout.append("hca")
            mem_inserted += 1
    return tuple(layout)


def resolve_block_layout(cfg: EfficientHGConfig):
    layout = cfg.block_layout if cfg.block_layout is not None else _default_layout(cfg)
    unknown = [name for name in layout if name not in _BLOCK_BUILDERS]
    if unknown:
        raise ValueError(f"unknown block types {unknown}; valid: {sorted(_BLOCK_BUILDERS)}")
    return tuple(layout)


class EfficientHypergraphLM(nn.Module):
    def __init__(self, cfg: EfficientHGConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.block_layout = resolve_block_layout(cfg)
        blocks = [_BLOCK_BUILDERS[name](cfg) for name in self.block_layout]

        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        cos, sin = precompute_rope_cache(self.cfg.n_embd // self.cfg.n_head, T, idx.device)
        x = self.drop(self.token_embedding(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=50,
        top_p=None,
        repetition_penalty=1.0,
        repetition_window=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        no_repeat_ngram_size=None,
        eos_token_id=None,
        min_new_tokens=0,
    ):
        self.eval()
        for step in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = _apply_token_penalties(
                logits,
                idx,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            logits = _apply_no_repeat_ngram(logits, idx, no_repeat_ngram_size)
            if eos_token_id is not None and step < min_new_tokens:
                logits[:, eos_token_id] = -float("inf")
            logits = logits / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            logits = _apply_top_p(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token_id is not None and torch.all(next_id == eos_token_id):
                break
        return idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
