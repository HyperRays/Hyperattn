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


class EfficientHypergraphLM(nn.Module):
    def __init__(self, cfg: EfficientHGConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        blocks = []
        for _ in range(cfg.n_local_attn_layers):
            blocks.append(LocalAttentionBlock(cfg))

        mem_insert_every = max(1, cfg.n_span_layers // max(1, cfg.n_compressed_memory_layers))
        mem_inserted = 0
        for i in range(cfg.n_span_layers):
            blocks.append(CausalSpanHypergraphBlock(cfg))
            if mem_inserted < cfg.n_compressed_memory_layers and ((i + 1) % mem_insert_every == 0):
                blocks.append(CausalCompressedMemoryAttentionBlock(cfg))
                mem_inserted += 1

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
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
