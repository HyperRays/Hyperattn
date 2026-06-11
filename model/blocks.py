import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EfficientHGConfig
from .rope import apply_rope, apply_rope_at_positions


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CausalLocalSelfAttention(nn.Module):
    def __init__(self, cfg: EfficientHGConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.local_window = cfg.local_window
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        qkv = self.qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        pos = torch.arange(T, device=x.device)
        i = pos[:, None]
        j = pos[None, :]
        mask = (j <= i) & ((i - j) < self.local_window)
        mask = mask.view(1, 1, T, T)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out(y))


class LocalAttentionBlock(nn.Module):
    def __init__(self, cfg: EfficientHGConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalLocalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg.n_embd, cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSpanHypergraphBlock(nn.Module):
    """Linear-time causal span-hypergraph block."""

    def __init__(self, cfg: EfficientHGConfig, gate_init: float = -3.0):
        super().__init__()
        self.widths = tuple(cfg.span_widths)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.in_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.edge_proj = nn.Linear(cfg.n_embd * len(self.widths), cfg.n_embd)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg.n_embd, cfg.dropout)

    @staticmethod
    def span_mean_from_prefix(h, prefix, width: int):
        B, T, C = h.shape
        ends = torch.arange(1, T + 1, device=h.device)
        starts = (ends - width).clamp_min(0)
        span_sum = prefix[:, ends, :] - prefix[:, starts, :]
        span_len = (ends - starts).to(h.dtype).view(1, T, 1)
        return span_sum / span_len

    def forward(self, x, cos=None, sin=None):
        residual = x
        h = self.in_proj(self.ln1(x))
        prefix = torch.cat([torch.zeros_like(h[:, :1, :]), h.cumsum(dim=1)], dim=1)
        spans = [self.span_mean_from_prefix(h, prefix, w) for w in self.widths]
        z = torch.cat(spans, dim=-1)
        z = self.edge_proj(z)
        z = F.gelu(z)
        z = self.drop(self.out_proj(z))
        x = residual + torch.sigmoid(self.gate) * z
        x = x + self.mlp(self.ln2(x))
        return x


class CausalCompressedMemoryAttentionBlock(nn.Module):
    """HCA-style compressed global memory block."""

    def __init__(self, cfg: EfficientHGConfig, gate_init: float = -4.0):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.compression_block = cfg.compression_block
        self.dropout = cfg.dropout

        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.pool_score = nn.Linear(cfg.n_embd, 1, bias=False)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        self.null_k = nn.Parameter(torch.zeros(1, cfg.n_head, 1, self.head_dim))
        self.null_v = nn.Parameter(torch.zeros(1, cfg.n_head, 1, self.head_dim))
        nn.init.normal_(self.null_k, std=0.02)
        nn.init.normal_(self.null_v, std=0.02)

        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg.n_embd, cfg.dropout)

    def _compress_blocks(self, y):
        B, T, C = y.shape
        cb = self.compression_block
        pad_len = (-T) % cb
        y_pad = F.pad(y, (0, 0, 0, pad_len)) if pad_len else y
        Tp = y_pad.size(1)
        nb = Tp // cb
        chunks = y_pad.view(B, nb, cb, C)

        valid = torch.arange(Tp, device=y.device).view(nb, cb) < T
        score = self.pool_score(chunks).squeeze(-1)
        score = score.masked_fill(~valid.view(1, nb, cb), torch.finfo(score.dtype).min)
        weight = F.softmax(score, dim=-1)
        mem = (weight.unsqueeze(-1) * chunks).sum(dim=2)
        return mem, nb

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        residual = x
        y = self.ln1(x)

        q = self.q_proj(y).view(B, T, H, D).transpose(1, 2)
        q = apply_rope(q, cos, sin)

        mem, nb = self._compress_blocks(y)
        k = self.k_proj(mem).view(B, nb, H, D).transpose(1, 2)
        v = self.v_proj(mem).view(B, nb, H, D).transpose(1, 2)

        block_ends = (torch.arange(nb, device=x.device) + 1) * self.compression_block - 1
        block_ends = block_ends.clamp_max(T - 1)
        k = apply_rope_at_positions(k, cos, sin, block_ends)

        null_k = self.null_k.expand(B, -1, -1, -1)
        null_v = self.null_v.expand(B, -1, -1, -1)
        k = torch.cat([null_k, k], dim=2)
        v = torch.cat([null_v, v], dim=2)

        token_pos = torch.arange(T, device=x.device)[:, None]
        allow_blocks = block_ends[None, :] <= token_pos
        allow = torch.cat([torch.ones(T, 1, device=x.device, dtype=torch.bool), allow_blocks], dim=1)
        allow = allow.view(1, 1, T, nb + 1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allow,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.drop(self.out(out))
        x = residual + torch.sigmoid(self.gate) * out
        x = x + self.mlp(self.ln2(x))
        return x
