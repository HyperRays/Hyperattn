import torch


def precompute_rope_cache(head_dim: int, max_seq_len: int, device, base: float = 10_000.0):
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.einsum("t,d->td", t, inv_freq)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: [B, H, T, D]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos[:, :, : x.size(-2), :]
    sin = sin[:, :, : x.size(-2), :]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos
    return torch.stack((y_even, y_odd), dim=-1).flatten(-2)


def apply_rope_at_positions(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor):
    # x: [B, H, M, D], positions: [M]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    c = cos[:, :, positions, :]
    s = sin[:, :, positions, :]
    y_even = x_even * c - x_odd * s
    y_odd = x_even * s + x_odd * c
    return torch.stack((y_even, y_odd), dim=-1).flatten(-2)
