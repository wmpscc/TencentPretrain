import torch
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=inv_freq.device, dtype=inv_freq.dtype)

    freqs = torch.outer(t, inv_freq)

    return freqs


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = xq.dtype
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs_cis, freqs_cis), dim=-1)
    cos = emb.cos().to(dtype).to(xq.device)
    sin = emb.sin().to(dtype).to(xq.device)
    if position_ids is None:
        _, _, seq_len, _ = xq.shape
        cos, sin = cos[:seq_len], sin[:seq_len]
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)

    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed
