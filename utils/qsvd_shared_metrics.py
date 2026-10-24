"""
Paper α / γ / η metrics for SHARED joint QKV (NEW FILE).

Baseline (no SVD):     α=3E² , γ=3 L E² , η=2 L E
Separate SVD (3×B):    α=6rE , γ=6 L r E , η=2 r L
Joint QSVD (1+3):      α=4rE , γ=4 L r E , η=r L

We report the paper formulas (no MAC×2). Also report unique-param ratio vs 3E².
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch.nn as nn

from models.QSVD_Transformer64_shared import SharedAttention, collect_shared_attention_modules


def _unwrap(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def _stage_token_counts(model: nn.Module):
    core = _unwrap(model)
    bottom = getattr(core, "bottom_width", None)
    if bottom is None:
        args = getattr(core, "args", None)
        bottom = getattr(args, "bottom_width", 8) if args is not None else 8
    return [(bottom * s) ** 2 for s in (1, 2, 4, 8)]


def _stage_depths(model: nn.Module):
    args = getattr(_unwrap(model), "args", None)
    if args is not None and hasattr(args, "g_depth"):
        return [int(x) for x in str(args.g_depth).split(",")]
    return [5, 5, 5, 5]


def compute_shared_qsvd_metrics(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 1, 64, 64),
) -> Dict[str, float]:
    """
    Returns paper α, γ, η for QKV (shared) vs full-rank baseline formulas.
    """
    attentions = collect_shared_attention_modules(model)
    _, _, H, W = input_size
    if H % 64 or W % 64:
        raise ValueError(f"H/W must be multiples of 64, got {H}x{W}")
    num_patches = (H // 64) * (W // 64)
    token_counts = _stage_token_counts(model)
    depths = _stage_depths(model)

    alpha_cur = 0.0
    alpha_full = 0.0
    gamma_cur = 0.0
    gamma_full = 0.0
    eta_cur = 0.0
    eta_full = 0.0
    ranks = []

    stage_idx = 0
    block_in_stage = 0
    for attn in attentions:
        if stage_idx >= len(depths):
            break
        E = int(attn.dim)
        N = float(token_counts[stage_idx])
        r = int(getattr(attn, "qsvd_rank", E) or E)
        if not getattr(attn, "use_qsvd", False) or not getattr(attn, "_qsvd_applied", False):
            r = E
        ranks.append(r)

        # Prefer unique SharedQKV params if module present
        if getattr(attn, "shared_qkv", None) is not None and attn._qsvd_applied:
            alpha_cur += float(attn.shared_qkv.num_unique_params())
        else:
            # Fallback: 4 r E (no bias)
            alpha_cur += 4.0 * r * E
        alpha_full += 3.0 * E * E

        # Paper γ: 4 L r E vs 3 L E²  (L = N tokens here, × patches later)
        gamma_cur += 4.0 * N * r * E
        gamma_full += 3.0 * N * E * E

        # Paper η: r L vs 2 E L
        eta_cur += r * N
        eta_full += 2.0 * E * N

        block_in_stage += 1
        if block_in_stage >= depths[stage_idx]:
            stage_idx += 1
            block_in_stage = 0

    gamma_cur *= num_patches
    gamma_full *= num_patches
    eta_cur *= num_patches
    eta_full *= num_patches

    total_params = sum(p.numel() for p in model.parameters())

    def _ratio(a, b):
        return (a / b) if b > 0 else 0.0

    return {
        "total_params": float(total_params),
        "alpha_qkv": alpha_cur,
        "alpha_qkv_full": alpha_full,
        "alpha_ratio": _ratio(alpha_cur, alpha_full),
        "alpha_reduction_pct": (1.0 - _ratio(alpha_cur, alpha_full)) * 100.0,
        "gamma_qkv": gamma_cur,
        "gamma_qkv_full": gamma_full,
        "gamma_ratio": _ratio(gamma_cur, gamma_full),
        "gamma_reduction_pct": (1.0 - _ratio(gamma_cur, gamma_full)) * 100.0,
        "eta_qkv": eta_cur,
        "eta_qkv_full": eta_full,
        "eta_ratio": _ratio(eta_cur, eta_full),
        "eta_reduction_pct": (1.0 - _ratio(eta_cur, eta_full)) * 100.0,
        "num_attention_layers": float(len(attentions)),
        "num_patches": float(num_patches),
        "ranks": ranks,
    }


def print_shared_qsvd_metrics(model: nn.Module, input_size=(1, 1, 64, 64)):
    m = compute_shared_qsvd_metrics(model, input_size=input_size)
    print("=" * 60)
    print("SHARED QSVD Efficiency Metrics (paper alpha / gamma / eta)")
    print("=" * 60)
    print(f"Total parameters:     {m['total_params'] / 1e6:.3f} M")
    print(f"Attention layers:     {int(m['num_attention_layers'])}")
    print(f"Ranks:                {m['ranks']}")
    print("-" * 60)
    print("alpha weight size (QKV unique params = 4rE)")
    print(f"  alpha current/full: {m['alpha_qkv']/1e6:.4f} M / {m['alpha_qkv_full']/1e6:.4f} M")
    print(f"  alpha ratio:        {m['alpha_ratio']:.4f}  (reduction {m['alpha_reduction_pct']:.1f}%)")
    print("gamma compute cost (paper: 4LrE vs 3LE^2)")
    print(f"  gamma current/full: {m['gamma_qkv']/1e9:.4f} G / {m['gamma_qkv_full']/1e9:.4f} G")
    print(f"  gamma ratio:        {m['gamma_ratio']:.4f}  (reduction {m['gamma_reduction_pct']:.1f}%)")
    print("eta buffer (paper: rL vs 2LE — store C_qkv)")
    print(f"  eta current/full:   {m['eta_qkv']/1e6:.4f} M / {m['eta_qkv_full']/1e6:.4f} M")
    print(f"  eta ratio:          {m['eta_ratio']:.4f}  (reduction {m['eta_reduction_pct']:.1f}%)")
    print("=" * 60)
    return m
