"""
QSVD efficiency metrics.

Provides parameter counts and theoretical FLOPs for the transformer QKV
projections in CSFormer / CSFormer_QSVD. The numbers are focused on the
parts that QSVD actually changes; the CNN branch and unchanged transformer
layers are counted for total parameters but are not part of the FLOPs
comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel / DistributedDataParallel if needed."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def _get_attention_modules(model: nn.Module):
    """Return all QSVD-aware Attention modules in the transformer stem."""
    # Import here to avoid circular imports at package import time.
    from models.QSVD_Transformer64 import Attention
    return [m for m in _unwrap_model(model).modules() if isinstance(m, Attention)]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Return (total_params, trainable_params) for ``model``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_qkv_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in the transformer Q/K/V projections.

    Returns a dict with keys:
        - qkv_current: current number of qkv params
        - qkv_full_rank: number of qkv params if every layer were full rank
        - num_attention_layers: number of attention layers inspected
    """
    attentions = _get_attention_modules(model)
    current = 0
    full_rank = 0

    for attn in attentions:
        dim = attn.dim
        if attn.use_qsvd:
            if attn.adaptive_rank:
                # Effective rank is the number of selected singular values.
                mask = attn.get_qsvd_mask(force_eval=True)
                rank = int((mask > 0.5).sum().item()) if mask is not None else dim
                rank = max(1, min(rank, dim))
            else:
                # Each projection is two matrices: [dim, rank] and [rank, dim].
                rank = attn.qsvd_rank
            current += 3 * (dim * rank + rank * dim)
        else:
            # Fused qkv: [dim, 3*dim] plus optional bias.
            if attn.qkv is not None:
                current += sum(p.numel() for p in attn.qkv.parameters())
        full_rank += dim * (dim * 3)  # W_q, W_k, W_v each [dim, dim]
        if attn.qkv is not None and attn.qkv.bias is not None:
            full_rank += dim * 3

    return {
        'qkv_current': current,
        'qkv_full_rank': full_rank,
        'num_attention_layers': len(attentions),
    }


def compute_qkv_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 1, 64, 64),
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute theoretical QKV-projection FLOPs for a given input size.

    CSFormer splits the input into non-overlapping 64x64 patches and runs
    the transformer stem on each patch. This helper follows the same logic:
    it computes per-patch FLOPs and multiplies by the number of patches.

    Args:
        model: CSformer or CSformer_QSVD.
        input_size: (B, C, H, W). H and W must be multiples of 64.
        batch_size: if None, taken from input_size[0].

    Returns:
        dict with keys:
            - qkv_flops_current: current QKV FLOPs
            - qkv_flops_full_rank: QKV FLOPs if every attention were full rank
            - num_patches: number of 64x64 patches in the input
            - total_flops_current: same as qkv_flops_current (legacy name)
            - total_flops_full_rank: same as qkv_flops_full_rank (legacy name)
    """
    model = _unwrap_model(model)
    attentions = _get_attention_modules(model)

    if batch_size is None:
        batch_size = input_size[0]
    _, _, H, W = input_size
    if H % 64 != 0 or W % 64 != 0:
        raise ValueError(f"input_size H/W must be multiples of 64, got {H}x{W}")
    num_patches = (H // 64) * (W // 64)

    # Token counts for the four transformer stages.
    bottom_width = getattr(model, 'bottom_width', None)
    if bottom_width is None:
        bottom_width = getattr(model, 'args', None)
        if bottom_width is not None:
            bottom_width = getattr(bottom_width, 'bottom_width', 8)
        else:
            bottom_width = 8

    stage_scale = [1, 2, 4, 8]
    token_counts = [(bottom_width * s) ** 2 for s in stage_scale]

    # Stage output dimensions depend on the CSFormer_QSVD architecture.
    embed_dim = getattr(model, 'embed_dim', None)
    if embed_dim is None:
        args = getattr(model, 'args', None)
        if args is not None:
            embed_dim = getattr(args, 'gf_dim', 128)
        else:
            embed_dim = 128
    stage_dims = [embed_dim * 2, embed_dim, embed_dim // 2, embed_dim // 4]

    # Number of blocks per stage from args.g_depth.
    args = getattr(model, 'args', None)
    if args is not None and hasattr(args, 'g_depth'):
        depths = [int(x) for x in str(args.g_depth).split(',')]
    else:
        depths = [5, 5, 5, 5]

    flops_current = 0.0
    flops_full_rank = 0.0

    # Map each attention layer to its stage by counting blocks per stage.
    stage_idx = 0
    block_in_stage = 0
    for attn in attentions:
        if stage_idx >= len(depths):
            break
        dim = attn.dim
        N = token_counts[stage_idx]

        # Find rank: either stored rank or full rank.
        rank = getattr(attn, 'qsvd_rank', dim)
        if not attn.use_qsvd:
            rank = dim

        # Fused qkv: [B, N, dim] @ [dim, 3*dim] -> 2*B*N*dim*3*dim = 6*B*N*dim^2
        flops_full_rank += 6.0 * batch_size * N * dim * dim
        # Low-rank qkv: three projections, each two matmuls.
        # Per projection: 2*B*N*dim*rank + 2*B*N*rank*dim = 4*B*N*dim*rank
        # Three projections: 12*B*N*dim*rank
        flops_current += 12.0 * batch_size * N * dim * rank

        block_in_stage += 1
        if block_in_stage >= depths[stage_idx]:
            stage_idx += 1
            block_in_stage = 0

    # Multiply by the number of 64x64 patches in the full image.
    flops_current *= num_patches
    flops_full_rank *= num_patches

    return {
        'qkv_flops_current': flops_current,
        'qkv_flops_full_rank': flops_full_rank,
        'num_patches': num_patches,
        'total_flops_current': flops_current,
        'total_flops_full_rank': flops_full_rank,
    }


def compute_qsvd_metrics(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 1, 64, 64),
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper returning all QSVD efficiency metrics in one dict.

    Keys:
        - total_params
        - trainable_params
        - qkv_params
        - qkv_params_full_rank
        - qkv_param_ratio
        - qkv_param_reduction_pct
        - qkv_flops
        - qkv_flops_full_rank
        - qkv_flops_ratio
        - qkv_flops_reduction_pct
        - num_attention_layers
    """
    total, trainable = count_parameters(model)
    qkv_counts = count_qkv_parameters(model)
    flops = compute_qkv_flops(model, input_size=input_size, batch_size=batch_size)

    qkv_current = qkv_counts['qkv_current']
    qkv_full = qkv_counts['qkv_full_rank']
    flops_current = flops['qkv_flops_current']
    flops_full = flops['qkv_flops_full_rank']

    return {
        'total_params': total,
        'trainable_params': trainable,
        'qkv_params': qkv_current,
        'qkv_params_full_rank': qkv_full,
        'qkv_param_ratio': qkv_current / qkv_full if qkv_full > 0 else 1.0,
        'qkv_param_reduction_pct': (1.0 - qkv_current / qkv_full) * 100 if qkv_full > 0 else 0.0,
        'qkv_flops': flops_current,
        'qkv_flops_full_rank': flops_full,
        'qkv_flops_ratio': flops_current / flops_full if flops_full > 0 else 1.0,
        'qkv_flops_reduction_pct': (1.0 - flops_current / flops_full) * 100 if flops_full > 0 else 0.0,
        'num_attention_layers': qkv_counts['num_attention_layers'],
        'num_patches': flops['num_patches'],
    }


def print_qsvd_metrics(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 1, 64, 64)):
    """Print a readable summary of QSVD efficiency metrics."""
    metrics = compute_qsvd_metrics(model, input_size=input_size)
    print("=" * 60)
    print("QSVD Efficiency Metrics")
    print("=" * 60)
    print(f"Total parameters:        {metrics['total_params'] / 1e6:.3f} M")
    print(f"Trainable parameters:    {metrics['trainable_params'] / 1e6:.3f} M")
    print(f"Attention layers:        {metrics['num_attention_layers']}")
    print(f"Reference input size:    {input_size}")
    print(f"64x64 patches per image: {metrics['num_patches']}")
    print("-" * 60)
    print("QKV projection only (the part QSVD changes)")
    print("-" * 60)
    print(f"QKV params current:      {metrics['qkv_params'] / 1e6:.3f} M")
    print(f"QKV params full-rank:    {metrics['qkv_params_full_rank'] / 1e6:.3f} M")
    print(f"QKV param ratio:         {metrics['qkv_param_ratio']:.3f}")
    print(f"QKV param reduction:     {metrics['qkv_param_reduction_pct']:.1f}%")
    print(f"QKV FLOPs current:       {metrics['qkv_flops'] / 1e9:.3f} G")
    print(f"QKV FLOPs full-rank:     {metrics['qkv_flops_full_rank'] / 1e9:.3f} G")
    print(f"QKV FLOPs ratio:         {metrics['qkv_flops_ratio']:.3f}")
    print(f"QKV FLOPs reduction:     {metrics['qkv_flops_reduction_pct']:.1f}%")
    print("=" * 60)


# ====================== Adaptive rank allocation losses ======================

def count_qkv_parameters_adaptive(model: nn.Module) -> Tuple[int, int, int]:
    """
    Count QKV parameters using the learned adaptive masks.

    Returns:
        (current_qkv_params, full_qkv_params, total_selected_ranks)
    """
    from models.QSVD_Transformer64 import Attention
    attentions = [m for m in _unwrap_model(model).modules() if isinstance(m, Attention)]

    current = 0
    full = 0
    total_ranks = 0

    for attn in attentions:
        dim = attn.dim
        full += dim * (dim * 3)
        if attn.qkv is not None and attn.qkv.bias is not None:
            full += dim * 3

        if not attn.adaptive_rank:
            rank = getattr(attn, 'qsvd_rank', dim)
            current += 3 * (dim * rank + rank * dim)
            total_ranks += rank
        else:
            mask = attn.get_qsvd_mask()
            if mask is None:
                rank = dim
            else:
                rank = int((mask > 0.5).sum().item())
                rank = max(1, min(rank, dim))
            current += 3 * (dim * rank + rank * dim)
            total_ranks += rank

    return current, full, total_ranks


def compute_adaptive_rank_losses(
    model: nn.Module,
    target_param_ratio: float = 0.75,
    gamma_align: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute compression and alignment losses for adaptive rank allocation.

    Args:
        model: CSformer_QSVD model with adaptive-rank Attention modules.
        target_param_ratio: desired QKV parameter ratio vs full-rank QKV.
        gamma_align: weight for the alignment loss.

    Returns:
        (compression_loss, alignment_loss, current_param_ratio)
    """
    from models.QSVD_Transformer64 import Attention
    attentions = [m for m in _unwrap_model(model).modules() if isinstance(m, Attention)]
    if not any(attn.adaptive_rank for attn in attentions):
        return torch.tensor(0.0), torch.tensor(0.0), 1.0

    device = next(model.parameters()).device

    # Compression regularizer.
    total_current = 0.0
    total_full = 0.0
    for attn in attentions:
        dim = attn.dim
        full = dim * (dim * 3)
        if attn.qkv is not None and attn.qkv.bias is not None:
            full += dim * 3
        total_full += full

        if attn.adaptive_rank:
            mask = attn.get_qsvd_mask()
            if mask is None:
                rank = dim
            else:
                rank = mask.sum()
            current = rank * (6 * dim)
        else:
            rank = getattr(attn, 'qsvd_rank', dim)
            current = 3 * (dim * rank + rank * dim)
        total_current += current

    current_ratio = total_current / total_full if total_full > 0 else 1.0
    target_params = target_param_ratio * total_full

    # Penalize only when we are *above* the target ratio (too few params removed).
    # If we are below target, the ratio is clamped to the target so the loss is zero.
    clamped_current = max(total_current, target_params)
    clamped_ratio = clamped_current / total_full if total_full > 0 else target_param_ratio
    compression_loss = torch.log(
        torch.as_tensor(clamped_ratio / target_param_ratio, device=device, dtype=torch.float32)
    )

    # Alignment loss: encourage the soft mask to select large singular values.
    alignment_loss = torch.tensor(0.0, device=device)
    num_adaptive = 0
    for attn in attentions:
        if not attn.adaptive_rank or attn.svd_info is None:
            continue
        S = attn.svd_info['S_full'].to(device).float()
        soft_mask = getattr(attn, 'qsvd_soft_mask_cache', None)
        if soft_mask is None:
            soft_mask = attn.get_qsvd_mask()
        if soft_mask is None:
            continue

        # Build a target that keeps the top-k singular values, where k is the
        # number of singular values currently selected by the soft mask.
        with torch.no_grad():
            k = max(1, min(int(soft_mask.sum().item()), S.numel()))
            _, topk_indices = torch.topk(S, k)
            target_mask = torch.zeros_like(S)
            target_mask[topk_indices] = 1.0

        alignment_loss += F.mse_loss(soft_mask * S, target_mask * S, reduction='mean')
        num_adaptive += 1

    if num_adaptive > 0:
        alignment_loss = alignment_loss / num_adaptive

    return compression_loss, gamma_align * alignment_loss, current_ratio
