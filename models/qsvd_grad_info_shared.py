#qsvd_grad_info_shared.py
"""
Grad-info rank allocation for SHARED (1-down + 3-up) QSVD. NEW FILE.

Scoring identical to models/qsvd_grad_info.py (|S|*|S_grad|).
Budget cost per singular value uses paper α: **4E** (not 6E).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from models.QSVD_Transformer64_shared import (
    SharedAttention,
    collect_shared_attention_modules,
)
from models.qsvd_act_aware_calib import calib_qkv_input_scales_shared
from models.qsvd_grad_info import compute_importance_scores


def _unwrap(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def prepare_joint_svd_for_all_shared(
    model: nn.Module,
    act_aware: bool = False,
    scaling_by_attn: Optional[Dict[SharedAttention, torch.Tensor]] = None,
):
    attentions = collect_shared_attention_modules(_unwrap(model))
    for idx, attn in enumerate(attentions):
        if attn.qkv is None:
            raise RuntimeError(f"SharedAttention {idx} missing fused qkv for SVD.")
        scaling = None
        if scaling_by_attn is not None:
            scaling = scaling_by_attn.get(attn)
        if act_aware and scaling is None:
            raise RuntimeError(
                f"act_aware=True but no scaling_diag_matrix for SharedAttention {idx}. "
                f"Run calib_qkv_input_scales_shared first."
            )
        attn.prepare_qsvd_svd_info(
            act_aware=act_aware, scaling_diag_matrix=scaling
        )
        print(
            f"    [shared-grad] Prepared SVD Attention {idx}: "
            f"dim={attn.dim}, S.shape={tuple(attn.svd_info['S_full'].shape)}, "
            f"act_aware={act_aware}"
        )
    return attentions


def _freeze_all_but_fused_qkv_shared(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for attn in collect_shared_attention_modules(_unwrap(model)):
        if attn.qkv is None:
            continue
        attn.qkv.weight.requires_grad_(True)
        if attn.qkv.bias is not None:
            attn.qkv.bias.requires_grad_(True)
        attn.use_qsvd = False
        attn._qsvd_applied = False


@torch.enable_grad()
def calibrate_singular_grad_info_shared(
    model: nn.Module,
    calibration_batches: Sequence[torch.Tensor],
    loss_fn: nn.Module,
    device: Union[str, torch.device] = "cpu",
    accumulation_steps: int = 1,
) -> Dict[SharedAttention, torch.Tensor]:
    device = torch.device(device)
    model = model.to(device)
    model.train()
    _freeze_all_but_fused_qkv_shared(model)

    attentions = collect_shared_attention_modules(_unwrap(model))
    accum = {
        attn: torch.zeros(attn.dim, device=device, dtype=torch.float32)
        for attn in attentions
    }
    batch_count = 0
    n_updates = 0

    for batch in calibration_batches:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(device).float()
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(1)
        model.zero_grad(set_to_none=True)
        gen_imgs, _, _, _ = model(imgs)
        loss = loss_fn(gen_imgs, imgs) / float(accumulation_steps)
        loss.backward()
        batch_count += 1
        if batch_count % accumulation_steps != 0:
            continue
        n_updates += 1
        for attn in attentions:
            grad = attn.qkv.weight.grad
            if grad is None:
                continue
            grad = grad.detach().to(device=device, dtype=torch.float32)
            U = attn.svd_info["U_full"].to(device=device, dtype=torch.float32)
            V = attn.svd_info["V_full"].to(device=device, dtype=torch.float32)
            S_grad = torch.sum(U * (grad @ V), dim=0)
            accum[attn] += S_grad.pow(2)
        model.zero_grad(set_to_none=True)

    if n_updates == 0:
        raise RuntimeError("No grad updates during shared calib.")

    out = {}
    for attn in attentions:
        out[attn] = (accum[attn] / float(n_updates)).sqrt().cpu()
        print(
            f"    [shared-grad] dim={attn.dim}: "
            f"S_grad mean={out[attn].mean().item():.6e}"
        )
    return out


def allocate_ranks_shared_alpha_budget(
    importance_scores: Dict[SharedAttention, torch.Tensor],
    target_alpha_ratio: float,
    rank_align: int = 1,
    min_rank: int = 1,
) -> Dict[SharedAttention, torch.Tensor]:
    """
    Global allocation under paper α budget.

    Cost of one singular value at dim E: **4E** (shared down + 3 ups).
    Full α per layer: **3 E²**.
    """
    original_total = 0.0
    cost_per_sv = {}
    dims = {}
    for attn, scores in importance_scores.items():
        dim = int(scores.numel())
        dims[attn] = dim
        original_total += 3.0 * dim * dim
        cost_per_sv[attn] = 4.0 * dim

    target_total = float(target_alpha_ratio) * float(original_total)

    candidates = []
    for attn, scores in importance_scores.items():
        for idx in range(scores.numel()):
            candidates.append((attn, idx, float(scores[idx].item())))
    candidates.sort(key=lambda x: x[2], reverse=True)

    selected = {attn: [] for attn in importance_scores}
    current = 0.0
    for attn, sv_idx, _ in candidates:
        cost = cost_per_sv[attn]
        if current + cost <= target_total:
            selected[attn].append(sv_idx)
            current += cost

    for attn, scores in importance_scores.items():
        need = max(int(min_rank), 1) - len(selected[attn])
        if need <= 0:
            continue
        taken = set(selected[attn])
        remaining = [
            (i, float(scores[i].item())) for i in range(dims[attn]) if i not in taken
        ]
        remaining.sort(key=lambda t: t[1], reverse=True)
        for i, _ in remaining[:need]:
            selected[attn].append(i)
            current += cost_per_sv[attn]

    selected_tensors = {}
    for attn, indices in selected.items():
        dim = dims[attn]
        if not indices:
            indices = [0]
        indices_t = torch.tensor(sorted(set(indices)), dtype=torch.long)
        if rank_align > 1:
            n = int(indices_t.numel())
            aligned_n = min(int(np.ceil(n / rank_align) * rank_align), dim)
            if aligned_n > n:
                taken = set(indices_t.tolist())
                rem = sorted(
                    [i for i in range(dim) if i not in taken],
                    key=lambda i: float(importance_scores[attn][i].item()),
                    reverse=True,
                )
                indices_t = torch.tensor(
                    sorted(set(indices_t.tolist()) | set(rem[: aligned_n - n])),
                    dtype=torch.long,
                )
        selected_tensors[attn] = indices_t

    actual = sum(
        cost_per_sv[attn] * int(selected_tensors[attn].numel())
        for attn in selected_tensors
    )
    print(
        f"    [shared-grad] α allocation: target={target_alpha_ratio:.4f}, "
        f"actual={actual / original_total:.4f}, "
        f"params={int(actual)}/{int(original_total)}"
    )
    return selected_tensors


def apply_selected_indices_shared(
    attentions: Sequence[SharedAttention],
    selected: Dict[SharedAttention, torch.Tensor],
    sigma_fuse: str = "UV",
) -> List[int]:
    ranks = []
    for idx, attn in enumerate(attentions):
        indices = selected.get(attn)
        if indices is None or indices.numel() == 0:
            indices = torch.tensor([0], dtype=torch.long)
        attn.apply_shared_qsvd(selected_indices=indices, sigma_fuse=sigma_fuse)
        ranks.append(int(attn.qsvd_rank))
        print(
            f"    [shared-grad] Attention {idx}: rank {attn.qsvd_rank}/{attn.dim} "
            f"(indices[:8]={indices[:8].tolist()})"
        )
    return ranks


def run_shared_grad_info_pipeline(
    model: nn.Module,
    calibration_batches: Sequence,
    loss_fn: nn.Module,
    target_alpha_ratio: float = 0.75,
    device: Union[str, torch.device] = "cpu",
    sigma_fuse: str = "UV",
    grad_alpha: float = 1.0,
    rank_align: int = 1,
    act_aware: bool = False,
    accumulation_steps: int = 1,
    score_mode: str = "ne",
):
    scaling_by_attn = None
    if act_aware:
        print("[shared-grad] (0/4) Act-aware input-scale calib (ASVD-style)...")
        scaling_by_attn = calib_qkv_input_scales_shared(
            model,
            calibration_batches,
            device=device,
            method="abs_mean",
        )

    print("[shared-grad] (1/4) Joint QKV SVD...")
    attentions = prepare_joint_svd_for_all_shared(
        model, act_aware=act_aware, scaling_by_attn=scaling_by_attn
    )

    print("[shared-grad] (2/4) Calibrating S_grad (full recon)...")
    s_grad = calibrate_singular_grad_info_shared(
        model,
        calibration_batches,
        loss_fn,
        device=device,
        accumulation_steps=accumulation_steps,
    )

    print(f"[shared-grad] (3/4) Scores |S|*|S_grad|^alpha  mode={score_mode}...")
    # compute_importance_scores expects Attention sequence - SharedAttention is subclass
    importance = compute_importance_scores(
        attentions, s_grad, grad_alpha=grad_alpha, score_mode=score_mode
    )

    print("[shared-grad] (4/4) Allocate under α=4rE budget + apply shared...")
    selected = allocate_ranks_shared_alpha_budget(
        importance,
        target_alpha_ratio=target_alpha_ratio,
        rank_align=rank_align,
        min_rank=1,
    )
    ranks = apply_selected_indices_shared(attentions, selected, sigma_fuse=sigma_fuse)
    # Explicit Job-C-style summary line for before/after compute-aware score compares.
    dims = [int(a.dim) for a in attentions]
    ratios = [r / float(e) for r, e in zip(ranks, dims)]
    print(
        f"    [shared-grad] VERIFY ranks={ranks}  "
        f"r/E(min,max)=({min(ratios):.4f},{max(ratios):.4f})  "
        f"any_full={any(r >= e for r, e in zip(ranks, dims))}"
    )
    return ranks, selected, importance
