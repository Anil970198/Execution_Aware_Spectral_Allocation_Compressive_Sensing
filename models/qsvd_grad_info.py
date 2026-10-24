#qsvd_grad_info.py
"""
QSVD-style cross-layer rank allocation for CSFormer (NEW FILE).

Faithful adaptation of QSVD-main/fake_quant/grad_info_utils.py:
  1. Joint QKV SVD (already via Attention.prepare_qsvd_svd_info)
  2. Calibrate with FULL-MODEL reconstruction loss (CS domain analogue of LM CE)
  3. Per-singular-value score = |S| * |S_grad|^grad_alpha
     where S_grad_i = diag(U^T G V)_i  via  sum(U * (G @ V), dim=0)
  4. Global top selection under a QKV parameter budget

Does NOT modify QSVD_Transformer64.py or the spectral Job2 path.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from models.QSVD_Transformer64 import (
    Attention,
    collect_attention_modules,
)


def _unwrap(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def prepare_joint_svd_for_all(
    model: nn.Module,
    act_aware: bool = False,
) -> List[Attention]:
    """
    Run joint QKV SVD on every Attention and store svd_info (U_full, S_full, V_full).

    Must be called while original fused attn.qkv still exists (before discard).
    """
    attentions = collect_attention_modules(_unwrap(model))
    for idx, attn in enumerate(attentions):
        if attn.qkv is None:
            raise RuntimeError(
                f"Attention {idx} has no fused qkv; cannot prepare SVD for grad-info."
            )
        attn.prepare_qsvd_svd_info(act_aware=act_aware)
        print(f"    [grad-info] Prepared SVD for Attention {idx}: "
              f"dim={attn.dim}, S.shape={tuple(attn.svd_info['S_full'].shape)}")
    return attentions


def _freeze_all_but_fused_qkv(model: nn.Module) -> None:
    """Match QSVD: only QKV weights receive gradients during calib."""
    for p in model.parameters():
        p.requires_grad_(False)
    for attn in collect_attention_modules(_unwrap(model)):
        if attn.qkv is None:
            continue
        attn.qkv.weight.requires_grad_(True)
        if attn.qkv.bias is not None:
            attn.qkv.bias.requires_grad_(True)
        # Ensure forward uses fused qkv (not low-rank path).
        attn.use_qsvd = False
        attn._qsvd_applied = False


@torch.enable_grad()
def calibrate_singular_grad_info(
    model: nn.Module,
    calibration_batches: Sequence[torch.Tensor],
    loss_fn: nn.Module,
    device: Union[str, torch.device] = "cpu",
    accumulation_steps: int = 1,
) -> Dict[Attention, torch.Tensor]:
    """
    Accumulate S_grad_info = sqrt(mean_n (diag(U^T G V))^2 ) per Attention.

    This mirrors grad_info_utils.calib_grad_info (without |S| yet).

    Args:
        model: CSformer_QSVD with svd_info prepared and fused qkv present.
        calibration_batches: list of image tensors (B, 1, H, W), H/W multiples of 64.
        loss_fn: reconstruction loss module (e.g. MSELoss / L1Loss).
        device: cpu or cuda.
        accumulation_steps: gradient accumulation (QSVD default 1).

    Returns:
        {Attention: S_grad_info [dim] on CPU}
    """
    device = torch.device(device)
    model = model.to(device)
    model.train()
    _freeze_all_but_fused_qkv(model)

    attentions = collect_attention_modules(_unwrap(model))
    for attn in attentions:
        if attn.svd_info is None:
            raise RuntimeError("svd_info missing; call prepare_joint_svd_for_all first.")

    # Accumulators on device
    accum = {
        attn: torch.zeros(attn.dim, device=device, dtype=torch.float32)
        for attn in attentions
    }
    batch_count = 0
    n_updates = 0

    for batch in calibration_batches:
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch
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
            # Fused weight is already [3*dim, dim] — same layout as cat(Gq,Gk,Gv).
            U = attn.svd_info["U_full"].to(device=device, dtype=torch.float32)
            V = attn.svd_info["V_full"].to(device=device, dtype=torch.float32)
            # Efficient diag(U^T @ G @ V)
            S_grad = torch.sum(U * (grad @ V), dim=0)
            accum[attn] += S_grad.pow(2)

        model.zero_grad(set_to_none=True)

    if n_updates == 0:
        raise RuntimeError(
            "No gradient updates during calib; check calibration_batches / accumulation_steps."
        )

    s_grad_info: Dict[Attention, torch.Tensor] = {}
    for attn in attentions:
        s_grad_info[attn] = (accum[attn] / float(n_updates)).sqrt().cpu()
        print(f"    [grad-info] Attention dim={attn.dim}: "
              f"S_grad rms mean={s_grad_info[attn].mean().item():.6e}, "
              f"max={s_grad_info[attn].max().item():.6e}")

    return s_grad_info


# Stage channel dim E → token count N used by paper γ metrics
# (matches utils.qsvd_shared_metrics._stage_token_counts for bottom_width=8).
# Used for γ/η reporting and legacy score_mode "ne" only — NOT classical_4e.
_N_FROM_E = {256: 64, 128: 256, 64: 1024, 32: 4096}


def compute_importance_scores(
    attentions: Sequence[Attention],
    s_grad_info: Dict[Attention, torch.Tensor],
    grad_alpha: float = 1.0,
    score_mode: str = "ne",
) -> Dict[Attention, torch.Tensor]:
    """
    Per-singular importance for greedy α packing.

    score_mode:
      ne / n_e / legacy_knapsack (LOCKED default):
        s = (σ · |g|^β) / (N · E)     ≡ λ=1 compute penalty
      classical_4e / 4e (ablation only — DO NOT use for new locked runs):
        s = (σ · |g|^β) / (4 E)
      sigma / s / mag (ablation ladder):
        s = |σ|
      sigma_g / sg / sigma_grad (ablation ladder):
        s = |σ| · |g|^β               ≡ λ=0 (no NE denom)
      ne_l025 / ne_l05 / ne_l075:
        s = (σ · |g|^β) / (N · E)^λ   with λ ∈ {0.25, 0.5, 0.75}
      rn / r_over_n / n:
        s = (σ · |g|^β) / N
      re / r_over_e / e:
        s = (σ · |g|^β) / E
    """
    mode = str(score_mode).strip().lower()
    lambda_ne = None  # if set: divide by (N*E)^lambda_ne

    if mode in ("classical_4e", "4e"):
        mode_key = "classical_4e"
    elif mode in ("ne", "n_e", "legacy_knapsack"):
        mode_key = "ne"
        lambda_ne = 1.0
    elif mode in ("sigma", "s", "mag", "singular"):
        mode_key = "sigma"
    elif mode in ("sigma_g", "sg", "sigma_grad", "mag_grad"):
        mode_key = "sigma_g"
    elif mode in ("ne_l025", "ne_lambda025", "ne_l0.25", "lambda025"):
        mode_key = "ne_lambda"
        lambda_ne = 0.25
    elif mode in ("ne_l05", "ne_l050", "ne_lambda05", "ne_l0.5", "lambda05"):
        mode_key = "ne_lambda"
        lambda_ne = 0.5
    elif mode in ("ne_l075", "ne_lambda075", "ne_l0.75", "lambda075"):
        mode_key = "ne_lambda"
        lambda_ne = 0.75
    elif mode in ("rn", "r_over_n", "n", "div_n"):
        mode_key = "rn"
    elif mode in ("re", "r_over_e", "e", "div_e"):
        mode_key = "re"
    else:
        raise ValueError(
            f"Unknown score_mode={score_mode!r}; "
            f"use ne|classical_4e|sigma|sigma_g|ne_l025|ne_l05|ne_l075|rn|re"
        )

    scores: Dict[Attention, torch.Tensor] = {}
    for attn in attentions:
        if attn.svd_info is None:
            raise RuntimeError("svd_info missing for importance scoring.")
        if attn not in s_grad_info:
            raise RuntimeError("S_grad_info missing for an Attention module.")
        S = attn.svd_info["S_full"].float().abs()
        S_grad = s_grad_info[attn].float().abs()
        if S.numel() != S_grad.numel():
            raise RuntimeError(
                f"S ({S.numel()}) and S_grad ({S_grad.numel()}) length mismatch."
            )
        E = float(attn.dim)
        if int(S.numel()) != int(attn.dim):
            raise RuntimeError(
                f"S length ({S.numel()}) != attn.dim ({attn.dim}); "
                f"cannot use knapsack scoring."
            )
        val = S * (S_grad ** float(grad_alpha))
        if mode_key == "sigma":
            scores[attn] = S.clone()
            label = "|S|"
        elif mode_key == "sigma_g":
            scores[attn] = val
            label = "|S|*|S_grad|^alpha"
        elif mode_key == "classical_4e":
            scores[attn] = val / (4.0 * E)
            label = "|S|*|S_grad|^alpha / (4*E)"
        elif mode_key == "re":
            scores[attn] = val / E
            label = "|S|*|S_grad|^alpha / E"
        elif mode_key in ("ne", "ne_lambda", "rn"):
            if int(attn.dim) not in _N_FROM_E:
                raise RuntimeError(
                    f"Unexpected attn.dim={attn.dim}; cannot resolve N for score."
                )
            N = float(_N_FROM_E[int(attn.dim)])
            if mode_key == "rn":
                scores[attn] = val / N
                label = f"|S|*|S_grad|^alpha / N  N={N:g}"
            else:
                # ne (λ=1) or ne_lambda (λ∈{0.25,0.5,0.75})
                lam = 1.0 if lambda_ne is None else float(lambda_ne)
                denom = (N * E) ** lam
                scores[attn] = val / denom
                label = (
                    f"|S|*|S_grad|^alpha / (N*E)^{lam:g}  N={N:g}"
                )
        else:
            raise RuntimeError(f"Unhandled mode_key={mode_key}")
        print(
            f"    [grad-info] Scores {label}  "
            f"mode={mode_key} dim={attn.dim} E={E:g}: "
            f"mean={scores[attn].mean().item():.6e}, "
            f"top5={scores[attn].topk(min(5, scores[attn].numel())).values.tolist()}"
        )
    return scores


def allocate_ranks_global_param_budget(
    importance_scores: Dict[Attention, torch.Tensor],
    target_param_ratio: float,
    rank_align: int = 1,
    min_rank: int = 1,
) -> Dict[Attention, torch.Tensor]:
    """
    Global cross-layer allocation under a QKV *parameter* budget.

    Cost of keeping one singular value in a layer of dim E:
      3 projections * (E*r + r*E) / r  =>  6E params per SV
    Original fused QKV params per layer: 3 * E * E

    Differs from QSVD's legacy SV-count budget (rank_ratio/2 * L * E) but
    matches CSFormer's uniform/Job comparisons at target_param_ratio (e.g. 0.75).

    Unlike Job2 spectral path: NO per-layer score normalization, NO floor
    fraction — pure global sort by importance (QSVD spirit). min_rank=1 avoids
    empty layers which would break CSFormer forward.
    """
    if not importance_scores:
        return {}

    original_total = 0
    cost_per_sv = {}
    dims = {}
    for attn, scores in importance_scores.items():
        dim = int(scores.numel())
        dims[attn] = dim
        original_total += 3 * dim * dim
        cost_per_sv[attn] = 6 * dim

    target_total = float(target_param_ratio) * float(original_total)

    candidates = []
    for attn, scores in importance_scores.items():
        for idx in range(scores.numel()):
            candidates.append((attn, idx, float(scores[idx].item())))
    candidates.sort(key=lambda x: x[2], reverse=True)

    selected = {attn: [] for attn in importance_scores}
    current = 0.0
    for attn, sv_idx, _score in candidates:
        cost = cost_per_sv[attn]
        if current + cost <= target_total:
            selected[attn].append(sv_idx)
            current += cost

    # Ensure every layer has at least min_rank (steal from lowest global scores
    # already chosen only if needed — otherwise pick next-best unused for that layer).
    for attn, scores in importance_scores.items():
        dim = dims[attn]
        need = max(int(min_rank), 1) - len(selected[attn])
        if need <= 0:
            continue
        taken = set(selected[attn])
        remaining = [
            (i, float(scores[i].item()))
            for i in range(dim) if i not in taken
        ]
        remaining.sort(key=lambda t: t[1], reverse=True)
        for i, _ in remaining[:need]:
            selected[attn].append(i)
            current += cost_per_sv[attn]

    selected_tensors: Dict[Attention, torch.Tensor] = {}
    for attn, indices in selected.items():
        dim = dims[attn]
        if not indices:
            indices = [0]
        indices_t = torch.tensor(sorted(set(indices)), dtype=torch.long)

        if rank_align > 1:
            n = int(indices_t.numel())
            aligned_n = int(np.ceil(n / rank_align) * rank_align)
            aligned_n = min(aligned_n, dim)
            if aligned_n > n:
                taken = set(indices_t.tolist())
                remaining = [
                    i for i in range(dim) if i not in taken
                ]
                remaining.sort(
                    key=lambda i: float(importance_scores[attn][i].item()),
                    reverse=True,
                )
                extra = remaining[: aligned_n - n]
                indices_t = torch.tensor(
                    sorted(set(indices_t.tolist()) | set(extra)),
                    dtype=torch.long,
                )

        selected_tensors[attn] = indices_t

    actual_params = sum(
        len(selected_tensors[attn]) * cost_per_sv[attn]
        for attn in selected_tensors
    )
    actual_ratio = actual_params / original_total if original_total > 0 else 0.0
    print(f"    [grad-info] Global allocation: target={target_param_ratio:.4f}, "
          f"actual={actual_ratio:.4f}, params={int(actual_params)}/{int(original_total)}")
    return selected_tensors


def apply_selected_indices(
    attentions: Sequence[Attention],
    selected: Dict[Attention, torch.Tensor],
    sigma_fuse: str = "UV",
) -> List[int]:
    """Apply selected singular indices via Attention.apply_qsvd; return ranks."""
    ranks: List[int] = []
    for idx, attn in enumerate(attentions):
        indices = selected.get(attn, None)
        if indices is None or len(indices) == 0:
            indices = torch.arange(1, dtype=torch.long)
        attn.apply_qsvd(selected_indices=indices, sigma_fuse=sigma_fuse)
        ranks.append(int(attn.qsvd_rank))
        print(f"    [grad-info] Attention {idx}: rank {attn.qsvd_rank}/{attn.dim} "
              f"(indices[:8]={indices[:8].tolist()})")
    return ranks


def run_grad_info_pipeline(
    model: nn.Module,
    calibration_batches: Sequence[torch.Tensor],
    loss_fn: nn.Module,
    target_param_ratio: float = 0.75,
    device: Union[str, torch.device] = "cpu",
    sigma_fuse: str = "UV",
    grad_alpha: float = 1.0,
    rank_align: int = 1,
    act_aware: bool = False,
    accumulation_steps: int = 1,
) -> Tuple[List[int], Dict[Attention, torch.Tensor], Dict[Attention, torch.Tensor]]:
    """
    End-to-end: prepare SVD → calib grads → |S|*|S_grad|^α → allocate → apply.

    Returns:
        ranks, selected_indices, importance_scores
    """
    print("[grad-info] (1/4) Joint QKV SVD...")
    attentions = prepare_joint_svd_for_all(model, act_aware=act_aware)

    print("[grad-info] (2/4) Calibrating singular gradients (full recon loss)...")
    s_grad_info = calibrate_singular_grad_info(
        model,
        calibration_batches,
        loss_fn=loss_fn,
        device=device,
        accumulation_steps=accumulation_steps,
    )

    print("[grad-info] (3/4) Computing |S| * |S_grad|^alpha scores...")
    importance = compute_importance_scores(
        attentions, s_grad_info, grad_alpha=grad_alpha, score_mode="ne"
    )

    print("[grad-info] (4/4) Global rank allocation + apply...")
    selected = allocate_ranks_global_param_budget(
        importance,
        target_param_ratio=target_param_ratio,
        rank_align=rank_align,
        min_rank=1,
    )
    ranks = apply_selected_indices(attentions, selected, sigma_fuse=sigma_fuse)
    return ranks, selected, importance
