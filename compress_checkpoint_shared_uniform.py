"""
Compress baseline CSFormer → CSformer_QSVD_Shared with UNIFORM ranks.

Paper graph: 1 shared down + 3 ups. Rank formula = prior jobs (_compute_rank).
NEW FILE — does not touch compress_checkpoint.py.
"""

from __future__ import annotations

import argparse
import copy
import math
import os

import torch
import torch.nn as nn

import cfg
import models.CSformer as CSformer
import models.CSformer_QSVD_Shared as CSformer_QSVD_Shared
from models.QSVD_Transformer64_shared import (
    apply_shared_qsvd_to_model,
    discard_original_qkv_in_shared_model,
    collect_shared_attention_modules,
)
from utils.qsvd_shared_metrics import print_shared_qsvd_metrics


def _strip_module_prefix(state_dict):
    if not list(state_dict.keys())[0].startswith("module."):
        return state_dict
    return {k[7:]: v for k, v in state_dict.items()}


def _filter_compatible(model, state_dict):
    model_sd = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state_dict.items():
        if k not in model_sd or tuple(model_sd[k].shape) != tuple(v.shape):
            skipped.append(k)
            continue
        filtered[k] = v
    return filtered, skipped


def _set_n_input(args):
    ratio_dict = {1: 3, 4: 11, 10: 26, 25: 64, 30: 77, 40: 103, 50: 128}
    args.n_input = ratio_dict[args.cs_ratio]
    return args


def _rank_from_shared_alpha(dim, target_alpha, rank_align=1):
    """Convert the paper shared-QKV ratio alpha=4rE/(3E^2) to rank r.

    The old ``qsvd_rank_ratio`` flag follows the separate-SVD legacy rule
    r=E*ratio/2.  Reusing 0.75 there therefore gives shared alpha=0.50.
    This helper uses the shared 1-down/3-up cost directly:

        r = 3 * target_alpha * E / 4.

    We round down to the requested alignment so the realized parameter ratio
    never exceeds the requested budget.
    """
    dim = int(dim)
    rank_align = max(1, int(rank_align))
    raw_rank = 0.75 * float(target_alpha) * dim
    rank = int(math.floor(raw_rank / rank_align) * rank_align)
    return max(1, min(rank, dim))


def main():
    parser = argparse.ArgumentParser(description="Shared QSVD uniform compression")
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--reference_input_size", type=int, nargs=4, default=[1, 1, 64, 64])
    parser.add_argument(
        "--qsvd_target_alpha_ratio",
        type=float,
        default=None,
        help=(
            "Target shared-QKV parameter ratio alpha=4rE/(3E^2). "
            "For alpha=0.75 this gives ranks {144,72,36,18}. "
            "If omitted, the legacy --qsvd_rank_ratio behavior is retained."
        ),
    )
    args, remaining = parser.parse_known_args()
    cfg_parser = cfg.parse_args(remaining)
    for k, v in vars(cfg_parser).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args = _set_n_input(args)

    args.gen_model = "CSformer_QSVD_Shared"
    args.use_qsvd = False  # build fused first
    args.use_sib_acs = False
    args.qsvd_adaptive_rank = False

    save_path = os.path.join(args.save_dir, "Model", "checkpoint_best.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("=" * 70)
    print("SHARED QSVD Uniform Compression (1 down + 3 ups)")
    print("=" * 70)
    print(f"Baseline: {args.load_path}")
    print(f"Save:     {save_path}")
    target_alpha = args.qsvd_target_alpha_ratio
    if target_alpha is not None and not (0.0 < float(target_alpha) <= 1.0):
        raise ValueError(
            f"--qsvd_target_alpha_ratio must be in (0,1], got {target_alpha}"
        )
    if target_alpha is None:
        print(
            f"CS={args.cs_ratio}  legacy rank_ratio={args.qsvd_rank_ratio}; "
            "shared alpha will equal approximately 2*rank_ratio/3"
        )
    else:
        # Preserve constructor/checkpoint compatibility with the old rank flag.
        # Legacy r=E*ratio/2 equals shared r=3*alpha*E/4 when ratio=1.5*alpha.
        args.qsvd_rank_ratio = 1.5 * float(target_alpha)
        print(
            f"CS={args.cs_ratio}  target shared alpha={float(target_alpha):.6f}; "
            f"compatibility rank_ratio={args.qsvd_rank_ratio:.6f}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.load_path, map_location="cpu")
    baseline_args = copy.copy(args)
    baseline_args.gen_model = "CSformer"
    baseline_args.use_qsvd = False
    baseline = CSformer.CSformer(baseline_args)
    sd = _strip_module_prefix(ckpt["gen_state_dict"] if "gen_state_dict" in ckpt else ckpt)
    filtered, skipped = _filter_compatible(baseline, sd)
    baseline.load_state_dict(filtered, strict=False)
    print(f"Baseline loaded (skipped {len(skipped)} keys)")

    shared_model = CSformer_QSVD_Shared.CSformer_QSVD_Shared(args)
    sd_b = baseline.state_dict()
    filtered2, skipped2 = _filter_compatible(shared_model, sd_b)
    shared_model.load_state_dict(filtered2, strict=False)
    print(f"Copied into Shared model (skipped {len(skipped2)} keys)")

    print("Applying shared QSVD (uniform ranks)...")
    args.use_qsvd = True
    if target_alpha is None:
        apply_shared_qsvd_to_model(
            shared_model,
            rank_ratio=float(args.qsvd_rank_ratio),
            sigma_fuse=getattr(args, "qsvd_sigma_fuse", "UV"),
            act_aware=bool(getattr(args, "qsvd_act_aware", False)),
            rank_align=int(getattr(args, "qsvd_rank_align", 1)),
        )
    else:
        rank_align = int(getattr(args, "qsvd_rank_align", 1))
        for attn in collect_shared_attention_modules(shared_model):
            rank = _rank_from_shared_alpha(
                attn.dim, float(target_alpha), rank_align=rank_align
            )
            attn.apply_shared_qsvd(
                rank_or_ratio=rank,
                sigma_fuse=getattr(args, "qsvd_sigma_fuse", "UV"),
                act_aware=bool(getattr(args, "qsvd_act_aware", False)),
                rank_align=rank_align,
            )
    discard_original_qkv_in_shared_model(shared_model)

    ranks = [a.qsvd_rank for a in collect_shared_attention_modules(shared_model)]
    print(f"Ranks: {ranks}")
    metrics = print_shared_qsvd_metrics(shared_model, tuple(args.reference_input_size))
    if target_alpha is not None:
        realized_alpha = float(metrics["alpha_ratio"])
        print(
            f"Requested shared alpha={float(target_alpha):.6f}; "
            f"realized shared alpha={realized_alpha:.6f}"
        )
        if realized_alpha > float(target_alpha) + 1e-8:
            raise RuntimeError(
                f"Realized alpha {realized_alpha:.8f} exceeds target "
                f"{float(target_alpha):.8f}"
            )

    args.use_qsvd = True
    args.qsvd_discard_original = True
    args.gen_model = "CSformer_QSVD_Shared"
    torch.save(
        {
            "epoch": 0,
            "gen_state_dict": shared_model.state_dict(),
            "best_psnr": 0.0,
            "best_ssim": 0.0,
            "qsvd_discard_original": True,
            "qsvd_ranks": ranks,
            "qsvd_shared": True,
            "qsvd_target_alpha_ratio": target_alpha,
            "args": vars(args),
            "path_helper": {
                "prefix": args.save_dir,
                "ckpt_path": os.path.join(args.save_dir, "Model"),
                "log_path": os.path.join(args.save_dir, "Log"),
                "sample_path": os.path.join(args.save_dir, "Samples"),
            },
            "shared_metrics": {k: v for k, v in metrics.items() if k != "ranks"},
            "importance_method": "shared_uniform",
        },
        save_path,
    )
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
