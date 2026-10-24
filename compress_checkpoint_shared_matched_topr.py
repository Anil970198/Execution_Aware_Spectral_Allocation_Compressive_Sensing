"""
Matched-rank ablation: same per-layer ranks as a reference knapsack ckpt,
but keep conventional top-r singular directions (not selected S_ℓ).

Loads qsvd_ranks from --ranks_from_ckpt (compressed or FT shared ckpt),
applies shared QSVD with rank_or_ratio=int(r) per layer → indices 0..r-1.
"""

from __future__ import annotations

import argparse
import copy
import os

import torch

import cfg
import models.CSformer as CSformer
import models.CSformer_QSVD_Shared as CSformer_QSVD_Shared
from models.QSVD_Transformer64_shared import (
    collect_shared_attention_modules,
    discard_original_qkv_in_shared_model,
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


def _load_ranks(ranks_ckpt: str):
    ckpt = torch.load(ranks_ckpt, map_location="cpu")
    ranks = ckpt.get("qsvd_ranks", None)
    if ranks is None:
        raise RuntimeError(
            f"No qsvd_ranks in {ranks_ckpt}. "
            "Pass a shared knapsack compressed/FT checkpoint."
        )
    ranks = [int(r) for r in ranks]
    selected = ckpt.get("qsvd_selected_indices", None)
    return ranks, selected, ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Shared QSVD matched-rank top-r compression"
    )
    parser.add_argument("--load_path", type=str, required=True,
                        help="Dense baseline CSformer checkpoint")
    parser.add_argument("--ranks_from_ckpt", type=str, required=True,
                        help="Reference shared ckpt with qsvd_ranks")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--reference_input_size", type=int, nargs=4, default=[1, 1, 64, 64]
    )
    args, remaining = parser.parse_known_args()
    cfg_parser = cfg.parse_args(remaining)
    for k, v in vars(cfg_parser).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args = _set_n_input(args)

    sigma_fuse = getattr(args, "qsvd_sigma_fuse", "UV")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(args.save_dir, "Model", "checkpoint_best.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ranks, ref_selected, ref_ckpt = _load_ranks(args.ranks_from_ckpt)
    print("=" * 70)
    print("SHARED QSVD Matched-rank TOP-r (same r_ℓ, indices 0..r-1)")
    print("=" * 70)
    print(f"Baseline:     {args.load_path}")
    print(f"Ranks from:   {args.ranks_from_ckpt}")
    print(f"Per-layer r:  {ranks}")
    print(f"Device:       {device}")

    ckpt = torch.load(args.load_path, map_location="cpu")
    base_args = copy.copy(args)
    base_args.gen_model = "CSformer"
    base_args.use_qsvd = False
    baseline = CSformer.CSformer(base_args)
    sd = _strip_module_prefix(ckpt.get("gen_state_dict", ckpt))
    filtered, _ = _filter_compatible(baseline, sd)
    baseline.load_state_dict(filtered, strict=False)

    args.gen_model = "CSformer_QSVD_Shared"
    args.use_qsvd = False
    args.use_sib_acs = False
    args.qsvd_adaptive_rank = False
    shared_model = CSformer_QSVD_Shared.CSformer_QSVD_Shared(args)
    filtered2, _ = _filter_compatible(shared_model, baseline.state_dict())
    shared_model.load_state_dict(filtered2, strict=False)
    shared_model = shared_model.to(device)

    attentions = collect_shared_attention_modules(shared_model)
    if len(attentions) != len(ranks):
        raise RuntimeError(
            f"Rank list length {len(ranks)} != #attentions {len(attentions)}"
        )

    selected_lists = []
    for attn, r in zip(attentions, ranks):
        r_i = max(1, min(int(r), int(attn.dim)))
        attn.apply_shared_qsvd(rank_or_ratio=r_i, sigma_fuse=sigma_fuse)
        # Convention: top-r ⇒ indices [0, r)
        selected_lists.append(list(range(r_i)))
        print(f"  layer dim={attn.dim}: top-r={r_i}")

    discard_original_qkv_in_shared_model(shared_model)
    shared_model = shared_model.cpu()
    metrics = print_shared_qsvd_metrics(shared_model, tuple(args.reference_input_size))

    target_ratio = float(
        getattr(args, "qsvd_target_param_ratio", None)
        or (ref_ckpt.get("args") or {}).get("qsvd_target_param_ratio", 0.75)
    )
    args.use_qsvd = True
    args.qsvd_discard_original = True
    args.qsvd_rank_ratio = target_ratio
    args.gen_model = "CSformer_QSVD_Shared"

    torch.save(
        {
            "epoch": 0,
            "gen_state_dict": shared_model.state_dict(),
            "best_psnr": 0.0,
            "best_ssim": 0.0,
            "qsvd_discard_original": True,
            "qsvd_ranks": ranks,
            "qsvd_selected_indices": selected_lists,
            "qsvd_shared": True,
            "args": vars(args),
            "path_helper": {
                "prefix": args.save_dir,
                "ckpt_path": os.path.join(args.save_dir, "Model"),
                "log_path": os.path.join(args.save_dir, "Log"),
                "sample_path": os.path.join(args.save_dir, "Samples"),
            },
            "importance_method": "matched_topr",
            "ranks_from_ckpt": os.path.abspath(args.ranks_from_ckpt),
            "ref_qsvd_selected_indices": ref_selected,
            "qsvd_score_mode": "matched_topr",
            "shared_metrics": {k: v for k, v in metrics.items() if k != "ranks"},
        },
        save_path,
    )
    print(f"Saved {save_path}")
    print(f"Per-layer ranks (matched top-r): {ranks}")


if __name__ == "__main__":
    main()
