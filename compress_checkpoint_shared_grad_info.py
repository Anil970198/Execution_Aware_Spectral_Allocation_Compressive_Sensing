#compress_checkpoint_shared_grad_info.py
"""
Compress baseline → CSformer_QSVD_Shared with GRAD-INFO ranks + shared 1+3 graph.

NEW FILE. Budget uses paper α cost 4E per singular value.
"""

from __future__ import annotations

import argparse
import copy
import os
import random

import torch
import torch.nn as nn

import cfg
import models.CSformer as CSformer
import models.CSformer_QSVD_Shared as CSformer_QSVD_Shared
from models.QSVD_Transformer64_shared import (
    collect_shared_attention_modules,
    discard_original_qkv_in_shared_model,
)
from models.qsvd_grad_info_shared import run_shared_grad_info_pipeline
from models.qsvd_grad_info_knapsack_floor import (
    run_shared_grad_info_pipeline_knapsack_floor,
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


def _build_synthetic_calib(nsamples, batch_size, patch, seed):
    g = torch.Generator().manual_seed(int(seed))
    batches, remaining = [], int(nsamples)
    while remaining > 0:
        b = min(batch_size, remaining)
        batches.append(torch.rand(b, 1, patch, patch, generator=g) * 2.0 - 1.0)
        remaining -= b
    return batches


def _build_image_calib(args, nsamples, batch_size):
    from datasets import ImgData
    from torch.utils.data import DataLoader, Subset

    ds = ImgData(args, train=True)
    if len(ds) == 0:
        raise RuntimeError(f"No images under data_path={args.data_path}")
    n = min(int(nsamples), len(ds))
    indices = list(range(n))
    random.Random(int(getattr(args, "random_seed", 12345))).shuffle(indices)
    loader = DataLoader(
        Subset(ds, indices[:n]),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batches = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        batches.append(batch)
    if not batches:
        raise RuntimeError("Empty calib loader")
    print(f"    Loaded {len(batches)} real batches from {args.data_path}")
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--reference_input_size", type=int, nargs=4, default=[1, 1, 64, 64])
    parser.add_argument("--qsvd_target_param_ratio", type=float, default=0.75)
    parser.add_argument("--qsvd_grad_alpha", type=float, default=1.0)
    parser.add_argument("--qsvd_calib_nsamples", type=int, default=64)
    parser.add_argument("--qsvd_calib_batch_size", type=int, default=2)
    parser.add_argument("--allow_synthetic_calib", action="store_true")
    parser.add_argument("--rec_loss_type_calib", type=str, default=None)
    # Parallel path: knapsack density + per-layer minimum-rank floor.
    # Default OFF → pure knapsack (existing Job / path unchanged).
    parser.add_argument(
        "--use_knapsack_floor",
        action="store_true",
        help="Use knapsack + min-rank floor allocator (parallel to pure knapsack).",
    )
    parser.add_argument(
        "--min_rank_floor_ratio",
        type=float,
        default=0.15,
        help="Floor r_min = max(1, floor(ratio * E)). Only used with --use_knapsack_floor.",
    )
    parser.add_argument(
        "--qsvd_score_mode",
        type=str,
        default="ne",
        help="Importance density: ne (=σ|g|^β/(N·E), LOCKED default); "
             "classical_4e (=σ|g|^β/(4E)); sigma (=|σ|); "
             "sigma_g (=σ|g|^β); ne_l025|ne_l05|ne_l075 (=σ|g|^β/(NE)^λ); "
             "rn (=σ|g|^β/N); re (=σ|g|^β/E).",
    )
    args, remaining = parser.parse_known_args()
    cfg_parser = cfg.parse_args(remaining)
    for k, v in vars(cfg_parser).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args = _set_n_input(args)

    target_ratio = float(args.qsvd_target_param_ratio)
    sigma_fuse = getattr(args, "qsvd_sigma_fuse", "UV")
    rank_align = int(getattr(args, "qsvd_rank_align", 1))
    grad_alpha = float(args.qsvd_grad_alpha)
    nsamples = int(args.qsvd_calib_nsamples)
    batch_size = int(args.qsvd_calib_batch_size)
    seed = int(getattr(args, "random_seed", 12345))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join(args.save_dir, "Model", "checkpoint_best.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    use_floor = bool(getattr(args, "use_knapsack_floor", False))
    floor_ratio = float(getattr(args, "min_rank_floor_ratio", 0.15))
    score_mode = str(getattr(args, "qsvd_score_mode", "ne"))

    print("=" * 70)
    if use_floor:
        print(
            "SHARED QSVD Knapsack+Floor Compression "
            "(1 down + 3 ups, α=4rE, floor then density)"
        )
    else:
        print("SHARED QSVD Grad-Info Compression (1 down + 3 ups, α=4rE budget)")
    print("=" * 70)
    print(f"Baseline: {args.load_path}")
    print(f"Target α ratio: {target_ratio}")
    print(f"Score mode: {score_mode}")
    print(f"Allocator: {'knapsack+floor' if use_floor else 'pure knapsack'}")
    if use_floor:
        print(f"Min-rank floor ratio: {floor_ratio}")
    print(f"Device: {device}")

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

    patch = int(getattr(args, "train_patch_size", 128))
    if hasattr(args, "img_size"):
        # calib can use 64 min; ImgData uses train size
        pass
    if args.allow_synthetic_calib or not getattr(args, "data_path", None):
        if not args.allow_synthetic_calib and not getattr(args, "data_path", None):
            raise RuntimeError("Need --data_path or --allow_synthetic_calib")
        if args.allow_synthetic_calib:
            print("    Using SYNTHETIC calibration")
            calib = _build_synthetic_calib(nsamples, batch_size, 64, seed)
        else:
            calib = _build_image_calib(args, nsamples, batch_size)
    else:
        calib = _build_image_calib(args, nsamples, batch_size)

    loss_type = args.rec_loss_type_calib or getattr(args, "rec_loss_type", "l2")
    loss_fn = nn.L1Loss() if str(loss_type).lower() in ("l1", "mae") else nn.MSELoss()
    print(f"    Calib loss: {loss_fn.__class__.__name__}")

    if use_floor:
        ranks, selected, _imp = run_shared_grad_info_pipeline_knapsack_floor(
            shared_model,
            calibration_batches=calib,
            loss_fn=loss_fn,
            target_alpha_ratio=target_ratio,
            device=device,
            sigma_fuse=sigma_fuse,
            grad_alpha=grad_alpha,
            rank_align=rank_align,
            act_aware=bool(getattr(args, "qsvd_act_aware", False)),
            min_rank_floor_ratio=floor_ratio,
            score_mode=score_mode,
        )
        importance_method = "qsvd_grad_info_knapsack_floor"
    else:
        ranks, selected, _imp = run_shared_grad_info_pipeline(
            shared_model,
            calibration_batches=calib,
            loss_fn=loss_fn,
            target_alpha_ratio=target_ratio,
            device=device,
            sigma_fuse=sigma_fuse,
            grad_alpha=grad_alpha,
            rank_align=rank_align,
            act_aware=bool(getattr(args, "qsvd_act_aware", False)),
            score_mode=score_mode,
        )
        importance_method = "qsvd_grad_info_shared"

    attentions = collect_shared_attention_modules(shared_model)
    selected_lists = [
        selected[a].cpu().tolist() if a in selected else [0] for a in attentions
    ]

    discard_original_qkv_in_shared_model(shared_model)
    shared_model = shared_model.cpu()
    metrics = print_shared_qsvd_metrics(shared_model, tuple(args.reference_input_size))

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
            "importance_method": importance_method,
            "qsvd_grad_alpha": grad_alpha,
            "qsvd_score_mode": score_mode,
            "use_knapsack_floor": use_floor,
            "min_rank_floor_ratio": floor_ratio if use_floor else None,
            "shared_metrics": {k: v for k, v in metrics.items() if k != "ranks"},
        },
        save_path,
    )
    print(f"Saved {save_path}")
    print(f"Per-layer ranks: {ranks}")


if __name__ == "__main__":
    main()
