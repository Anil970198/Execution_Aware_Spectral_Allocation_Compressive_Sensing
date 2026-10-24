#prune_ffn_structured.py
#!/usr/bin/env python3
"""
Structured FFN prune for dense CSformer OR shared knapsack CSformer_QSVD_Shared.

Dense:
  python scripts/prune_ffn_structured.py --checkpoint dense.pth --method magnitude ...

Product (knapsack @ alpha 0.75):
  python scripts/prune_ffn_structured.py \\
    --checkpoint .../CS25_knapsack_pure_a0.75/Model/checkpoint_best.pth \\
    --gen_model CSformer_QSVD_Shared --method magnitude ...
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cfg
from utils.ffn_prune import (
    analytic_ffn_macs,
    assert_twenty_mlps,
    compute_magnitude_indices,
    compute_random_indices,
    compute_taylor_indices,
    count_params,
    indices_are_trivial_half,
    prune_mlps_with_indices,
    strip_module_prefix,
    strip_thop_keys,
    try_thop_gflops,
    unwrap_model,
    verify_prune_shapes,
)


def _set_n_input(args):
    ratio_dict = {1: 3, 4: 11, 10: 26, 25: 64, 30: 77, 40: 103, 50: 128}
    args.n_input = ratio_dict[args.cs_ratio]
    return args


def _arch_args(cs_ratio: int = 25):
    args = cfg.parse_args([])
    args.cs_ratio = cs_ratio
    args.gf_dim = 128
    args.g_depth = "5,5,5,5"
    args.num_heads = "16,8,4,2"
    args.g_window_size = 8
    args.bottom_width = 8
    args.img_size = 64
    args.datarange = "-11"
    args.g_mlp = 4
    args.qsvd_sigma_fuse = "UV"
    return _set_n_input(args)


def build_dense_csformer(cs_ratio: int = 25):
    args = _arch_args(cs_ratio)
    args.gen_model = "CSformer"
    args.use_qsvd = False
    mod = importlib.import_module("models.CSformer")
    return mod.CSformer(args), args


def build_shared_csformer(cs_ratio: int = 25, use_factored_qk: bool = False):
    """Build shared model WITHOUT applying ranks in __init__ (ranks from ckpt)."""
    args = _arch_args(cs_ratio)
    args.gen_model = "CSformer_QSVD_Shared"
    args.use_qsvd = False  # install ranks after build from qsvd_ranks
    args.use_factored_qk = use_factored_qk
    mod = importlib.import_module("models.CSformer_QSVD_Shared")
    return mod.CSformer_QSVD_Shared(args), args


def install_shared_ranks(model: nn.Module, ranks: List[int], sigma_fuse: str = "UV") -> None:
    from models.QSVD_Transformer64_shared import collect_shared_attention_modules

    root = unwrap_model(model)
    attentions = collect_shared_attention_modules(root)
    if len(ranks) != len(attentions):
        raise RuntimeError(f"ranks ({len(ranks)}) != attentions ({len(attentions)})")
    for attn, rank in zip(attentions, ranks):
        attn.apply_shared_qsvd(rank_or_ratio=int(rank), sigma_fuse=sigma_fuse)
    print(f"Installed SHARED QSVD ranks ({len(ranks)}): {ranks}")


def shared_qkv_shapes_snapshot(model: nn.Module) -> List[Tuple[int, int]]:
    """List of (r, E) for each SharedQKV.down weight [r, E]."""
    from models.QSVD_Transformer64_shared import collect_shared_attention_modules

    out = []
    for attn in collect_shared_attention_modules(unwrap_model(model)):
        if attn.shared_qkv is None:
            raise RuntimeError("shared_qkv missing — not a shared install")
        # down: Linear(E, r) → weight [r, E]
        w = attn.shared_qkv.down.weight
        out.append((int(w.shape[0]), int(w.shape[1])))
    return out


def load_dense_checkpoint(model: nn.Module, path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit(f"checkpoint is not a dict: {path}")
    sd = ckpt.get("gen_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
    if sd is None and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
        ckpt = {"gen_state_dict": sd}
    if sd is None:
        raise SystemExit(f"No gen_state_dict in {path}")
    sd = strip_module_prefix(sd)
    model.load_state_dict(sd, strict=True)
    print(f"Loaded dense ckpt: {path}")
    return ckpt


def load_shared_knapsack_checkpoint(
    path: str,
    cs_ratio: int = 25,
    use_factored_qk: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Any, Dict[str, Any], List[int]]:
    """
    Build CSformer_QSVD_Shared, install ckpt qsvd_ranks, load weights.
    Fails if qsvd_ranks missing (knapsack product requires them).
    """
    from models.QSVD_Transformer64_shared import discard_original_qkv_in_shared_model

    ckpt = torch.load(path, map_location="cpu")
    if "qsvd_ranks" not in ckpt:
        raise SystemExit(
            f"Shared knapsack checkpoint missing qsvd_ranks: {path}. "
            "Refuse to use legacy rank_ratio alone."
        )
    ranks = [int(r) for r in ckpt["qsvd_ranks"]]
    model, args = build_shared_csformer(cs_ratio, use_factored_qk=use_factored_qk)
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        for k in (
            "gf_dim", "g_depth", "num_heads", "g_window_size", "bottom_width",
            "img_size", "datarange", "qsvd_sigma_fuse", "cs_ratio",
        ):
            if k in ckpt["args"]:
                setattr(args, k, ckpt["args"][k])
        args = _set_n_input(args)

    install_shared_ranks(model, ranks, sigma_fuse=getattr(args, "qsvd_sigma_fuse", "UV"))
    args.use_qsvd = True
    args.gen_model = "CSformer_QSVD_Shared"

    sd = ckpt.get("gen_state_dict") or ckpt.get("avg_gen_state_dict")
    if sd is None:
        raise SystemExit(f"No gen_state_dict in {path}")
    sd = strip_module_prefix(sd)
    has_shared = any("shared_qkv." in k for k in sd.keys())
    has_fused = any("attn.qkv.weight" in k for k in sd.keys())
    discarded = bool(ckpt.get("qsvd_discard_original", False) or (has_shared and not has_fused))
    missing, unexpected = model.load_state_dict(sd, strict=not discarded)
    if discarded:
        discard_original_qkv_in_shared_model(model)
        print(f"Loaded shared knapsack (strict=False); missing sample: {list(missing)[:3]}")
    else:
        print(f"Loaded shared knapsack strict=True")
    print(f"  path={path}")
    print(f"  ranks={ranks}")
    if device is not None:
        model = model.to(device)
    return model, args, ckpt, ranks


def make_calib_loader(args, data_path: str, batch_size: int, num_workers: int):
    import datasets as ds

    args.data_path = data_path
    args.dataset = "coco"
    args.gen_batch_size = batch_size
    args.num_workers = num_workers
    return ds.ImageDataset(args).train


def synthetic_calib_loader(batch_size: int, n_batches: int, device):
    for _ in range(n_batches):
        yield torch.randn(batch_size, 1, 64, 64, device=device)


def main():
    p = argparse.ArgumentParser(description="Structured FFN prune (dense or shared knapsack)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--method", choices=["magnitude", "taylor", "random"], required=True)
    p.add_argument("--keep_ratio", type=float, default=0.5)
    p.add_argument(
        "--random_seed",
        type=int,
        default=12345,
        help="Seed for --method random (per-MLP independent draws).",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cs_ratio", type=int, default=25)
    p.add_argument(
        "--gen_model",
        default="CSformer",
        choices=["CSformer", "CSformer_QSVD_Shared"],
        help="CSformer=dense ablation; CSformer_QSVD_Shared=knapsack product",
    )
    p.add_argument("--calib_batches", type=int, default=64)
    p.add_argument("--calib_batch_size", type=int, default=4)
    p.add_argument("--data_path", default="")
    p.add_argument("--synthetic_calib", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--use_factored_qk", action="store_true")
    cli = p.parse_args()

    os.makedirs(cli.out_dir, exist_ok=True)
    device = torch.device(cli.device)
    ranks: Optional[List[int]] = None
    qkv_snap_before = None

    if cli.gen_model == "CSformer":
        model, args = build_dense_csformer(cli.cs_ratio)
        ckpt_meta = load_dense_checkpoint(model, cli.checkpoint)
        model = model.to(device)
    else:
        model, args, ckpt_meta, ranks = load_shared_knapsack_checkpoint(
            cli.checkpoint,
            cs_ratio=cli.cs_ratio,
            use_factored_qk=cli.use_factored_qk,
            device=device,
        )
        qkv_snap_before = shared_qkv_shapes_snapshot(model)
        print(f"SharedQKV (r,E) sample stage0: {qkv_snap_before[0]}")

    params_before = count_params(model)
    analytic_before = analytic_ffn_macs(model, "before")
    thop_before = try_thop_gflops(model, device=device)

    mlps = assert_twenty_mlps(model)
    print(f"Discovered {len(mlps)} MLPs")
    print(f"Params before FFN prune: {params_before} ({params_before/1e6:.4f} M)")

    if cli.method == "magnitude":
        indices = compute_magnitude_indices(model, keep_ratio=cli.keep_ratio)
    elif cli.method == "random":
        indices = compute_random_indices(
            model, keep_ratio=cli.keep_ratio, seed=int(cli.random_seed)
        )
        print(f"Random FFN prune seed={cli.random_seed}")
    else:
        if cli.synthetic_calib or not cli.data_path:
            print("Taylor calib: SYNTHETIC")
            loader = synthetic_calib_loader(
                cli.calib_batch_size, cli.calib_batches, device
            )
        else:
            if not os.path.isdir(cli.data_path):
                raise SystemExit(f"COCO path missing: {cli.data_path}")
            print(f"Taylor calib: COCO {cli.data_path}, batches={cli.calib_batches}")
            loader = make_calib_loader(
                args, cli.data_path, cli.calib_batch_size, cli.num_workers
            )
        indices = compute_taylor_indices(
            model,
            loader,
            device=device,
            keep_ratio=cli.keep_ratio,
            max_batches=cli.calib_batches,
            datarange=args.datarange,
        )

    trivial = indices_are_trivial_half(indices)
    if trivial:
        print(f"WARNING: {len(trivial)} MLPs kept first-half set (may be coincidence)")
    else:
        print("OK: no MLP uses trivial first-half index set")

    prune_mlps_with_indices(model, indices)
    verify_prune_shapes(model, keep_ratio=cli.keep_ratio)

    if qkv_snap_before is not None:
        qkv_snap_after = shared_qkv_shapes_snapshot(model)
        if qkv_snap_before != qkv_snap_after:
            raise RuntimeError(
                f"FFN prune changed SharedQKV shapes!\n before={qkv_snap_before}\n after={qkv_snap_after}"
            )
        print("OK: SharedQKV shapes unchanged after FFN prune")

    params_after = count_params(model)
    analytic_after = analytic_ffn_macs(model, "after")
    thop_after = try_thop_gflops(model, device=device)

    print(f"Params after:  {params_after} ({params_after/1e6:.4f} M)")
    print(
        f"Saved: {params_before - params_after} "
        f"({100 * (params_before - params_after) / params_before:.2f}%)"
    )
    print(
        f"Analytic FFN GMACs: {analytic_before['ffn_gmacs']:.4f} -> "
        f"{analytic_after['ffn_gmacs']:.4f}"
    )
    if thop_before is not None and thop_after is not None:
        print(f"thop GFLOPs: {thop_before:.4f} -> {thop_after:.4f}")
    else:
        print("thop GFLOPs: UNAVAILABLE")

    idx_path = os.path.join(cli.out_dir, "prune_indices.pt")
    torch.save(indices, idx_path)
    print(f"Wrote {idx_path}")

    model_cpu = model.cpu()
    # Never persist thop total_ops/total_params — train.py strict load rejects them.
    gen_sd = strip_thop_keys(model_cpu.state_dict())
    out_ckpt: Dict[str, Any] = {
        "epoch": 0,
        "gen_state_dict": gen_sd,
        "ffn_prune_method": cli.method,
        "ffn_prune_keep_ratio": cli.keep_ratio,
        "ffn_prune_indices": indices,
        "params_before": params_before,
        "params_after": params_after,
        "analytic_ffn_gmacs_before": analytic_before["ffn_gmacs"],
        "analytic_ffn_gmacs_after": analytic_after["ffn_gmacs"],
        "thop_gflops_before": thop_before,
        "thop_gflops_after": thop_after,
        "qkv_base_checkpoint": cli.checkpoint,
        "args": dict(vars(args)) if hasattr(args, "__dict__") else {},
    }
    if "args" in ckpt_meta and isinstance(ckpt_meta["args"], dict):
        out_ckpt["args"] = dict(ckpt_meta["args"])
    out_ckpt["args"]["gen_model"] = cli.gen_model
    if cli.gen_model == "CSformer_QSVD_Shared":
        out_ckpt["args"]["use_qsvd"] = True
        out_ckpt["qsvd_ranks"] = ranks
        out_ckpt["qsvd_discard_original"] = True
        out_ckpt["qsvd_target_param_ratio"] = ckpt_meta.get(
            "qsvd_target_param_ratio",
            ckpt_meta.get("args", {}).get("qsvd_target_param_ratio", 0.75)
            if isinstance(ckpt_meta.get("args"), dict)
            else 0.75,
        )
        # knapsack product: paper alpha budget 0.75
        out_ckpt["qsvd_target_alpha_ratio"] = ckpt_meta.get(
            "qsvd_target_alpha_ratio", 0.75
        )
    else:
        out_ckpt["args"]["use_qsvd"] = False

    # preserve other useful meta
    for k in ("qsvd_ranks", "qsvd_discard_original", "qsvd_target_param_ratio",
              "qsvd_target_alpha_ratio"):
        if k in ckpt_meta and k not in out_ckpt:
            out_ckpt[k] = ckpt_meta[k]

    model_dir = os.path.join(cli.out_dir, "Model")
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "checkpoint_best.pth")
    torch.save(out_ckpt, ckpt_path)
    flat = os.path.join(cli.out_dir, "checkpoint_pruned.pth")
    torch.save(out_ckpt, flat)
    print(f"Wrote {ckpt_path}")
    print(f"Wrote {flat}")
    print("DONE prune", cli.method, cli.gen_model)


if __name__ == "__main__":
    main()
