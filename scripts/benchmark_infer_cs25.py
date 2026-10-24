#benchmark_infer_cs25.py

"""
Pure-forward inference benchmark + optional Set11 overlap eval.

Appends one CSV row per model (baseline / Way-1 / Way-2).

Example:
  python scripts/benchmark_infer_cs25.py \\
    --model_tag baseline --way baseline \\
    --load_path logs/checkpoint_coco/checkpoint_CS25.pth \\
    --gen_model CSformer --cs_ratio 25 \\
    --out_csv /scratch/.../summary.csv \\
    --testdata_path data/Set11 --run_set11
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cfg
from functions import test_overlap
from utils.report_efficiency import (
    build_efficiency_report,
    compute_paper_agy,
    count_params,
    measure_latency,
    checkpoint_size_mb,
)
from utils.utils import create_logger, set_log_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_ARCH_KEYS = [
    "gen_model", "cs_ratio", "gf_dim", "g_depth", "num_heads",
    "g_window_size", "bottom_width", "img_size", "datarange",
    "use_qsvd", "qsvd_rank_ratio", "qsvd_sigma_fuse",
    "qsvd_act_aware", "qsvd_rank_align",
    "qsvd_adaptive_rank", "qsvd_target_param_ratio",
    "qsvd_target_alpha_ratio",
    "qsvd_adaptive_rank_tau", "qsvd_adaptive_rank_simple",
    "use_factored_qk",
    "use_sib_acs", "sib_acs_phi_size", "sib_acs_num_stages",
    "sib_acs_base_ratio", "sib_acs_channels", "sib_acs_aux_blocks",
    "sib_acs_aux_layers", "sib_acs_headroom", "sib_acs_max_measurements",
]


def _set_n_input(args):
    ratio_dict = {1: 3, 4: 11, 10: 26, 25: 64, 30: 77, 40: 103, 50: 128}
    args.n_input = ratio_dict[args.cs_ratio]
    return args


def _unwrap(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def load_checkpoint_model(
    load_path: str,
    gen_model: str,
    cs_ratio: int,
    use_qsvd: bool = False,
    use_factored_qk: bool = False,
    qsvd_sigma_fuse: str = "UV",
    expect_alpha: Optional[float] = None,
    alpha_tol: float = 0.02,
    ffn_prune_indices: str = "",
    use_tome: bool = False,
    tome_r: int = 4,
    tome_prop_attn: bool = True,
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """
    Load baseline or shared-QKV checkpoint.

    For shared models we NEVER trust legacy --qsvd_rank_ratio alone
    (0.75 there means paper alpha=0.50). We require checkpoint['qsvd_ranks']
    and install those integer ranks. Optional expect_alpha asserts paper alpha.
    """
    assert os.path.isfile(load_path), f"missing checkpoint: {load_path}"
    checkpoint = torch.load(load_path, map_location=device)

    args = cfg.parse_args([])
    args.gen_model = gen_model
    args.cs_ratio = cs_ratio
    args.use_qsvd = use_qsvd
    args.use_factored_qk = use_factored_qk
    args.qsvd_sigma_fuse = qsvd_sigma_fuse
    args.gf_dim = 128
    args.g_depth = "5,5,5,5"
    args.num_heads = "16,8,4,2"
    args.g_window_size = 8
    args.bottom_width = 8
    args.img_size = 64
    args.datarange = "-11"

    if "args" in checkpoint:
        saved = checkpoint["args"]
        for k in _ARCH_KEYS:
            if k in saved:
                setattr(args, k, saved[k])
    if use_factored_qk:
        args.use_factored_qk = True

    # Build WITHOUT applying legacy rank_ratio in __init__.
    # Shared ranks come only from checkpoint['qsvd_ranks'] (integers).
    want_shared = (
        args.gen_model == "CSformer_QSVD_Shared" and (use_qsvd or getattr(args, "use_qsvd", False))
    )
    if want_shared:
        args.use_qsvd = False

    args = _set_n_input(args)
    model_module = importlib.import_module("models." + args.gen_model)
    model_class = getattr(model_module, args.gen_model)
    gen_net = model_class(args).to(device)

    # Structured FFN prune reshape before load
    from utils.ffn_prune import apply_ffn_prune_from_indices, load_indices_file
    if ffn_prune_indices:
        if not os.path.isfile(ffn_prune_indices):
            raise FileNotFoundError(f"ffn_prune_indices missing: {ffn_prune_indices}")
        apply_ffn_prune_from_indices(gen_net, load_indices_file(ffn_prune_indices))
        print(f"    Applied FFN prune indices from {ffn_prune_indices}")
    elif checkpoint.get("ffn_prune_indices"):
        apply_ffn_prune_from_indices(
            gen_net,
            {
                k: v.long() if torch.is_tensor(v) else torch.tensor(v).long()
                for k, v in checkpoint["ffn_prune_indices"].items()
            },
        )
        print("    Applied FFN prune indices from checkpoint")
    elif checkpoint.get("ffn_prune_method"):
        raise RuntimeError(
            "Pruned FFN checkpoint requires ffn_prune_indices path or embedded indices"
        )

    if want_shared:
        if "qsvd_ranks" not in checkpoint:
            raise RuntimeError(
                f"Shared checkpoint missing qsvd_ranks: {load_path}. "
                "Refusing to use legacy qsvd_rank_ratio (0.75 -> alpha 0.50)."
            )
        from models.QSVD_Transformer64_shared import collect_shared_attention_modules
        ranks = [int(r) for r in checkpoint["qsvd_ranks"]]
        attentions = collect_shared_attention_modules(gen_net)
        if len(ranks) != len(attentions):
            raise RuntimeError(
                f"Rank count {len(ranks)} != attention count {len(attentions)}"
            )
        print(f"    Installing SHARED QKV from saved integer ranks only: {ranks}")
        print(
            "    (not using legacy qsvd_rank_ratio; "
            f"ckpt rank_ratio={getattr(args, 'qsvd_rank_ratio', None)}, "
            f"target_alpha={checkpoint.get('qsvd_target_alpha_ratio', getattr(args, 'qsvd_target_alpha_ratio', None))})"
        )
        for attn, rank in zip(attentions, ranks):
            attn.apply_shared_qsvd(
                rank_or_ratio=int(rank), sigma_fuse=args.qsvd_sigma_fuse
            )
        args.use_qsvd = True

    state_dict = checkpoint.get("gen_state_dict") or checkpoint.get("avg_gen_state_dict")
    if state_dict is None:
        raise KeyError(f"No gen_state_dict in {load_path}")
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.endswith("total_ops") and not k.endswith("total_params")
    }

    if list(state_dict.keys())[0].startswith("module."):
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[k[7:]] = v
        state_dict = new_sd

    sd_keys = list(state_dict.keys())
    has_shared = any("shared_qkv." in k for k in sd_keys)
    has_fused_qkv = any("attn.qkv.weight" in k for k in sd_keys)
    discarded_qkv = bool(
        checkpoint.get("qsvd_discard_original", False)
        or (has_shared and not has_fused_qkv)
    )
    gen_net.load_state_dict(state_dict, strict=not discarded_qkv)
    if discarded_qkv:
        from models.QSVD_Transformer64_shared import discard_original_qkv_in_shared_model
        discard_original_qkv_in_shared_model(gen_net)

    if args.gen_model == "CSformer_QSVD_Shared" and args.use_factored_qk:
        from models.QSVD_Transformer64_shared import set_use_factored_qk_on_model
        n = set_use_factored_qk_on_model(gen_net, True)
        print(f"    Way-2 factored QK enabled on {n} layers")

    if want_shared and expect_alpha is not None:
        agy = compute_paper_agy(gen_net, input_size=(1, 1, 64, 64))
        got = float(agy["alpha_ratio"])
        print(f"    Verified paper alpha_ratio={got:.4f} (expect {expect_alpha:.4f})")
        print(f"    ranks={agy.get('ranks')}")
        if abs(got - float(expect_alpha)) > float(alpha_tol):
            raise RuntimeError(
                f"Paper alpha_ratio {got:.6f} != expect {expect_alpha:.6f} "
                f"(tol={alpha_tol}). Wrong checkpoint or legacy rank path."
            )
        # Honest alpha=0.75 uniform should be near {144,72,36,18} per stage.
        if abs(float(expect_alpha) - 0.75) < 1e-6:
            stage0 = set(agy["ranks"][:5])
            if stage0 == {96}:
                raise RuntimeError(
                    "Ranks look like LEGACY uni0.75 (alpha 0.50, stage0=96). "
                    "Use CS25_uniform_alpha0.75 instead."
                )

    if use_tome:
        from utils.tome_csformer import enable_tome_on_shared_model
        info = enable_tome_on_shared_model(
            gen_net, r=int(tome_r), prop_attn=bool(tome_prop_attn), enabled=True
        )
        print(
            f"    ToMe ON  r={info['r']}  prop_attn={info['prop_attn']}  "
            f"stages={info.get('n_stage')} blocks={info.get('n_block')}"
        )

    gen_net.eval()
    return gen_net, args, checkpoint


def _peak_mem_mb(model: nn.Module, input_size=(1, 1, 256, 256)) -> Optional[float]:
    core = _unwrap(model)
    if not torch.cuda.is_available():
        return None
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(*input_size, device=device)
    with torch.no_grad():
        _ = core(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def _run_set11(model: nn.Module, args, testdata_path: str) -> Tuple[float, float]:
    args.testdata_path = testdata_path
    args.valdata_path = testdata_path
    args.overlap = True
    args.overlapstep = 8
    args.exp_name = "benchmark_set11_tmp"
    args.path_helper = set_log_dir("logs_eval", args.exp_name)
    logger = create_logger(args.path_helper["log_path"], phase="test")
    logger.setLevel(logging.INFO)

    class _Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.psnr = None
            self.ssim = None

        def emit(self, record):
            msg = record.getMessage()
            if "[all cross]" in msg and "Mean PSNR" in msg:
                import re
                m = re.search(r"Mean PSNR loss: ([0-9.]+).*Mean SSIM loss: ([0-9.]+)", msg)
                if m:
                    self.psnr = float(m.group(1))
                    self.ssim = float(m.group(2))

    cap = _Capture()
    logger.addHandler(cap)
    test_overlap(args, model, logger)
    if cap.psnr is None:
        raise RuntimeError("Set11 eval did not produce PSNR/SSIM in logs")
    return cap.psnr, cap.ssim


def append_csv_row(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.isfile(path)
    fieldnames = list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser(description="CS25 inference benchmark (baseline / Way-1 / Way-2)")
    p.add_argument("--model_tag", required=True, help="baseline | way1 | way2")
    p.add_argument("--way", required=True, choices=["baseline", "way1", "way2"])
    p.add_argument("--load_path", required=True)
    p.add_argument("--gen_model", default="CSformer")
    p.add_argument("--cs_ratio", type=int, default=25)
    p.add_argument("--alpha_budget", type=float, default=-1.0)
    p.add_argument("--use_qsvd", action="store_true")
    p.add_argument("--use_factored_qk", action="store_true")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--testdata_path", default="")
    p.add_argument("--run_set11", action="store_true")
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument(
        "--expect_alpha",
        type=float,
        default=-1.0,
        help="If >0, assert paper QKV alpha_ratio matches (blocks legacy 0.75->0.50 mistake).",
    )
    p.add_argument("--alpha_tol", type=float, default=0.02)
    p.add_argument(
        "--fuse_up",
        action="store_true",
        help="Infer-time: stack three ups into one Linear(r->3E) (2 GEMMs). Same weights.",
    )
    p.add_argument(
        "--batches",
        type=str,
        default="1",
        help="Comma-separated batch sizes for latency (e.g. 1 or 1,8).",
    )
    p.add_argument(
        "--ffn_prune_indices",
        type=str,
        default="",
        help="prune_indices.pt for structurally FFN-pruned dense CSformer.",
    )
    p.add_argument("--use_tome", action="store_true", help="Enable ToMe at inference")
    p.add_argument("--tome-r", "--tome_r", type=int, default=4, dest="tome_r")
    p.add_argument("--tome-prop-attn", "--tome_prop_attn", type=int, default=1, dest="tome_prop_attn")
    cli = p.parse_args()

    batches = [int(x.strip()) for x in cli.batches.split(",") if x.strip()]
    if not batches:
        batches = [1]

    use_factored = cli.way == "way2" or cli.use_factored_qk
    use_qsvd = cli.way in ("way1", "way2") or cli.use_qsvd
    if cli.way == "baseline":
        cli.gen_model = "CSformer"
        use_qsvd = False
        use_factored = False
    elif cli.way in ("way1", "way2"):
        cli.gen_model = "CSformer_QSVD_Shared"
        use_qsvd = True

    print("=" * 64)
    print(f"BENCHMARK  tag={cli.model_tag}  way={cli.way}  ckpt={cli.load_path}")
    print(f"fuse_up={cli.fuse_up}  batches={batches}  use_tome={cli.use_tome} r={cli.tome_r}")
    print("=" * 64)

    expect_alpha = cli.expect_alpha if cli.expect_alpha > 0 else None
    model, args, _ckpt = load_checkpoint_model(
        cli.load_path,
        gen_model=cli.gen_model,
        cs_ratio=cli.cs_ratio,
        use_qsvd=use_qsvd,
        use_factored_qk=use_factored,
        expect_alpha=expect_alpha,
        alpha_tol=cli.alpha_tol,
        ffn_prune_indices=cli.ffn_prune_indices,
        use_tome=cli.use_tome,
        tome_r=cli.tome_r,
        tome_prop_attn=bool(cli.tome_prop_attn),
    )

    if cli.fuse_up:
        if cli.way == "baseline":
            raise SystemExit("--fuse_up is only for shared-QKV models")
        from models.qsvd_shared_qkv import set_fused_up_on_model
        n_fuse = set_fused_up_on_model(model, True)
        print(f"Enabled fused-up on {n_fuse} SharedQKV modules")

    agy = compute_paper_agy(model, input_size=(1, 1, 64, 64))
    params = count_params(model)
    ckpt_mb = checkpoint_size_mb(cli.load_path)

    # Latency matrix: resolution x batch
    lat_cells = {}
    for bs in batches:
        for h in (64, 256):
            key = f"b{bs}_{h}"
            lat_cells[key] = measure_latency(
                model,
                input_size=(bs, 1, h, h),
                warmup=cli.warmup,
                iterations=cli.iters,
                device=device,
            )

    peak_mb = _peak_mem_mb(model, (1, 1, 256, 256))
    peak_mb_b8 = None
    if 8 in batches:
        peak_mb_b8 = _peak_mem_mb(model, (8, 1, 256, 256))

    psnr, ssim = float("nan"), float("nan")
    if cli.run_set11:
        if not cli.testdata_path or not os.path.isdir(cli.testdata_path):
            raise FileNotFoundError(f"Set11 path missing: {cli.testdata_path}")
        psnr, ssim = _run_set11(model, args, cli.testdata_path)
        print(f"Set11: PSNR={psnr:.2f}  SSIM={ssim:.4f}")

    alpha_budget = cli.alpha_budget
    ckpt_args = _ckpt.get("args", {}) if isinstance(_ckpt.get("args", {}), dict) else {}
    if alpha_budget < 0:
        if _ckpt.get("qsvd_target_alpha_ratio") is not None:
            alpha_budget = float(_ckpt["qsvd_target_alpha_ratio"])
        elif "qsvd_target_alpha_ratio" in ckpt_args and ckpt_args["qsvd_target_alpha_ratio"] is not None:
            alpha_budget = float(ckpt_args["qsvd_target_alpha_ratio"])
        elif "qsvd_target_param_ratio" in ckpt_args:
            alpha_budget = float(ckpt_args["qsvd_target_param_ratio"])
        elif expect_alpha is not None:
            alpha_budget = float(expect_alpha)

    # Keep legacy column names for batch-1 (paper table compatibility)
    lat64 = lat_cells.get("b1_64") or lat_cells[f"b{batches[0]}_64"]
    lat256 = lat_cells.get("b1_256") or lat_cells[f"b{batches[0]}_256"]

    row = {
        "model": cli.model_tag,
        "way": cli.way,
        "fuse_up": int(bool(cli.fuse_up)),
        "cs_ratio": cli.cs_ratio,
        "alpha_budget": alpha_budget if alpha_budget >= 0 else "",
        "psnr_set11": f"{psnr:.4f}" if psnr == psnr else "",
        "ssim_set11": f"{ssim:.4f}" if ssim == ssim else "",
        "params_M": f"{params['total_params_M']:.4f}",
        "ckpt_MB": f"{ckpt_mb:.2f}" if ckpt_mb is not None else "",
        "alpha_ratio": f"{agy['alpha_ratio']:.4f}",
        "gamma_ratio": f"{agy['gamma_ratio']:.4f}",
        "eta_ratio": f"{agy['eta_ratio']:.4f}",
        "lat_ms_64": f"{lat64['latency_ms_per_image']:.3f}",
        "lat_std_64": f"{lat64['latency_std_ms']:.3f}",
        "thr_64": f"{lat64['throughput_img_s']:.2f}",
        "lat_ms_256": f"{lat256['latency_ms_per_image']:.3f}",
        "lat_std_256": f"{lat256['latency_std_ms']:.3f}",
        "thr_256": f"{lat256['throughput_img_s']:.2f}",
        "peak_mem_MB": f"{peak_mb:.1f}" if peak_mb is not None else "",
        "peak_mem_MB_b8": f"{peak_mb_b8:.1f}" if peak_mb_b8 is not None else "",
        "ckpt_path": cli.load_path,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    # Optional batch-8 columns
    if "b8_64" in lat_cells:
        row["lat_ms_64_b8"] = f"{lat_cells['b8_64']['latency_ms_per_image']:.3f}"
        row["thr_64_b8"] = f"{lat_cells['b8_64']['throughput_img_s']:.2f}"
    if "b8_256" in lat_cells:
        row["lat_ms_256_b8"] = f"{lat_cells['b8_256']['latency_ms_per_image']:.3f}"
        row["thr_256_b8"] = f"{lat_cells['b8_256']['throughput_img_s']:.2f}"

    append_csv_row(cli.out_csv, row)
    print(f"Wrote row to {cli.out_csv}")
    print(
        f"Latency 64: {row['lat_ms_64']} ms  |  256: {row['lat_ms_256']} ms  |  "
        f"gamma_ratio={row['gamma_ratio']}  fuse_up={cli.fuse_up}"
    )


if __name__ == "__main__":
    main()
