"""
Dump α/γ/η + sensing + optional latency for any CSFormer checkpoint.

Usage:
  python report_model_metrics.py \
      --load_path /path/to/checkpoint_best.pth \
      --save_json /path/to/efficiency.json \
      --run_latency
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import importlib
import os
from collections import OrderedDict

import torch

import cfg
from utils.report_efficiency import (
    build_efficiency_report,
    print_efficiency_report,
    save_efficiency_report,
)


def _set_n_input(args):
    ratio_dict = {1: 3, 4: 11, 10: 26, 25: 64, 30: 77, 40: 103, 50: 128}
    args.n_input = ratio_dict[args.cs_ratio]
    return args


def main():
    parser = argparse.ArgumentParser(description="Report CSFormer efficiency metrics")
    parser.add_argument("--load_path", type=str, required=True,
                        help="Checkpoint .pth OR directory containing Model/checkpoint_best.pth")
    parser.add_argument("--save_json", type=str, default="",
                        help="Optional path to write JSON report")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--run_latency", action="store_true", default=False)
    parser.add_argument("--latency_input", type=int, nargs=4, default=[1, 1, 256, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args, remaining = parser.parse_known_args()

    cfg_args = cfg.parse_args(remaining)
    for k, v in vars(cfg_args).items():
        if not hasattr(args, k):
            setattr(args, k, v)

    load_path = args.load_path
    if os.path.isdir(load_path):
        load_path = os.path.join(load_path, "Model", "checkpoint_best.pth")
    assert os.path.isfile(load_path), f"Missing checkpoint: {load_path}"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(load_path, map_location=device)

    if "args" in ckpt:
        saved = ckpt["args"]
        # saved may be Namespace or dict
        if hasattr(saved, "__dict__"):
            saved = vars(saved)
        for k in (
            "gen_model", "cs_ratio", "gf_dim", "g_depth", "num_heads",
            "g_window_size", "bottom_width", "img_size", "datarange",
            "use_qsvd", "qsvd_rank_ratio", "qsvd_sigma_fuse",
            "use_sib_acs", "sib_acs_phi_size", "sib_acs_num_stages",
            "sib_acs_base_ratio", "sib_acs_channels", "sib_acs_aux_blocks",
            "sib_acs_aux_layers", "sib_acs_headroom", "sib_acs_max_measurements",
        ):
            if k in saved:
                setattr(args, k, saved[k])

    args = _set_n_input(args)
    model_module = importlib.import_module("models." + args.gen_model)
    model_class = getattr(model_module, args.gen_model)
    model = model_class(args).to(device)

    # Re-apply ranks for QSVD / shared
    if "qsvd_ranks" in ckpt and getattr(args, "use_qsvd", False):
        ranks = ckpt["qsvd_ranks"]
        if args.gen_model == "CSformer_QSVD_Shared":
            from models.QSVD_Transformer64_shared import collect_shared_attention_modules
            atts = collect_shared_attention_modules(model)
            for attn, rank in zip(atts, ranks):
                attn.apply_shared_qsvd(rank_or_ratio=int(rank),
                                       sigma_fuse=getattr(args, "qsvd_sigma_fuse", "UV"))
        else:
            from models.QSVD_Transformer64 import collect_attention_modules
            atts = collect_attention_modules(model)
            for attn, rank in zip(atts, ranks):
                attn.apply_qsvd(rank_or_ratio=int(rank),
                                sigma_fuse=getattr(args, "qsvd_sigma_fuse", "UV"))

    state = ckpt.get("gen_state_dict") or ckpt.get("avg_gen_state_dict")
    if list(state.keys())[0].startswith("module."):
        state = OrderedDict((k[7:], v) for k, v in state.items())
    strict = not ckpt.get("qsvd_discard_original", False)
    model.load_state_dict(state, strict=strict)
    model.eval()

    report = build_efficiency_report(
        model,
        args=args,
        ckpt_path=load_path,
        run_latency=args.run_latency,
        latency_input=tuple(args.latency_input),
        tag=args.tag or os.path.basename(os.path.dirname(os.path.dirname(load_path))),
    )
    if args.run_latency and "runtime" in report and "error" not in report.get("runtime", {}):
        # overwrite with user warmup/iters if measure_latency used defaults — already in build
        pass
    print_efficiency_report(report)

    save_path = args.save_json
    if not save_path:
        save_path = os.path.join(os.path.dirname(load_path), "efficiency_report.json")
    save_efficiency_report(report, save_path)


if __name__ == "__main__":
    main()
