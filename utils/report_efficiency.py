#report_efficiency.py
"""
Unified efficiency report for publication-style comparisons.

Reports:
  - paper α / γ / η for QKV (shared, triple-B, or full)
  - total params / checkpoint size
  - sensing budget (cs_ratio, N_avg, bank M) + optional ASA stats
  - optional latency / throughput

Use from train/eval/job scripts so every run dumps the same fields.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


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


def _collect_attention_ranks(model: nn.Module):
    """
    Return list of (dim, rank, graph) where graph in
    {'shared', 'triple', 'full'}.
    """
    core = _unwrap(model)
    out = []
    try:
        from models.QSVD_Transformer64_shared import SharedAttention
        for m in core.modules():
            if isinstance(m, SharedAttention):
                E = int(m.dim)
                if getattr(m, "use_qsvd", False) and getattr(m, "_qsvd_applied", False):
                    r = int(getattr(m, "qsvd_rank", E) or E)
                    out.append((E, r, "shared"))
                else:
                    out.append((E, E, "full"))
        if out:
            return out
    except Exception:
        pass

    try:
        from models.QSVD_Transformer64 import Attention as QSVDAttention
        for m in core.modules():
            if isinstance(m, QSVDAttention):
                E = int(m.dim)
                if getattr(m, "use_qsvd", False) and getattr(m, "_qsvd_applied", False):
                    r = int(getattr(m, "qsvd_rank", E) or E)
                    out.append((E, r, "triple"))
                else:
                    out.append((E, E, "full"))
        if out:
            return out
    except Exception:
        pass

    # Plain CSFormer baseline (Transformer64.Attention) — always full QKV.
    # Prefer isinstance; also match by class/module name so reloads / import
    # path quirks on clusters cannot silently zero α/γ/η.
    try:
        from models.Transformer64 import Attention as FullAttention
    except Exception:
        FullAttention = None  # type: ignore

    for m in core.modules():
        is_full = FullAttention is not None and isinstance(m, FullAttention)
        if not is_full:
            cls = type(m).__name__
            mod = getattr(type(m), "__module__", "") or ""
            is_full = (
                cls == "Attention"
                and "Transformer64" in mod
                and "QSVD" not in mod
                and hasattr(m, "qkv")
            )
        if not is_full:
            continue
        E = 0
        if hasattr(m, "qkv") and hasattr(m.qkv, "in_features"):
            E = int(m.qkv.in_features)
        elif hasattr(m, "qkv") and hasattr(m.qkv, "weight"):
            E = int(m.qkv.weight.shape[1])
        else:
            E = int(getattr(m, "dim", 0) or 0)
        if E > 0:
            out.append((E, E, "full"))
    if out:
        return out
    return out


def compute_paper_agy(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 1, 64, 64),
) -> Dict[str, Any]:
    """
    Paper α/γ/η vs full QKV baseline formulas.

    full:    α=3E², γ=3LE², η=2LE
    triple:  α=6rE, γ=6LrE, η=2rL
    shared:  α=4rE, γ=4LrE, η=rL
    """
    _, _, H, W = input_size
    if H % 64 or W % 64:
        raise ValueError(f"H/W must be multiples of 64, got {H}x{W}")
    num_patches = (H // 64) * (W // 64)
    token_counts = _stage_token_counts(model)
    depths = _stage_depths(model)
    attns = _collect_attention_ranks(model)

    alpha_cur = alpha_full = 0.0
    gamma_cur = gamma_full = 0.0
    eta_cur = eta_full = 0.0
    ranks = []
    graphs = []

    stage_idx = 0
    block_in_stage = 0
    for E, r, graph in attns:
        if stage_idx >= len(depths):
            break
        N = float(token_counts[stage_idx])
        ranks.append(int(r))
        graphs.append(graph)

        alpha_full += 3.0 * E * E
        gamma_full += 3.0 * N * E * E
        eta_full += 2.0 * E * N

        if graph == "shared":
            alpha_cur += 4.0 * r * E
            gamma_cur += 4.0 * N * r * E
            eta_cur += r * N
        elif graph == "triple":
            alpha_cur += 6.0 * r * E
            gamma_cur += 6.0 * N * r * E
            eta_cur += 2.0 * r * N
        else:
            alpha_cur += 3.0 * E * E
            gamma_cur += 3.0 * N * E * E
            eta_cur += 2.0 * E * N

        block_in_stage += 1
        if block_in_stage >= depths[stage_idx]:
            stage_idx += 1
            block_in_stage = 0

    gamma_cur *= num_patches
    gamma_full *= num_patches
    eta_cur *= num_patches
    eta_full *= num_patches

    def _ratio(a, b):
        return (a / b) if b > 0 else 0.0

    graph_set = sorted(set(graphs)) if graphs else ["unknown"]
    return {
        "qkv_graph": "+".join(graph_set),
        "ranks": ranks,
        "num_attention_layers": len(ranks),
        "alpha_qkv": alpha_cur,
        "alpha_qkv_full": alpha_full,
        "alpha_ratio": _ratio(alpha_cur, alpha_full),
        "gamma_qkv": gamma_cur,
        "gamma_qkv_full": gamma_full,
        "gamma_ratio": _ratio(gamma_cur, gamma_full),
        "eta_qkv": eta_cur,
        "eta_qkv_full": eta_full,
        "eta_ratio": _ratio(eta_cur, eta_full),
        "num_patches": num_patches,
        "reference_input": list(input_size),
    }


def compute_sensing_metrics(model: nn.Module, args=None) -> Dict[str, Any]:
    core = _unwrap(model)
    args = args or getattr(core, "args", None)
    use_sib = bool(getattr(core, "use_sib_acs", False) or getattr(args, "use_sib_acs", False))
    cs_ratio = int(getattr(args, "cs_ratio", getattr(core, "cs_ratio", -1)) or -1)
    n_input = int(getattr(core, "n_input", getattr(args, "n_input", 0)) or 0)
    phi_size = int(getattr(args, "sib_acs_phi_size", 16) if args is not None else 16)
    block_pix = phi_size * phi_size

    out = {
        "use_sib_acs": use_sib,
        "cs_ratio_pct": cs_ratio,
        "n_input_avg_budget": n_input,
        "phi_size": phi_size,
        "avg_measurements_per_block": float(n_input),
        "avg_sample_rate_per_block": float(n_input) / float(block_pix) if block_pix else 0.0,
        "measurement_bandwidth_ratio": float(n_input) / float(block_pix) if block_pix else 0.0,
    }
    if use_sib:
        M = int(getattr(core, "sib_max_measurements", n_input) or n_input)
        out["phi_bank_M"] = M
        out["max_measurements_per_block"] = M
        out["headroom_ratio_M_over_n_input"] = (float(M) / float(n_input)) if n_input else 0.0
        stats = getattr(core, "last_sib_stats", None)
        if isinstance(stats, dict):
            out["asa_N_avg"] = stats.get("N_avg")
            out["asa_N_mean"] = stats.get("N_mean")
            out["asa_N_min"] = stats.get("N_min")
            out["asa_N_max"] = stats.get("N_max")
            out["asa_alive"] = bool(
                stats.get("N_max") is not None
                and stats.get("N_min") is not None
                and float(stats["N_max"]) > float(stats["N_min"]) + 0.5
            )
    else:
        out["phi_bank_M"] = n_input
        out["max_measurements_per_block"] = n_input
        out["headroom_ratio_M_over_n_input"] = 1.0
    return out


def count_params(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": float(total),
        "trainable_params": float(trainable),
        "total_params_M": total / 1e6,
        "trainable_params_M": trainable / 1e6,
    }


def checkpoint_size_mb(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.isfile(path):
        return None
    return os.path.getsize(path) / (1024.0 * 1024.0)


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 1, 256, 256),
    warmup: int = 10,
    iterations: int = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    core = _unwrap(model)
    if device is None:
        device = next(core.parameters()).device
    core.eval()
    x = torch.randn(*input_size, device=device)
    # Warmup
    for _ in range(warmup):
        _ = core(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = core(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
    batch = max(1, int(input_size[0]))
    return {
        "latency_ms_per_image": (avg * 1000.0) / batch,
        "latency_std_ms": (std * 1000.0) / batch,
        "throughput_img_s": (batch / avg) if avg > 0 else 0.0,
        "benchmark_input": list(input_size),
        "benchmark_batch": batch,
        "benchmark_warmup": warmup,
        "benchmark_iterations": iterations,
    }


def build_efficiency_report(
    model: nn.Module,
    args=None,
    ckpt_path: Optional[str] = None,
    input_size_agy: Tuple[int, int, int, int] = (1, 1, 64, 64),
    run_latency: bool = False,
    latency_input: Tuple[int, int, int, int] = (1, 1, 256, 256),
    tag: str = "",
) -> Dict[str, Any]:
    # One forward so SIB stats populate if enabled.
    core = _unwrap(model)
    was_training = core.training
    core.eval()
    try:
        with torch.no_grad():
            device = next(core.parameters()).device
            x = torch.randn(1, 1, 64, 64, device=device)
            _ = core(x)
    except Exception:
        pass
    if was_training:
        core.train()

    report: Dict[str, Any] = {
        "tag": tag,
        "gen_model": getattr(args, "gen_model", type(core).__name__),
        "params": count_params(model),
        "qkv_paper": compute_paper_agy(model, input_size=input_size_agy),
        "sensing": compute_sensing_metrics(model, args=args),
        "checkpoint_size_MB": checkpoint_size_mb(ckpt_path),
    }
    if run_latency:
        try:
            report["runtime"] = measure_latency(model, input_size=latency_input)
        except Exception as exc:
            report["runtime"] = {"error": str(exc)}
    return report


def print_efficiency_report(report: Dict[str, Any]) -> None:
    p = report.get("params", {})
    q = report.get("qkv_paper", {})
    s = report.get("sensing", {})
    r = report.get("runtime", {})
    print("=" * 64)
    print(f"EFFICIENCY REPORT  tag={report.get('tag', '')}  model={report.get('gen_model')}")
    print("=" * 64)
    print(f"Total params:     {p.get('total_params_M', 0):.3f} M  "
          f"(trainable {p.get('trainable_params_M', 0):.3f} M)")
    ck = report.get("checkpoint_size_MB")
    if ck is not None:
        print(f"Checkpoint size:  {ck:.2f} MB")
    print("-" * 64)
    print(f"QKV graph:        {q.get('qkv_graph')}")
    print(f"ranks:            {q.get('ranks')}")
    # Absolute counts first (paper table), then optional ratio.
    print(
        f"alpha ABSOLUTE:   {q.get('alpha_qkv', 0):.0f}  "
        f"(full QKV ref {q.get('alpha_qkv_full', 0):.0f})  "
        f"[ratio {q.get('alpha_ratio', 0):.4f}]"
    )
    print(
        f"gamma ABSOLUTE:   {q.get('gamma_qkv', 0):.0f}  "
        f"(full QKV ref {q.get('gamma_qkv_full', 0):.0f})  "
        f"= {q.get('gamma_qkv', 0)/1e9:.6f} GFLOP vs {q.get('gamma_qkv_full', 0)/1e9:.6f} GFLOP  "
        f"[ratio {q.get('gamma_ratio', 0):.4f}]"
    )
    print(
        f"eta ABSOLUTE:     {q.get('eta_qkv', 0):.0f}  "
        f"(full QKV ref {q.get('eta_qkv_full', 0):.0f})  "
        f"[ratio {q.get('eta_ratio', 0):.4f}]"
    )
    print("-" * 64)
    print(f"Sensing: SIB={s.get('use_sib_acs')}  cs_ratio={s.get('cs_ratio_pct')}%"
          f"  N_avg={s.get('n_input_avg_budget')}  M_bank={s.get('phi_bank_M')}")
    print(f"Measurement bandwidth (avg samples/block pixels): "
          f"{s.get('measurement_bandwidth_ratio', 0):.4f}")
    if s.get("use_sib_acs"):
        print(f"ASA N mean/min/max: {s.get('asa_N_mean')}/{s.get('asa_N_min')}/{s.get('asa_N_max')} "
              f"alive={s.get('asa_alive')}")
    if r:
        if "error" in r:
            print(f"Runtime: ERROR {r['error']}")
        else:
            print(f"Latency: {r.get('latency_ms_per_image', 0):.2f} +/- "
                  f"{r.get('latency_std_ms', 0):.2f} ms/img")
            print(f"Throughput: {r.get('throughput_img_s', 0):.2f} img/s  "
                  f"input={r.get('benchmark_input')}")
    print("=" * 64)


def save_efficiency_report(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _default(o):
        if isinstance(o, (torch.Tensor,)):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_default)
    print(f"Wrote efficiency report: {path}")
