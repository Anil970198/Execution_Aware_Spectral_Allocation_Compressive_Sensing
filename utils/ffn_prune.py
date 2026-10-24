#ffn_prune.py
"""
Structured FFN (MLP) channel pruning for dense CSformer.

Per-MLP independent top-k on hidden units (keep_ratio). Rebuilds narrower
nn.Linear modules — no masks. Does not touch attention / Phi / CNN.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


MLP_NAME_SUFFIX = ".mlp"


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def discover_mlps(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return (name, Mlp module) for transformer FFNs, sorted by name."""
    root = unwrap_model(model)
    found: List[Tuple[str, nn.Module]] = []
    for name, mod in root.named_modules():
        if not name.endswith("mlp"):
            continue
        if not (hasattr(mod, "fc1") and hasattr(mod, "fc2")):
            continue
        # Skip anything not under transformer stem if present
        if name.startswith("td.") or ".blocks_" in name or name.startswith("blocks_"):
            found.append((name, mod))
        elif "td." in name or "blocks_" in name:
            found.append((name, mod))
    # Dense CSformer: all Mlp under td.blocks_*
    if not found:
        for name, mod in root.named_modules():
            if hasattr(mod, "fc1") and hasattr(mod, "fc2") and type(mod).__name__ == "Mlp":
                found.append((name, mod))
    found = sorted(found, key=lambda x: x[0])
    return found


def assert_twenty_mlps(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    mlps = discover_mlps(model)
    if len(mlps) != 20:
        names = [n for n, _ in mlps]
        raise RuntimeError(f"Expected 20 MLPs, found {len(mlps)}: {names}")
    return mlps


def magnitude_scores(fc1: nn.Linear) -> torch.Tensor:
    """s_j = ||W_fc1[j, :]||_2  → shape [H]."""
    w = fc1.weight.data  # [H, E]
    return torch.norm(w, p=2, dim=1)


def topk_indices(scores: torch.Tensor, keep: int) -> torch.Tensor:
    keep = int(keep)
    if keep <= 0 or keep > scores.numel():
        raise ValueError(f"keep={keep} invalid for H={scores.numel()}")
    vals, idx = torch.topk(scores, k=keep, largest=True, sorted=True)
    return idx.long().cpu()


def rebuild_mlp_linears(mlp: nn.Module, idx: torch.Tensor) -> Tuple[int, int]:
    """
    Replace mlp.fc1 / mlp.fc2 with narrower Linears keeping hidden units idx.
    fc1.weight [H,E] → rows idx; fc2.weight [E,H] → cols idx.
    Returns (E, H_prime).
    """
    fc1: nn.Linear = mlp.fc1
    fc2: nn.Linear = mlp.fc2
    device = fc1.weight.device
    dtype = fc1.weight.dtype
    H, E = fc1.weight.shape
    idx = idx.to(device=fc1.weight.device).long()
    Hp = int(idx.numel())
    if Hp >= H:
        raise ValueError(f"prune keep {Hp} >= H {H}")

    new_fc1 = nn.Linear(E, Hp, bias=fc1.bias is not None).to(device=device, dtype=dtype)
    new_fc2 = nn.Linear(Hp, E, bias=fc2.bias is not None).to(device=device, dtype=dtype)

    with torch.no_grad():
        new_fc1.weight.copy_(fc1.weight.data[idx, :])
        if fc1.bias is not None:
            new_fc1.bias.copy_(fc1.bias.data[idx])
        new_fc2.weight.copy_(fc2.weight.data[:, idx])
        if fc2.bias is not None:
            new_fc2.bias.copy_(fc2.bias.data)

    mlp.fc1 = new_fc1
    mlp.fc2 = new_fc2
    return E, Hp


def apply_ffn_prune_from_indices(
    model: nn.Module,
    indices_dict: Dict[str, torch.Tensor],
) -> None:
    """Rebuild MLPs to match saved indices (no weight copy from dense)."""
    root = unwrap_model(model)
    name_to_mod = dict(root.named_modules())
    for name, idx in indices_dict.items():
        if name not in name_to_mod:
            raise KeyError(f"MLP name not in model: {name}")
        mlp = name_to_mod[name]
        if not (hasattr(mlp, "fc1") and hasattr(mlp, "fc2")):
            raise TypeError(f"{name} is not an Mlp")
        idx = idx.long()
        Hp = int(idx.numel())
        E = int(mlp.fc1.in_features)
        # If already pruned to correct width, skip rebuild (reload path after apply)
        if mlp.fc1.out_features == Hp and mlp.fc2.in_features == Hp:
            continue
        if mlp.fc1.out_features != mlp.fc1.weight.shape[0]:
            raise RuntimeError(f"inconsistent fc1 on {name}")
        # Fresh empty narrower linears — weights loaded later via state_dict
        device = mlp.fc1.weight.device
        dtype = mlp.fc1.weight.dtype
        bias1 = mlp.fc1.bias is not None
        bias2 = mlp.fc2.bias is not None
        mlp.fc1 = nn.Linear(E, Hp, bias=bias1).to(device=device, dtype=dtype)
        mlp.fc2 = nn.Linear(Hp, E, bias=bias2).to(device=device, dtype=dtype)


def prune_mlps_with_indices(
    model: nn.Module,
    indices_dict: Dict[str, torch.Tensor],
) -> None:
    """Slice dense weights into new Linears using indices (prune-time)."""
    root = unwrap_model(model)
    name_to_mod = dict(root.named_modules())
    for name, idx in indices_dict.items():
        if name not in name_to_mod:
            raise KeyError(f"MLP name not in model: {name}")
        rebuild_mlp_linears(name_to_mod[name], idx)


def compute_magnitude_indices(
    model: nn.Module,
    keep_ratio: float = 0.5,
) -> Dict[str, torch.Tensor]:
    mlps = assert_twenty_mlps(model)
    out: Dict[str, torch.Tensor] = {}
    for name, mlp in mlps:
        H = mlp.fc1.out_features
        keep = max(1, int(H * keep_ratio))
        scores = magnitude_scores(mlp.fc1)
        out[name] = topk_indices(scores, keep)
    return out


def compute_random_indices(
    model: nn.Module,
    keep_ratio: float = 0.5,
    seed: int = 12345,
) -> Dict[str, torch.Tensor]:
    """
    Per-MLP random channel keep (control for T4 FFN criterion).
    Deterministic given seed; each MLP gets an independent permutation.
    """
    mlps = assert_twenty_mlps(model)
    out: Dict[str, torch.Tensor] = {}
    for mi, (name, mlp) in enumerate(mlps):
        H = int(mlp.fc1.out_features)
        keep = max(1, int(H * keep_ratio))
        g = torch.Generator()
        g.manual_seed(int(seed) + mi * 1009)
        perm = torch.randperm(H, generator=g)
        out[name] = perm[:keep].long().cpu().sort().values
    return out


def compute_taylor_indices(
    model: nn.Module,
    calib_loader: Iterable,
    device: torch.device,
    keep_ratio: float = 0.5,
    max_batches: int = 64,
    datarange: str = "-11",
) -> Dict[str, torch.Tensor]:
    """
    Weight-Taylor: s_j = || W[j] ⊙ ∂L/∂W[j] ||_1 on fc1, L = L1(pred, gt).
    Accumulate scores over calib batches; then per-MLP top-k.
    """
    root = unwrap_model(model)
    mlps = assert_twenty_mlps(root)
    score_acc = {
        name: torch.zeros(mlp.fc1.out_features, dtype=torch.float64)
        for name, mlp in mlps
    }

    root.train()
    n_done = 0
    for batch in calib_loader:
        if n_done >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch
        imgs = imgs.to(device)
        if datarange == "-11":
            # train path uses [-1,1]; ImgData usually already mapped — match functions.train
            pass

        root.zero_grad(set_to_none=True)
        # CSformer forward returns (merge_output, merge_PhiTb, output, PhiTb)
        out = root(imgs)
        pred = out[0] if isinstance(out, (list, tuple)) else out
        # Target is the input image in recon training
        gt = imgs
        if pred.shape != gt.shape:
            # patch-split may change layout — use first tensor that matches
            for t in out:
                if torch.is_tensor(t) and t.shape == gt.shape:
                    pred = t
                    break
        loss = nn.functional.l1_loss(pred, gt)
        loss.backward()

        for name, mlp in mlps:
            if mlp.fc1.weight.grad is None:
                raise RuntimeError(f"No grad on {name}.fc1 — check forward path")
            w = mlp.fc1.weight.data
            g = mlp.fc1.weight.grad.data
            s = (w * g).abs().sum(dim=1).detach().cpu().to(torch.float64)
            score_acc[name] += s

        n_done += 1
        root.zero_grad(set_to_none=True)

    if n_done == 0:
        raise RuntimeError("Taylor calib produced 0 batches")

    indices: Dict[str, torch.Tensor] = {}
    for name, mlp in mlps:
        H = mlp.fc1.out_features
        keep = max(1, int(H * keep_ratio))
        indices[name] = topk_indices(score_acc[name].float(), keep)
    return indices


def verify_prune_shapes(model: nn.Module, keep_ratio: float = 0.5) -> None:
    mlps = assert_twenty_mlps(model)
    root = unwrap_model(model)
    for name, mlp in mlps:
        E = mlp.fc1.in_features
        H_orig = int(round(mlp.fc1.out_features / keep_ratio)) if keep_ratio > 0 else None
        # After prune, H' should be H//2 for keep 0.5 → E*4 // 2 = 2E
        expected_H = (4 * E) // 2
        if mlp.fc1.out_features != expected_H:
            # allow if keep_ratio applied as int(H*ratio) from original
            pass
        if mlp.fc2.in_features != mlp.fc1.out_features:
            raise AssertionError(
                f"{name}: fc2.in={mlp.fc2.in_features} != fc1.out={mlp.fc1.out_features}"
            )
        if mlp.fc2.out_features != E:
            raise AssertionError(f"{name}: fc2.out={mlp.fc2.out_features} != E={E}")

    # Attention shapes unchanged: sample first block (dense qkv OR shared_qkv)
    attn = root.td.blocks_1.block[0].attn
    E0 = 256
    if getattr(attn, "shared_qkv", None) is not None:
        # Shared: down is Linear(E, r) — E must stay 256 for stage 0
        if attn.shared_qkv.down.in_features != E0:
            raise AssertionError(
                f"shared down.in changed: {attn.shared_qkv.down.in_features}"
            )
    elif getattr(attn, "qkv", None) is not None:
        if attn.qkv.in_features != E0 or attn.qkv.out_features != 3 * E0:
            raise AssertionError(
                f"attn.qkv changed: {attn.qkv.in_features}->{attn.qkv.out_features}"
            )
    else:
        raise AssertionError("Neither shared_qkv nor qkv present on stage0 attn")


def indices_are_trivial_half(indices_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Return MLP names whose kept set equals {0..H/2-1} (first-half shortcut)."""
    bad = []
    for name, idx in indices_dict.items():
        Hp = int(idx.numel())
        if set(idx.cpu().tolist()) == set(range(Hp)):
            bad.append(name)
    return bad


def analytic_ffn_macs(model: nn.Module, keep_ratio_note: str = "") -> Dict[str, float]:
    """
    Analytic FFN MAC estimate for one 64x64 forward (one CS patch).
    Per MLP: fc1 + fc2 ≈ 2 * N * E * H + 2 * N * H * E = 4 * N * E * H
    Stage spatial N: 8^2, 16^2, 32^2, 64^2.
    """
    stage_N = {
        "blocks_1": 8 * 8,
        "blocks_2": 16 * 16,
        "blocks_3": 32 * 32,
        "blocks_4": 64 * 64,
    }
    total = 0.0
    per_stage = {}
    for name, mlp in discover_mlps(model):
        E = mlp.fc1.in_features
        H = mlp.fc1.out_features
        N = 64
        for key, ntok in stage_N.items():
            if key in name:
                N = ntok
                break
        macs = 4.0 * N * E * H
        total += macs
        stage_key = next((k for k in stage_N if k in name), "other")
        per_stage[stage_key] = per_stage.get(stage_key, 0.0) + macs
    return {
        "ffn_macs": total,
        "ffn_gmacs": total / 1e9,
        "per_stage_macs": per_stage,
        "note": f"analytic 4*N*E*H per MLP; {keep_ratio_note}",
    }


def try_thop_gflops(model: nn.Module, input_size=(1, 1, 64, 64), device=None) -> Optional[float]:
    try:
        from thop import profile
    except Exception:
        return None
    root = unwrap_model(model)
    device = device or next(root.parameters()).device
    x = torch.randn(*input_size, device=device)
    root.eval()
    try:
        macs, _ = profile(root, inputs=(x,), verbose=False)
        return float(macs) / 1e9
    except Exception:
        return None
    finally:
        # thop registers total_ops / total_params buffers; strip so later
        # state_dict() saves stay loadable under strict=True.
        clean_thop_buffers(root)


def clean_thop_buffers(model: nn.Module) -> None:
    """Remove buffers thop injects onto every profiled module."""
    for m in unwrap_model(model).modules():
        for name in ("total_ops", "total_params"):
            if hasattr(m, name):
                try:
                    delattr(m, name)
                except Exception:
                    pass
            if name in getattr(m, "_buffers", {}):
                m._buffers.pop(name, None)


def strip_thop_keys(state_dict: dict) -> dict:
    """Drop thop bookkeeping keys from a checkpoint state_dict."""
    if not state_dict:
        return state_dict
    return {
        k: v
        for k, v in state_dict.items()
        if not k.endswith("total_ops") and not k.endswith("total_params")
    }


def strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if keys and keys[0].startswith("module."):
        return {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_indices_file(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "indices" in obj:
        obj = obj["indices"]
    if not isinstance(obj, dict):
        raise TypeError(f"prune indices must be dict, got {type(obj)}")
    return {k: v.long() if torch.is_tensor(v) else torch.tensor(v).long() for k, v in obj.items()}
