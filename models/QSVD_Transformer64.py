"""
QSVD_Transformer64.py
Self-contained QSVD-aware transformer stem for CSFormer.

This file mirrors the relevant parts of models/Transformer64.py but replaces the
fused qkv projection inside every Attention block with a joint QKV low-rank
factorization inspired by QSVD (fake_quant/svd_utils.py and
fake_quant/grad_info_utils.py in QSVD-main).

The file is intentionally kept separate from models/Transformer64.py so it can
be tested and used in isolation.
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Sequence

from models.ViT_helper import trunc_normal_
from models.Transformer64 import window_partition, window_reverse, pixel_upsample, tf2cnn, cnn2tf, Upsample


# ====================== Helper: DropPath ======================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ====================== Original Helper Classes ======================

class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 @ x2


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        else:
            self.act_layer = lambda x: nn.functional.leaky_relu_(x, 0.2)

    def forward(self, x):
        return self.act_layer(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)


# ====================== Adaptive rank helpers ======================

def gumbel_sigmoid(logits, tau=0.5):
    """Apply Gumbel-Sigmoid to logits for differentiable binary mask sampling."""
    def sample_gumbel(shape, dtype, device, eps=1e-20):
        U = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(U + eps) + eps)

    gumbel_noise = sample_gumbel(logits.shape, logits.dtype, logits.device)
    gumbel_logits = logits + gumbel_noise
    y_soft = torch.sigmoid(gumbel_logits / tau)
    return y_soft


class STE(torch.autograd.Function):
    """Straight-Through Estimator for binary thresholding."""
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# ====================== SVDLinear (adapted from QSVD) ======================

class SVDLinear(nn.Module):
    """
    Low-rank linear layer: y = A @ B @ x + bias.
    The factorization follows the QSVD implementation in
    QSVD-main/fake_quant/svd_utils.py.

    Optionally supports an adaptive rank mask over the singular values.
    """

    def __init__(self, U, S, V, bias=None, sigma_fuse="UV"):
        """
        Args:
            U: [out_features, rank]
            S: [rank]
            V: [in_features, rank]
            bias: [out_features] or None
            sigma_fuse: how to split singular values between A and B.
                        "UV", "U", "V", float in [0,1], or "adaptive".
        """
        super().__init__()
        self.truncation_rank = S.size(0)
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)
        self.BLinear = nn.Linear(V.size(0), V.size(1), bias=False)

        if sigma_fuse == "UV":
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == "U":
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == "V":
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()
        elif sigma_fuse == "adaptive":
            eps = 1e-6
            scale = U.abs().max(dim=0).values / (V.abs().max(dim=0).values + eps)
            self.ALinear.weight.data = U.mul(S ** (1.0 / (scale + 1.0))).contiguous()
            self.BLinear.weight.data = V.t().mul(
                (S ** (scale / (scale + 1.0))).view(-1, 1)
            ).contiguous()
        elif isinstance(sigma_fuse, (int, float)) and 0.0 <= float(sigma_fuse) <= 1.0:
            beta = float(sigma_fuse)
            self.ALinear.weight.data = U.mul(S ** (1.0 - beta)).contiguous()
            self.BLinear.weight.data = V.t().mul((S ** beta).view(-1, 1)).contiguous()
        else:
            raise ValueError(f"Unsupported sigma_fuse mode: {sigma_fuse}")

        if bias is not None:
            self.ALinear.bias.data = bias

    def forward(self, x, mask=None):
        y = self.BLinear(x)
        if mask is not None:
            y = y * mask
        y = self.ALinear(y)
        return y


# ====================== QSVD helpers ======================

def _split_qkv_weight(original_qkv: nn.Linear):
    """Split a fused [dim*3, dim] qkv weight into Wq, Wk, Wv and biases."""
    dim = original_qkv.in_features
    w_q = original_qkv.weight.data[:dim, :].clone()
    w_k = original_qkv.weight.data[dim:2 * dim, :].clone()
    w_v = original_qkv.weight.data[2 * dim:, :].clone()

    if original_qkv.bias is not None:
        bias_q = original_qkv.bias.data[:dim].clone()
        bias_k = original_qkv.bias.data[dim:2 * dim].clone()
        bias_v = original_qkv.bias.data[2 * dim:].clone()
    else:
        bias_q = bias_k = bias_v = None

    return w_q, w_k, w_v, bias_q, bias_k, bias_v


def _compute_rank(dim: int, param_ratio: float, rank_align: int = 1) -> int:
    """Compute a target rank for a single qkv layer, matching QSVD's logic."""
    param_ratio = float(param_ratio)
    param_ratio = min(param_ratio, 2.0)
    n_params = dim * dim
    compressed_params = int(n_params * param_ratio)
    rank = compressed_params // (dim + dim)
    rank = int(np.ceil(rank / rank_align) * rank_align)
    rank = max(1, min(rank, dim))
    return rank


def prepare_qsvd_svd_info(
    original_qkv: nn.Linear,
    act_aware: bool = False,
    scaling_diag_matrix: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Perform full joint QKV SVD on a fused qkv Linear and return the full
    SVD factors needed for importance scoring and rank allocation.

    Args:
        original_qkv: fused nn.Linear(dim, dim*3).
        act_aware: if True, apply ASVD-style activation scaling before SVD.
        scaling_diag_matrix: per-channel scaling [dim]. If None and act_aware
            is True, an all-ones scaling is used (no effect).

    Returns:
        svd_info dict with keys:
            'U_full': [3*dim, dim]
            'S_full': [dim]
            'V_full': [dim, dim]
            'scaling_diag_matrix': [dim] or None
    """
    device = original_qkv.weight.device
    dtype = original_qkv.weight.dtype
    dim = original_qkv.in_features

    w_q, w_k, w_v, _, _, _ = _split_qkv_weight(original_qkv)
    w_concat = torch.cat([w_q, w_k, w_v], dim=0).to(device).float()

    if act_aware and scaling_diag_matrix is not None:
        scaling_diag_matrix = scaling_diag_matrix.to(device).float()
        w_concat = w_concat * scaling_diag_matrix.view(1, -1)

    U_full, S_full, Vt = torch.linalg.svd(w_concat, full_matrices=False)
    V_full = Vt.T

    svd_info = {
        'U_full': U_full.cpu(),
        'S_full': S_full.cpu(),
        'V_full': V_full.cpu(),
        'scaling_diag_matrix': scaling_diag_matrix.cpu() if scaling_diag_matrix is not None else None,
        'dim': dim,
    }
    return svd_info


def build_qsvd_projections(
    svd_info: Dict[str, torch.Tensor],
    rank_or_indices: Union[int, torch.Tensor, List[int], np.ndarray],
    sigma_fuse: str = "UV",
    bias_q: Optional[torch.Tensor] = None,
    bias_k: Optional[torch.Tensor] = None,
    bias_v: Optional[torch.Tensor] = None,
) -> Tuple[SVDLinear, SVDLinear, SVDLinear, int, torch.Tensor]:
    """
    Build three SVDLinear projections (Q/K/V) from full SVD info using either
    a target rank or explicitly selected singular-value indices.

    Args:
        svd_info: dict returned by prepare_qsvd_svd_info.
        rank_or_indices: int -> keep top-rank singular values;
                         tensor/list -> exact indices to keep.
        sigma_fuse: singular-value splitting mode.
        bias_q, bias_k, bias_v: optional biases for Q/K/V.

    Returns:
        (q_proj, k_proj, v_proj, rank, selected_indices)
    """
    device = svd_info['U_full'].device
    U_full = svd_info['U_full'].to(device).float()
    S_full = svd_info['S_full'].to(device).float()
    V_full = svd_info['V_full'].to(device).float()
    dim = svd_info['dim']

    if isinstance(rank_or_indices, int):
        rank = max(1, min(rank_or_indices, S_full.numel()))
        selected_indices = torch.arange(rank, device=device)
    else:
        selected_indices = torch.as_tensor(rank_or_indices, device=device, dtype=torch.long)
        selected_indices = torch.unique(selected_indices).sort().values
        rank = selected_indices.numel()

    U = U_full[:, selected_indices]
    S = S_full[selected_indices]
    V = V_full[:, selected_indices]

    # Compensate activation-aware scaling in V.
    if svd_info.get('scaling_diag_matrix') is not None:
        scaling_diag_matrix = svd_info['scaling_diag_matrix'].to(device).float()
        V = V / scaling_diag_matrix.view(-1, 1)

    U_q = U[:dim, :]
    U_k = U[dim:2 * dim, :]
    U_v = U[2 * dim:, :]

    q_proj = SVDLinear(U_q, S, V, bias=bias_q, sigma_fuse=sigma_fuse)
    k_proj = SVDLinear(U_k, S, V, bias=bias_k, sigma_fuse=sigma_fuse)
    v_proj = SVDLinear(U_v, S, V, bias=bias_v, sigma_fuse=sigma_fuse)

    # Move back to the device/dtype of the original weight if available.
    return q_proj, k_proj, v_proj, rank, selected_indices


def apply_qsvd_to_qkv(
    original_qkv: nn.Linear,
    rank_ratio: float = 1.0,
    sigma_fuse: str = "UV",
    act_aware: bool = False,
    scaling_diag_matrix: Optional[torch.Tensor] = None,
    rank_align: int = 1,
) -> Tuple[SVDLinear, SVDLinear, SVDLinear, int, Dict[str, torch.Tensor]]:
    """
    Convenience function: full joint QKV SVD + rank truncation + build projections.

    Returns:
        (q_proj, k_proj, v_proj, rank, svd_info)
    """
    dim = original_qkv.in_features
    rank = _compute_rank(dim, rank_ratio, rank_align=rank_align)

    svd_info = prepare_qsvd_svd_info(
        original_qkv,
        act_aware=act_aware,
        scaling_diag_matrix=scaling_diag_matrix,
    )

    _, _, _, bias_q, bias_k, bias_v = _split_qkv_weight(original_qkv)
    q_proj, k_proj, v_proj, rank, _ = build_qsvd_projections(
        svd_info, rank, sigma_fuse=sigma_fuse,
        bias_q=bias_q, bias_k=bias_k, bias_v=bias_v,
    )

    device = original_qkv.weight.device
    dtype = original_qkv.weight.dtype
    return (
        q_proj.to(device).to(dtype),
        k_proj.to(device).to(dtype),
        v_proj.to(device).to(dtype),
        rank,
        svd_info,
    )


# ====================== Importance scoring ======================

@torch.enable_grad()
def compute_qsvd_importance_scores(
    attentions: Sequence[nn.Module],
    calibration_data: Sequence,
    loss_fn,
    device: Union[str, torch.device] = 'cpu',
    accumulation_steps: int = 1,
) -> Dict[nn.Module, torch.Tensor]:
    """
    Compute QSVD-style per-singular-value importance scores using gradients
    w.r.t. the original fused qkv weight.

    This matches the computation in QSVD-main/fake_quant/grad_info_utils.py:
        S_grad      = sum(U_full * (grad_cat @ V_full), dim=0)
        importance  = sqrt(mean(S_grad ** 2))

    Args:
        attentions: sequence of Attention modules with prepared svd_info.
        calibration_data: list of input batches. Each batch can be a tensor
            [B, N, C] or a tuple/list whose first element is the tensor.
        loss_fn: callable(pred, target) -> scalar loss.
        device: device to run on.
        accumulation_steps: number of batches to accumulate gradients before
            updating importance scores.

    Returns:
        Dict mapping each attention module to a 1-D importance tensor of length
        equal to the full rank (dim).
    """
    device = torch.device(device)
    scores = {}

    for attn in attentions:
        if not hasattr(attn, 'svd_info') or attn.svd_info is None:
            raise RuntimeError(
                "Attention module does not have svd_info. "
                "Call attn.prepare_qsvd_svd_info() or attn.apply_qsvd() first."
            )

        attn.to(device)
        attn.train()

        # Ensure original qkv weight requires grad for calibration.
        if hasattr(attn, 'qkv'):
            attn.unfreeze_original_qkv()

        svd_info = attn.svd_info
        U_full = svd_info['U_full'].to(device).float()
        V_full = svd_info['V_full'].to(device).float()
        dim = svd_info['dim']

        S_grad_squared_accum = torch.zeros(dim, device=device)
        batch_count = 0

        for batch in calibration_data:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            # Forward using the ORIGINAL qkv so gradients are w.r.t. the
            # original [3*dim, dim] weight.
            output = attn(x, use_original_qkv=True)
            target = x.detach()
            loss = loss_fn(output, target)
            loss = loss / accumulation_steps
            loss.backward()

            batch_count += 1

            if batch_count % accumulation_steps == 0:
                grad = attn.qkv.weight.grad
                if grad is None:
                    attn.zero_grad()
                    continue

                grad = grad.detach().to(device).float()
                grad_q = grad[:dim, :]
                grad_k = grad[dim:2 * dim, :]
                grad_v = grad[2 * dim:, :]
                grad_cat = torch.cat([grad_q, grad_k, grad_v], dim=0)

                # Equivalent to diag(U_full^T @ grad_cat @ V_full).
                S_grad = torch.sum(U_full * (grad_cat @ V_full), dim=0)
                S_grad_squared_accum += S_grad.pow(2)

                attn.zero_grad()

        # Normalize any remaining accumulated gradients.
        if batch_count > 0:
            n_updates = batch_count // accumulation_steps
            if n_updates > 0:
                S_grad_squared_accum = S_grad_squared_accum / n_updates
            importance = S_grad_squared_accum.sqrt().cpu()
        else:
            importance = torch.zeros(dim)

        scores[attn] = importance

    return scores


def allocate_ranks_global(
    importance_scores: Dict[nn.Module, torch.Tensor],
    target_param_ratio: float,
    rank_align: int = 1,
) -> Dict[nn.Module, torch.Tensor]:
    """
    Global cross-layer rank allocation.

    Collects all singular values from all attention layers, sorts them by
    importance, and selects the top ones until the total parameter budget is
    reached. Returns selected singular-value indices for every layer.

    Args:
        importance_scores: dict {attention_module: importance_tensor [dim]}.
        target_param_ratio: desired ratio of compressed qkv params vs original
            qkv params (e.g. 0.75).
        rank_align: align each layer's rank to a multiple of this value.

    Returns:
        Dict {attention_module: sorted selected singular-value indices}.
    """
    if not importance_scores:
        return {}

    # Compute original and target total qkv parameter counts.
    original_total = 0
    cost_per_sv = {}
    dims = {}
    for attn, scores in importance_scores.items():
        dim = scores.numel()
        dims[attn] = dim
        original_total += 3 * dim * dim
        # Each kept singular value adds 2*dim params per Q/K/V projection.
        cost_per_sv[attn] = 6 * dim

    target_total = target_param_ratio * original_total

    # Collect candidates: (attention_module, singular_index, importance).
    candidates = []
    for attn, scores in importance_scores.items():
        for idx, score in enumerate(scores):
            candidates.append((attn, idx, score.item()))

    # Sort by importance descending.
    candidates.sort(key=lambda x: x[2], reverse=True)

    selected = {attn: [] for attn in importance_scores}
    current_total = 0

    for attn, sv_idx, score in candidates:
        cost = cost_per_sv[attn]
        if current_total + cost <= target_total:
            selected[attn].append(sv_idx)
            current_total += cost

    # Convert to sorted tensors and optionally align rank.
    selected_tensors = {}
    for attn, indices in selected.items():
        dim = dims[attn]
        if not indices:
            # Ensure at least rank 1 to avoid empty layers.
            indices = [0]
        indices_t = torch.tensor(indices, dtype=torch.long).sort().values

        if rank_align > 1:
            n = len(indices_t)
            aligned_n = int(np.ceil(n / rank_align) * rank_align)
            aligned_n = min(aligned_n, dim)
            if aligned_n > n:
                # Add the next highest-importance indices not yet selected.
                remaining = [i for i in range(dim) if i not in set(indices_t.tolist())]
                remaining.sort(key=lambda i: importance_scores[attn][i].item(), reverse=True)
                extra = remaining[:aligned_n - n]
                indices_t = torch.cat([
                    indices_t,
                    torch.tensor(extra, dtype=torch.long),
                ]).sort().values.unique()

        selected_tensors[attn] = indices_t

    actual_ratio = current_total / original_total if original_total > 0 else 0.0
    print(f"Global rank allocation: target ratio {target_param_ratio:.4f}, "
          f"actual ratio {actual_ratio:.4f}, total params {int(current_total)}")

    return selected_tensors


# ====================== Attention / Block / StageBlock ======================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 window_size=16, use_qsvd=False, qsvd_rank_ratio=1.0, sigma_fuse="UV",
                 act_aware=False, rank_align=1,
                 adaptive_rank=False, target_param_ratio=0.75, adaptive_rank_tau=0.4,
                 adaptive_rank_simple=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        self.use_qsvd = use_qsvd
        self.qsvd_rank_ratio = qsvd_rank_ratio
        self.sigma_fuse = sigma_fuse
        self.act_aware = act_aware
        self.rank_align = rank_align

        # Adaptive rank allocation state.
        self.adaptive_rank = adaptive_rank
        self.target_param_ratio = target_param_ratio
        self.adaptive_rank_tau = adaptive_rank_tau
        self.adaptive_rank_simple = adaptive_rank_simple
        self.qsvd_mask_logits = None
        self.qsvd_mask_cache = None
        self.qsvd_mask_frozen = False

        # Original fused qkv. Always kept because it is needed for:
        #   (a) weight initialization, and
        #   (b) exact importance scoring (use_original_qkv=True).
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.qkv_q = None
        self.qkv_k = None
        self.qkv_v = None
        self.qsvd_rank = None
        self.svd_info = None
        self._qsvd_applied = False

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()

        # Relative position bias (unchanged)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        if use_qsvd:
            if self.adaptive_rank:
                # For adaptive rank, start from full-rank projections so the
                # learnable mask can select any subset of singular values.
                self.apply_qsvd(rank_or_ratio=self.dim, sigma_fuse=sigma_fuse,
                                act_aware=act_aware, rank_align=rank_align)
            else:
                self.apply_qsvd(rank_ratio=qsvd_rank_ratio, sigma_fuse=sigma_fuse,
                                act_aware=act_aware, rank_align=rank_align)

    def prepare_qsvd_svd_info(self, act_aware: Optional[bool] = None,
                              scaling_diag_matrix: Optional[torch.Tensor] = None):
        """Prepare and store full joint QKV SVD info for this attention layer."""
        if act_aware is None:
            act_aware = self.act_aware
        self.svd_info = prepare_qsvd_svd_info(
            self.qkv,
            act_aware=act_aware,
            scaling_diag_matrix=scaling_diag_matrix,
        )

    def _build_adaptive_mask_predictor(self, full_rank: int):
        """Create a learnable mask predictor over the full-rank singular values."""
        if self.adaptive_rank_simple:
            # Simple learnable-vector predictor (analogous to LowrankLinearSimple).
            # Initialize so masks are mostly active (near full rank) but not so
            # saturated that the compression loss cannot push them to zero.
            # Larger singular values get higher logits so the initial tendency is
            # to keep the most important ones.
            init_vector = torch.linspace(2.5, -0.5, full_rank)
            self.qsvd_mask_logits = nn.Parameter(init_vector)
        else:
            # Small hypernetwork predictor (analogous to LowrankLinear).
            self.qsvd_mask_logits = nn.GRU(full_rank, full_rank, batch_first=True, bidirectional=True)
            self.qsvd_mask_proj = nn.Linear(full_rank * 2, 1)
            self.qsvd_mask_z = nn.Parameter(torch.randn(1, full_rank, full_rank))
            self.qsvd_mask_z.requires_grad = False

    def get_qsvd_mask(self, force_eval: bool = False):
        """
        Compute the differentiable or hard mask over singular values.

        During training, returns a Gumbel-Sigmoid + STE mask and caches the
        soft mask for use in the alignment loss.
        During evaluation or when frozen, returns a cached hard top-k mask.
        """
        if not self.adaptive_rank or self.qsvd_mask_logits is None:
            return None

        if (not self.training or force_eval) and self.qsvd_mask_cache is not None:
            return self.qsvd_mask_cache

        if self.adaptive_rank_simple:
            logits = self.qsvd_mask_logits
        else:
            self.qsvd_mask_z.requires_grad = False
            out, _ = self.qsvd_mask_logits(self.qsvd_mask_z)
            out = out[0, :, :]
            logits = self.qsvd_mask_proj(out)[:, 0]

        if self.training and not force_eval:
            soft_mask = gumbel_sigmoid(logits, tau=self.adaptive_rank_tau)
            mask = STE.apply(soft_mask)
            self.qsvd_mask_cache = mask
            self.qsvd_soft_mask_cache = soft_mask
        else:
            probs = torch.sigmoid(logits)
            hard_mask = probs > 0.5
            rank = int(hard_mask.sum().item())
            rank = max(1, min(rank, len(probs)))
            mask = torch.zeros_like(probs)
            # Keep the top-`rank` logits by value, which aligns with alignment loss.
            topk_indices = torch.topk(probs, rank).indices
            mask[topk_indices] = 1.0
            self.qsvd_mask_cache = mask.detach()
            self.qsvd_soft_mask_cache = None

        return self.qsvd_mask_cache

    def get_qsvd_selected_indices(self):
        """Return the sorted singular-value indices selected by the current mask."""
        mask = self.get_qsvd_mask(force_eval=True)
        if mask is None:
            return None
        indices = torch.nonzero(mask > 0.5, as_tuple=False).view(-1)
        return indices.sort().values

    def freeze_qsvd_mask(self):
        """Freeze the rank mask so its selected indices become fixed."""
        if not self.adaptive_rank or self.qsvd_mask_logits is None:
            return
        self.qsvd_mask_frozen = True
        if self.adaptive_rank_simple:
            self.qsvd_mask_logits.requires_grad = False
        else:
            for p in self.qsvd_mask_logits.parameters():
                p.requires_grad = False
            for p in self.qsvd_mask_proj.parameters():
                p.requires_grad = False
        # Materialize the cached hard mask.
        _ = self.get_qsvd_mask(force_eval=True)

    def unfreeze_qsvd_mask(self):
        """Unfreeze the rank mask for further learning."""
        if not self.adaptive_rank or self.qsvd_mask_logits is None:
            return
        self.qsvd_mask_frozen = False
        self.qsvd_mask_cache = None
        self.qsvd_soft_mask_cache = None
        if self.adaptive_rank_simple:
            self.qsvd_mask_logits.requires_grad = True
        else:
            for p in self.qsvd_mask_logits.parameters():
                p.requires_grad = True
            for p in self.qsvd_mask_proj.parameters():
                p.requires_grad = True

    def apply_qsvd(
        self,
        rank_or_ratio: Optional[Union[int, float]] = None,
        rank_ratio: Optional[float] = None,
        selected_indices: Optional[Union[torch.Tensor, List[int], np.ndarray]] = None,
        sigma_fuse: Optional[str] = None,
        act_aware: Optional[bool] = None,
        scaling_diag_matrix: Optional[torch.Tensor] = None,
        rank_align: Optional[int] = None,
    ):
        """
        Apply QSVD to this Attention module.

        Three usage modes:
          1. rank_or_ratio is an int -> fixed rank.
          2. rank_or_ratio is a float (or rank_ratio is passed) -> ratio-based rank.
          3. selected_indices is given -> use exactly these singular-value indices.
        """
        if sigma_fuse is None:
            sigma_fuse = self.sigma_fuse
        if act_aware is None:
            act_aware = self.act_aware
        if rank_align is None:
            rank_align = self.rank_align

        # Recompute full SVD info whenever we are doing ratio/int-based rank
        # selection, so that weight changes (e.g. loading a checkpoint) are
        # picked up. If selected_indices are supplied we assume svd_info was
        # already prepared from the correct weights.
        if selected_indices is None or self.svd_info is None:
            self.prepare_qsvd_svd_info(act_aware=act_aware,
                                       scaling_diag_matrix=scaling_diag_matrix)

        _, _, _, bias_q, bias_k, bias_v = _split_qkv_weight(self.qkv)

        if selected_indices is not None:
            q_proj, k_proj, v_proj, rank, _ = build_qsvd_projections(
                self.svd_info, selected_indices, sigma_fuse=sigma_fuse,
                bias_q=bias_q, bias_k=bias_k, bias_v=bias_v,
            )
        elif isinstance(rank_or_ratio, int):
            q_proj, k_proj, v_proj, rank, _ = build_qsvd_projections(
                self.svd_info, rank_or_ratio, sigma_fuse=sigma_fuse,
                bias_q=bias_q, bias_k=bias_k, bias_v=bias_v,
            )
        else:
            ratio = rank_or_ratio if rank_or_ratio is not None else (rank_ratio if rank_ratio is not None else self.qsvd_rank_ratio)
            rank = _compute_rank(self.dim, ratio, rank_align=rank_align)
            q_proj, k_proj, v_proj, rank, _ = build_qsvd_projections(
                self.svd_info, rank, sigma_fuse=sigma_fuse,
                bias_q=bias_q, bias_k=bias_k, bias_v=bias_v,
            )

        device = self.qkv.weight.device
        dtype = self.qkv.weight.dtype
        self.qkv_q = q_proj.to(device).to(dtype)
        self.qkv_k = k_proj.to(device).to(dtype)
        self.qkv_v = v_proj.to(device).to(dtype)
        self.qsvd_rank = rank
        self.use_qsvd = True
        self._qsvd_applied = True

        # Build adaptive mask predictor if requested and not yet built.
        if self.adaptive_rank and self.qsvd_mask_logits is None:
            self._build_adaptive_mask_predictor(full_rank=self.dim)

        # The original qkv is no longer used in the forward pass; freeze it
        # so optimizers do not waste memory/compute on it. It is kept only for
        # exact importance scoring (use_original_qkv=True).
        self.freeze_original_qkv()

    def convert_adaptive_to_fixed_rank(self, sigma_fuse: Optional[str] = None):
        """
        Convert an adaptive-rank Attention module to a fixed-rank module using
        the currently selected singular-value indices. This removes the mask
        predictor and makes the module suitable for standard inference.
        """
        if not self.adaptive_rank:
            return
        selected_indices = self.get_qsvd_selected_indices()
        if selected_indices is None or len(selected_indices) == 0:
            selected_indices = torch.arange(1, dtype=torch.long, device=self.qkv_q.ALinear.weight.device)
        # Re-apply QSVD with the selected indices.
        self.apply_qsvd(selected_indices=selected_indices, sigma_fuse=sigma_fuse)
        # Remove adaptive mask parameters.
        self.adaptive_rank = False
        self.qsvd_mask_logits = None
        self.qsvd_mask_cache = None
        self.qsvd_soft_mask_cache = None
        self.qsvd_mask_frozen = False
        if hasattr(self, 'qsvd_mask_proj'):
            self.qsvd_mask_proj = None
        if hasattr(self, 'qsvd_mask_z'):
            self.qsvd_mask_z = None

    def freeze_original_qkv(self):
        """Freeze the original fused qkv parameters."""
        if self.qkv is None:
            return
        self.qkv.weight.requires_grad_(False)
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad_(False)

    def unfreeze_original_qkv(self):
        """Unfreeze the original fused qkv parameters (e.g. for calibration)."""
        if self.qkv is None:
            return
        self.qkv.weight.requires_grad_(True)
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad_(True)

    def discard_original_qkv(self):
        """
        Remove the original fused qkv layer to recover the parameter/memory
        savings of QSVD. After calling this, only the low-rank qkv_q/k/v
        projections remain; use_original_qkv can no longer be used.
        """
        if not self.use_qsvd or self.qkv is None:
            return
        self.qkv = None

    def set_qsvd_projections(self, q_proj, k_proj, v_proj, rank, svd_info=None):
        """Manually attach pre-built SVDLinear projections."""
        self.qkv_q = q_proj
        self.qkv_k = k_proj
        self.qkv_v = v_proj
        self.qsvd_rank = rank
        self._qsvd_applied = True
        if svd_info is not None:
            self.svd_info = svd_info

    def forward(self, x, use_original_qkv: bool = False):
        B, N, C = x.shape
        assert C == self.dim, f"Expected dim {self.dim}, got {C}"

        mask = None
        if self.use_qsvd and self.adaptive_rank and not use_original_qkv:
            mask = self.get_qsvd_mask()

        if not self.use_qsvd or use_original_qkv or not self._qsvd_applied:
            if self.qkv is None:
                raise RuntimeError(
                    "Original qkv has been discarded; cannot use the full-rank path."
                )
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = self.qkv_q(x, mask=mask).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.qkv_k(x, mask=mask).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.qkv_v(x, mask=mask).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16,
                 use_qsvd=False, qsvd_rank_ratio=1.0, sigma_fuse="UV",
                 act_aware=False, rank_align=1,
                 adaptive_rank=False, target_param_ratio=0.75, adaptive_rank_tau=0.4,
                 adaptive_rank_simple=True):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size,
            use_qsvd=use_qsvd, qsvd_rank_ratio=qsvd_rank_ratio, sigma_fuse=sigma_fuse,
            act_aware=act_aware, rank_align=rank_align,
            adaptive_rank=adaptive_rank, target_param_ratio=target_param_ratio,
            adaptive_rank_tau=adaptive_rank_tau, adaptive_rank_simple=adaptive_rank_simple)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class StageBlock(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16,
                 use_qsvd=False, qsvd_rank_ratio=1.0, sigma_fuse="UV",
                 act_aware=False, rank_align=1,
                 adaptive_rank=False, target_param_ratio=0.75, adaptive_rank_tau=0.4,
                 adaptive_rank_simple=True):
        super().__init__()
        self.depth = depth
        models = [Block(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            act_layer=act_layer, norm_layer=norm_layer, window_size=window_size,
            use_qsvd=use_qsvd, qsvd_rank_ratio=qsvd_rank_ratio, sigma_fuse=sigma_fuse,
            act_aware=act_aware, rank_align=rank_align,
            adaptive_rank=adaptive_rank, target_param_ratio=target_param_ratio,
            adaptive_rank_tau=adaptive_rank_tau, adaptive_rank_simple=adaptive_rank_simple) for _ in range(depth)]
        self.block = nn.Sequential(*models)

    def forward(self, x):
        return self.block(x)


# ====================== Activation-aware calibration helper ======================

@torch.no_grad()
def calibrate_qkv_scaling(
    attentions: Sequence[Attention],
    calibration_data: Sequence,
    method: str = 'abs_mean',
    device: Union[str, torch.device] = 'cpu',
) -> Dict[Attention, torch.Tensor]:
    """
    Compute ASVD-style per-channel scaling matrices for each Attention module.

    IMPORTANT: calibration_data here is assumed to contain the actual inputs
    to each qkv projection (i.e. after LayerNorm). When integrating with the
    full CSFormer model, you should extract these inputs from the real forward
    pass or use forward hooks on the Attention modules.

    Args:
        attentions: sequence of Attention modules.
        calibration_data: list of input tensors [B, N, dim] or tuples whose
            first element is the tensor. The same tensor is fed to every
            attention module; for per-attention inputs call this helper once
            per module with the appropriate data.
        method: 'abs_mean' or 'abs_max'.
        device: device to run on.

    Returns:
        Dict {attention_module: scaling_diag_matrix [dim]}.
    """
    device = torch.device(device)
    scaling_matrices = {}

    for attn in attentions:
        attn.to(device)
        attn.eval()
        dim = attn.dim
        scaling = torch.zeros(dim, device=device)

        for batch in calibration_data:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            if x.shape[-1] != dim:
                raise ValueError(
                    f"Attention dim={dim} but calibration input has last dim={x.shape[-1]}. "
                    "Pass per-attention calibration data or pre-extract qkv inputs."
                )
            # x shape: [B, N, dim]. Aggregate over batch and token dims.
            if "abs_mean" in method:
                scaling += x.abs().mean(dim=[0, 1]).float()
            elif "abs_max" in method:
                scaling = torch.where(
                    x.abs().amax(dim=[0, 1]).float() > scaling,
                    x.abs().amax(dim=[0, 1]).float(),
                    scaling,
                )
            else:
                raise ValueError(f"Unknown calibration method: {method}")

        # Average over batches for abs_mean; abs_max already keeps running max.
        if "abs_mean" in method:
            n_batches = len(calibration_data)
            if n_batches > 0:
                scaling = scaling / n_batches

        # Avoid zeros.
        scaling = scaling + 1e-6
        scaling_matrices[attn] = scaling.cpu()

    return scaling_matrices


# ====================== Model-wide QSVD helpers ======================

def collect_attention_modules(model: nn.Module) -> List[Attention]:
    """Recursively collect all Attention modules from a model."""
    attentions = []
    for module in model.modules():
        if isinstance(module, Attention):
            attentions.append(module)
    return attentions


def apply_qsvd_to_model(
    model: nn.Module,
    rank_ratio: float = 1.0,
    sigma_fuse: str = "UV",
    act_aware: bool = False,
    rank_align: int = 1,
    scaling_matrices: Optional[Dict[Attention, torch.Tensor]] = None,
):
    """Apply fixed-ratio QSVD to every Attention module in a model."""
    attentions = collect_attention_modules(model)
    for idx, attn in enumerate(attentions):
        scaling = scaling_matrices.get(attn, None) if scaling_matrices else None
        attn.apply_qsvd(
            rank_ratio=rank_ratio,
            sigma_fuse=sigma_fuse,
            act_aware=act_aware,
            scaling_diag_matrix=scaling,
            rank_align=rank_align,
        )
        print(f"Applied QSVD to Attention {idx}: rank {attn.qsvd_rank}/{attn.dim}")


def discard_original_qkv_in_model(model: nn.Module):
    """
    Discard the original full-rank qkv weights in every QSVD Attention module.
    This recovers the true parameter/memory savings after compression.
    """
    discarded = 0
    for attn in collect_attention_modules(model):
        if attn.use_qsvd and attn.qkv is not None:
            attn.discard_original_qkv()
            discarded += 1
    print(f"Discarded original qkv weights from {discarded} Attention layers.")


def freeze_adaptive_ranks_in_model(model: nn.Module):
    """Freeze all adaptive rank mask predictors in the model."""
    frozen = 0
    for attn in collect_attention_modules(model):
        if attn.adaptive_rank:
            attn.freeze_qsvd_mask()
            frozen += 1
    print(f"Frozen adaptive rank masks in {frozen} Attention layers.")


def convert_adaptive_ranks_to_fixed_in_model(model: nn.Module, sigma_fuse: str = "UV"):
    """
    Convert every adaptive-rank Attention module to a fixed-rank module
    using the currently selected singular-value indices, and return the
    per-layer ranks so they can be saved in the checkpoint.
    """
    ranks = []
    converted = 0
    for attn in collect_attention_modules(model):
        if attn.adaptive_rank:
            attn.convert_adaptive_to_fixed_rank(sigma_fuse=sigma_fuse)
            converted += 1
        ranks.append(int(attn.qsvd_rank) if attn.qsvd_rank is not None else int(attn.dim))
    print(f"Converted {converted} adaptive-rank Attention layers to fixed ranks.")
    return ranks


def prepare_qsvd_model(
    model: nn.Module,
    act_aware: bool = False,
):
    """Prepare full SVD info for every Attention module (no truncation yet)."""
    attentions = collect_attention_modules(model)
    for idx, attn in enumerate(attentions):
        attn.prepare_qsvd_svd_info(act_aware=act_aware)
        print(f"Prepared QSVD info for Attention {idx}: full rank {attn.dim}")


def calibrate_and_rank_model(
    model: nn.Module,
    calibration_data: Sequence,
    loss_fn,
    target_param_ratio: float = 0.75,
    device: Union[str, torch.device] = 'cpu',
    rank_align: int = 1,
    sigma_fuse: str = "UV",
) -> Dict[nn.Module, torch.Tensor]:
    """
    End-to-end pipeline:
      1. Prepare full SVD info for all Attention modules.
      2. Compute QSVD importance scores on calibration data.
      3. Globally allocate ranks.
      4. Apply the selected ranks to each Attention module.

    Returns the selected singular-value indices for each Attention module.
    """
    device = torch.device(device)
    model.to(device)

    # 1. Prepare full SVD info if not already present.
    attentions = collect_attention_modules(model)
    for attn in attentions:
        if attn.svd_info is None:
            attn.prepare_qsvd_svd_info(act_aware=False)

    # 2. Compute importance scores.
    print("Computing QSVD importance scores...")
    importance_scores = compute_qsvd_importance_scores(
        attentions, calibration_data, loss_fn, device=device
    )

    # 3. Global rank allocation.
    print("Allocating ranks globally...")
    selected_indices = allocate_ranks_global(
        importance_scores, target_param_ratio, rank_align=rank_align
    )

    # 4. Apply selected ranks.
    for attn in attentions:
        indices = selected_indices.get(attn, None)
        if indices is None or len(indices) == 0:
            indices = torch.arange(1, dtype=torch.long)
        attn.apply_qsvd(selected_indices=indices, sigma_fuse=sigma_fuse)
        print(f"Attention rank set to {attn.qsvd_rank}/{attn.dim}")

    return selected_indices


# ====================== Full Transformer stem (QSVD drop-in) ======================

class Transformer(nn.Module):
    """
    QSVD-aware drop-in replacement for models.Transformer64.Transformer.
    All architecture hyper-parameters and forward logic are identical to the
    original; only the StageBlock/Block/Attention classes are taken from this
    file so they support use_qsvd.
    """
    def __init__(self, args, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=[2,2,2,2],
                 num_heads=[16,8,4,2], mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, upsample=Upsample):
        super().__init__()
        self.args = args
        self.ch = embed_dim
        self.bottom_width = args.bottom_width
        self.embed_dim = embed_dim = args.gf_dim
        self.window_size = args.g_window_size
        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        act_layer = args.g_act
        num_heads = [int(i) for i in args.num_heads.split(",")]

        # QSVD options passed through args (with sensible defaults).
        use_qsvd = getattr(args, 'use_qsvd', False)
        qsvd_rank_ratio = getattr(args, 'qsvd_rank_ratio', 1.0)
        qsvd_sigma_fuse = getattr(args, 'qsvd_sigma_fuse', 'UV')
        qsvd_act_aware = getattr(args, 'qsvd_act_aware', False)
        qsvd_rank_align = getattr(args, 'qsvd_rank_align', 1)

        # Adaptive rank allocation options.
        adaptive_rank = getattr(args, 'qsvd_adaptive_rank', False)
        target_param_ratio = getattr(args, 'qsvd_target_param_ratio', 0.75)
        adaptive_rank_tau = getattr(args, 'qsvd_adaptive_rank_tau', 0.4)
        adaptive_rank_simple = getattr(args, 'qsvd_adaptive_rank_simple', True)

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width ** 2, embed_dim*2))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width * 2) ** 2, embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width * 4) ** 2, embed_dim//2))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, (self.bottom_width * 8) ** 2, embed_dim//4))

        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4
        ]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]

        stage_block_kwargs = dict(
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.window_size,
            use_qsvd=use_qsvd,
            qsvd_rank_ratio=qsvd_rank_ratio,
            sigma_fuse=qsvd_sigma_fuse,
            act_aware=qsvd_act_aware,
            rank_align=qsvd_rank_align,
            adaptive_rank=adaptive_rank,
            target_param_ratio=target_param_ratio,
            adaptive_rank_tau=adaptive_rank_tau,
            adaptive_rank_simple=adaptive_rank_simple,
        )

        self.blocks_1 = StageBlock(
            depth=depth[0],
            dim=embed_dim*2,
            num_heads=num_heads[0],
            **stage_block_kwargs,
        )

        self.blocks_2 = StageBlock(
            depth=depth[1],
            dim=embed_dim,
            num_heads=num_heads[1],
            **stage_block_kwargs,
        )

        self.blocks_3 = StageBlock(
            depth=depth[2],
            dim=embed_dim//2,
            num_heads=num_heads[2],
            **stage_block_kwargs,
        )

        self.blocks_4 = StageBlock(
            depth=depth[3],
            dim=embed_dim//4,
            num_heads=num_heads[3],
            **stage_block_kwargs,
        )

        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)

        if args.datarange == '01':
            rgbact = nn.Sigmoid()
            print('You choose sigmoid for range [0,1]')
        elif args.datarange == '-11':
            rgbact = nn.Tanh()
            print('You choose tanh for range [-1,1]')

        self.padding3 = (3 + (3 - 1) * (1 - 1) - 1) // 2
        self.padding7 = (7 + (7 - 1) * (1 - 1) - 1) // 2
        self.to_rgb = nn.Sequential(
            nn.ReflectionPad2d(self.padding3),
            nn.Conv2d((embed_dim*2) // (2**3), (embed_dim*2) // (2**3), 3, 1, 0),
            nn.ReflectionPad2d(self.padding7),
            nn.Conv2d((embed_dim*2) // (2**3), 1, 7, 1, 0),
            nn.Tanh()
        )

        self.apply(self._init_weights)
        print('apply init weith trunc_normal')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, gsfeatures, inirgb):
        features = []
        outputrgb = []

        # -------block 1----------
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[0], x], 1)
        x, C, H, W = cnn2tf(x)
        x = x + self.pos_embed[0].to(x.device)
        x = self.blocks_1(x)

        # -------block 2----------
        x, H, W = pixel_upsample(x, H, W)
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[1], x], 1)
        x, C, H, W = cnn2tf(x)
        x = x + self.pos_embed[1].to(x.device)

        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_2(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)

        # -------block 3----------
        x, H, W = pixel_upsample(x, H, W)
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[2], x], 1)
        x, C, H, W = cnn2tf(x)
        x = x + self.pos_embed[2].to(x.device)
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_3(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)

        # -------block 4----------
        x, H, W = pixel_upsample(x, H, W)
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[3], x], 1)
        x, C, H, W = cnn2tf(x)
        x = x + self.pos_embed[3].to(x.device)
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_4(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)
        rgb_64 = self.to_rgb(x.permute(0, 2, 1).view(-1, C, H, W)) + inirgb

        return rgb_64


print("QSVD_Transformer64.py loaded successfully.")
