#qsvd_shared_qkv.py
"""
Paper-faithful joint QSVD compute graph for CSFormer (NEW FILE).

Theory (QSVD comparisons table / §joint QKV):
  Joint SVD on concatenated [Wq; Wk; Wv] yields shared V and split Uq,Uk,Uv.
  Forward MUST be:
      C  = X @ B_shared     # one shared down-projection, C shape [..., r]
      Q  = C @ Aq           # three up-projections
      K  = C @ Ak
      V  = C @ Av
  Unique weight size α = 4 r E   (1× down E×r + 3× up r×E)
  Compute cost     γ = 4 L r E   (paper; proportional to α × sequence)
  Buffer (η)       η = r L       (store C_qkv only, not full K and V)

This module stores ONE down Linear and THREE up Linears (no duplicated B).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def _fuse_uv(
    U: torch.Tensor,
    S: torch.Tensor,
    V: torch.Tensor,
    sigma_fuse: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (A_weight [out, r], B_weight [r, in]) matching SVDLinear UV fuse.
    U: [out, r], S: [r], V: [in, r]
    """
    if sigma_fuse == "UV":
        A = U.mul(S.sqrt()).contiguous()
        B = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
    elif sigma_fuse == "U":
        A = U.mul(S).contiguous()
        B = V.t().contiguous()
    elif sigma_fuse == "V":
        A = U.contiguous()
        B = V.t().mul(S.view(-1, 1)).contiguous()
    elif isinstance(sigma_fuse, (int, float)) and 0.0 <= float(sigma_fuse) <= 1.0:
        beta = float(sigma_fuse)
        A = U.mul(S ** (1.0 - beta)).contiguous()
        B = V.t().mul((S ** beta).view(-1, 1)).contiguous()
    else:
        raise ValueError(f"Unsupported sigma_fuse for SharedQKV: {sigma_fuse}")
    return A, B


class SharedQKV(nn.Module):
    """
    One shared down-projection + three up-projections.

      C = down(X)          # [B, N, r]  = C_qkv
      q, k, v = up_*(C)

    Optional inference fuse (same math, fewer GEMMs):
      C   = down(X)                 # E -> r
      QKV = up_fused(C)             # r -> 3E
      q,k,v = split(QKV)
    Train / checkpoints keep separate up_q/up_k/up_v; call enable_fused_up()
    after load for timing. Way-2 still reads separate ups.
    """

    def __init__(
        self,
        dim: int,
        rank: int,
        bias_q: Optional[torch.Tensor] = None,
        bias_k: Optional[torch.Tensor] = None,
        bias_v: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        # down: E -> r   weight [r, E]
        self.down = nn.Linear(dim, rank, bias=False)
        # ups:  r -> E   weight [E, r]
        self.up_q = nn.Linear(rank, dim, bias=bias_q is not None)
        self.up_k = nn.Linear(rank, dim, bias=bias_k is not None)
        self.up_v = nn.Linear(rank, dim, bias=bias_v is not None)
        if bias_q is not None:
            self.up_q.bias.data.copy_(bias_q)
        if bias_k is not None:
            self.up_k.bias.data.copy_(bias_k)
        if bias_v is not None:
            self.up_v.bias.data.copy_(bias_v)
        self.up_fused: Optional[nn.Linear] = None
        self._use_fused_up = False

    @property
    def last_C(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_C", None)

    def _has_any_up_bias(self) -> bool:
        return any(u.bias is not None for u in (self.up_q, self.up_k, self.up_v))

    def sync_fused_from_ups(self) -> None:
        """Copy concat(up_q, up_k, up_v) into up_fused (same math as three ups)."""
        if self.up_fused is None:
            raise RuntimeError("up_fused missing; call enable_fused_up() first")
        with torch.no_grad():
            W = torch.cat(
                [self.up_q.weight.data, self.up_k.weight.data, self.up_v.weight.data],
                dim=0,
            )
            self.up_fused.weight.copy_(W)
            if self.up_fused.bias is not None:
                parts = []
                for up in (self.up_q, self.up_k, self.up_v):
                    if up.bias is not None:
                        parts.append(up.bias.data)
                    else:
                        parts.append(torch.zeros(self.dim, device=W.device, dtype=W.dtype))
                self.up_fused.bias.copy_(torch.cat(parts, dim=0))

    def enable_fused_up(self) -> None:
        """Build / refresh Linear(r -> 3E) from current separate ups (infer only)."""
        need_bias = self._has_any_up_bias()
        if self.up_fused is None or (self.up_fused.bias is None) != (not need_bias):
            self.up_fused = nn.Linear(self.rank, 3 * self.dim, bias=need_bias)
            self.up_fused = self.up_fused.to(
                device=self.down.weight.device, dtype=self.down.weight.dtype
            )
        self.sync_fused_from_ups()
        self._use_fused_up = True

    def disable_fused_up(self) -> None:
        self._use_fused_up = False

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, N, E]
            mask: optional [r] mask over latent (adaptive ranks); multiplies C
        Returns:
            q, k, v each [B, N, E]; also caches C on self._last_C for η diagnostics
        """
        C = self.down(x)
        if mask is not None:
            C = C * mask
        self._last_C = C
        if self._use_fused_up and self.up_fused is not None:
            qkv = self.up_fused(C)
            q, k, v = qkv.split(self.dim, dim=-1)
            return q, k, v
        q = self.up_q(C)
        k = self.up_k(C)
        v = self.up_v(C)
        return q, k, v

    def num_unique_params(self) -> int:
        """α contribution: 4 r E (+ biases if any). Count separate ups (not fused)."""
        n = self.down.weight.numel()
        n += self.up_q.weight.numel() + self.up_k.weight.numel() + self.up_v.weight.numel()
        for lin in (self.up_q, self.up_k, self.up_v):
            if lin.bias is not None:
                n += lin.bias.numel()
        return n


def set_fused_up_on_model(model: nn.Module, enabled: bool) -> int:
    """Enable/disable 2-GEMM fused-up path on every SharedQKV (infer-time)."""
    n = 0
    for m in model.modules():
        if isinstance(m, SharedQKV):
            if enabled:
                m.enable_fused_up()
            else:
                m.disable_fused_up()
            n += 1
    return n


def build_shared_qkv_from_svd(
    svd_info: Dict[str, torch.Tensor],
    rank_or_indices: Union[int, torch.Tensor, List[int], np.ndarray],
    sigma_fuse: str = "UV",
    bias_q: Optional[torch.Tensor] = None,
    bias_k: Optional[torch.Tensor] = None,
    bias_v: Optional[torch.Tensor] = None,
) -> Tuple[SharedQKV, int, torch.Tensor]:
    """
    Build SharedQKV from joint SVD factors (same svd_info as prepare_qsvd_svd_info).
    """
    device = svd_info["U_full"].device
    U_full = svd_info["U_full"].to(device).float()
    S_full = svd_info["S_full"].to(device).float()
    V_full = svd_info["V_full"].to(device).float()
    dim = int(svd_info["dim"])

    if isinstance(rank_or_indices, int):
        rank = max(1, min(rank_or_indices, S_full.numel()))
        selected_indices = torch.arange(rank, device=device)
    else:
        selected_indices = torch.as_tensor(rank_or_indices, device=device, dtype=torch.long)
        selected_indices = torch.unique(selected_indices).sort().values
        rank = int(selected_indices.numel())

    U = U_full[:, selected_indices]
    S = S_full[selected_indices]
    V = V_full[:, selected_indices]

    if svd_info.get("scaling_diag_matrix") is not None:
        scaling = svd_info["scaling_diag_matrix"].to(device).float()
        V = V / scaling.view(-1, 1)

    U_q = U[:dim, :]
    U_k = U[dim : 2 * dim, :]
    U_v = U[2 * dim :, :]

    # Shared B from V,S; three A's from Uq/Uk/Uv,S.
    # Under UV fuse, A and B both absorb sqrt(S); one shared B is correct.
    Aq, B = _fuse_uv(U_q, S, V, sigma_fuse)
    Ak, B_k = _fuse_uv(U_k, S, V, sigma_fuse)
    Av, B_v = _fuse_uv(U_v, S, V, sigma_fuse)
    # B from any arm must match (same V,S). Use first; assert close in smoke tests.
    _ = B_k, B_v

    module = SharedQKV(dim, rank, bias_q=bias_q, bias_k=bias_k, bias_v=bias_v)
    with torch.no_grad():
        module.down.weight.copy_(B)
        module.up_q.weight.copy_(Aq)
        module.up_k.weight.copy_(Ak)
        module.up_v.weight.copy_(Av)

    return module, rank, selected_indices.cpu()


def compute_rank_matching_prior_jobs(dim: int, rank_ratio: float, rank_align: int = 1) -> int:
    """
    Same rank formula as models.QSVD_Transformer64._compute_rank (prior Job 0.75).

    With shared α=4rE this yields paper α_ratio = 4r/(3E) = 2*(rank_ratio)/3
    when rank = dim*rank_ratio/2 (for rank_ratio=0.75 → α_ratio=0.5).
    """
    from models.QSVD_Transformer64 import _compute_rank
    return _compute_rank(dim, rank_ratio, rank_align=rank_align)
