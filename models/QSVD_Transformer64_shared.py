#QSVD_Transformer64_shared.py
"""
QSVD Transformer with paper-faithful SHARED down-projection (NEW FILE).

Does NOT modify QSVD_Transformer64.py.

Forward graph per Attention:
    C = down(X)           # shared, shape [B,N,r] = C_qkv
    q,k,v = up_*(C)       # three ups
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import models.QSVD_Transformer64 as base
from models.qsvd_shared_qkv import (
    SharedQKV,
    build_shared_qkv_from_svd,
    compute_rank_matching_prior_jobs,
)
from utils.tome_merge import bipartite_soft_matching, merge_wavg, parse_r

prepare_qsvd_svd_info = base.prepare_qsvd_svd_info
_split_qkv_weight = base._split_qkv_weight
gelu = base.gelu
CustomNorm = base.CustomNorm
Mlp = base.Mlp
DropPath = base.DropPath
matmul = base.matmul
trunc_normal_ = base.trunc_normal_
Upsample = base.Upsample
tf2cnn = base.tf2cnn
cnn2tf = base.cnn2tf
pixel_upsample = base.pixel_upsample
window_partition = base.window_partition
window_reverse = base.window_reverse


class SharedAttention(base.Attention):
    """Attention with one shared down + three ups (SharedQKV)."""

    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        want_qsvd = bool(kwargs.pop("use_qsvd", False))
        use_factored_qk = bool(kwargs.pop("use_factored_qk", False))
        kwargs["use_qsvd"] = False
        kwargs["adaptive_rank"] = False
        super().__init__(*args, **kwargs)
        self.shared_qkv: Optional[SharedQKV] = None
        self.use_qsvd = want_qsvd
        self.use_factored_qk = use_factored_qk
        self._qsvd_applied = False
        self.qsvd_rank = None
        if want_qsvd:
            self.apply_shared_qsvd(rank_ratio=self.qsvd_rank_ratio)

    def _relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        return relative_position_bias.permute(2, 0, 1).contiguous()

    def _factored_qk_scores(self, C: torch.Tensor) -> torch.Tensor:
        """Way-2: scores_h = C @ (Wq_h^T Wk_h) @ C^T (+ bias correction if needed)."""
        B, N, r = C.shape
        H = self.num_heads
        d_h = self.dim // H

        Wq = self.shared_qkv.up_q.weight.view(H, d_h, r)
        Wk = self.shared_qkv.up_k.weight.view(H, d_h, r)
        bq = self.shared_qkv.up_q.bias
        bk = self.shared_qkv.up_k.bias

        if bq is None and bk is None:
            G = torch.einsum("hdr,hds->hrs", Wq, Wk)
            scores = torch.einsum("bnr,hrs,bmr->bhnm", C, G, C)
        else:
            CWq = torch.einsum("bnr,hdr->bhnd", C, Wq)
            CWk = torch.einsum("bnr,hdr->bhnd", C, Wk)
            if bq is not None:
                CWq = CWq + bq.view(1, H, 1, d_h)
            if bk is not None:
                CWk = CWk + bk.view(1, H, 1, d_h)
            scores = torch.einsum("bhnd,bhmd->bhnm", CWq, CWk)
        return scores * self.scale

    def apply_shared_qsvd(
        self,
        rank_or_ratio: Optional[Union[int, float]] = None,
        rank_ratio: Optional[float] = None,
        selected_indices: Optional[Union[torch.Tensor, List[int], np.ndarray]] = None,
        sigma_fuse: Optional[str] = None,
        act_aware: Optional[bool] = None,
        scaling_diag_matrix: Optional[torch.Tensor] = None,
        rank_align: Optional[int] = None,
    ):
        if sigma_fuse is None:
            sigma_fuse = self.sigma_fuse
        if act_aware is None:
            act_aware = self.act_aware
        if rank_align is None:
            rank_align = self.rank_align

        if selected_indices is None or self.svd_info is None:
            self.prepare_qsvd_svd_info(
                act_aware=act_aware, scaling_diag_matrix=scaling_diag_matrix
            )

        _, _, _, bias_q, bias_k, bias_v = _split_qkv_weight(self.qkv)

        if selected_indices is not None:
            shared, rank, _ = build_shared_qkv_from_svd(
                self.svd_info,
                selected_indices,
                sigma_fuse=sigma_fuse,
                bias_q=bias_q,
                bias_k=bias_k,
                bias_v=bias_v,
            )
        elif isinstance(rank_or_ratio, int):
            shared, rank, _ = build_shared_qkv_from_svd(
                self.svd_info,
                rank_or_ratio,
                sigma_fuse=sigma_fuse,
                bias_q=bias_q,
                bias_k=bias_k,
                bias_v=bias_v,
            )
        else:
            ratio = (
                rank_or_ratio
                if rank_or_ratio is not None
                else (rank_ratio if rank_ratio is not None else self.qsvd_rank_ratio)
            )
            rank = compute_rank_matching_prior_jobs(self.dim, float(ratio), rank_align)
            shared, rank, _ = build_shared_qkv_from_svd(
                self.svd_info,
                rank,
                sigma_fuse=sigma_fuse,
                bias_q=bias_q,
                bias_k=bias_k,
                bias_v=bias_v,
            )

        device = self.qkv.weight.device
        dtype = self.qkv.weight.dtype
        self.shared_qkv = shared.to(device=device, dtype=dtype)
        self.qkv_q = None
        self.qkv_k = None
        self.qkv_v = None
        self.qsvd_rank = int(rank)
        self.use_qsvd = True
        self._qsvd_applied = True
        self.freeze_original_qkv()

    def apply_qsvd(self, *args, **kwargs):
        return self.apply_shared_qsvd(*args, **kwargs)

    def forward(
        self,
        x,
        use_original_qkv: bool = False,
        size: Optional[torch.Tensor] = None,
        return_metric: bool = False,
    ):
        B, N, C = x.shape
        assert C == self.dim, f"Expected dim {self.dim}, got {C}"
        metric_k = None

        if (not self.use_qsvd) or use_original_qkv or (not self._qsvd_applied):
            if self.qkv is None:
                raise RuntimeError("Original qkv discarded; cannot use full-rank path.")
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
            if return_metric:
                metric_k = k.mean(1)
        elif self.use_factored_qk:
            if self.shared_qkv is None:
                raise RuntimeError("shared_qkv missing; call apply_shared_qsvd first.")
            C_latent = self.shared_qkv.down(x)
            attn = self._factored_qk_scores(C_latent)
            H = self.num_heads
            d_h = C // H
            Wv = self.shared_qkv.up_v.weight.view(H, d_h, -1)
            v = torch.einsum("bnr,hdr->bhnd", C_latent, Wv)
            if self.shared_qkv.up_v.bias is not None:
                v = v + self.shared_qkv.up_v.bias.view(1, H, 1, d_h)
            # No explicit K; use latent C as ToMe metric (per-token).
            if return_metric:
                metric_k = C_latent
        else:
            if self.shared_qkv is None:
                raise RuntimeError("shared_qkv missing; call apply_shared_qsvd first.")
            q, k, v = self.shared_qkv(x, mask=None)
            q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
            if return_metric:
                metric_k = k.mean(1)

        # RPB only when token count matches the window grid (ToMe may reduce N).
        n_win = self.window_size * self.window_size
        if N == n_win:
            relative_position_bias = self._relative_position_bias()
            attn = attn + relative_position_bias.unsqueeze(0)

        # Proportional attention (ToMe): once tokens represent multiple patches.
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_metric:
            return x, metric_k
        return x


class SharedBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=gelu,
        norm_layer=nn.LayerNorm,
        window_size=16,
        use_qsvd=False,
        qsvd_rank_ratio=1.0,
        sigma_fuse="UV",
        act_aware=False,
        rank_align=1,
        use_factored_qk=False,
        **_ignored,
    ):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = SharedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            use_qsvd=use_qsvd,
            qsvd_rank_ratio=qsvd_rank_ratio,
            sigma_fuse=sigma_fuse,
            act_aware=act_aware,
            rank_align=rank_align,
            use_factored_qk=use_factored_qk,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        info = getattr(self, "_tome_info", None)
        use_tome = bool(info is not None and info.get("enabled"))

        if not use_tome:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        size = info.get("size") if info.get("prop_attn") else None
        x_attn, metric = self.attn(
            self.norm1(x), size=size, return_metric=True
        )
        x = x + self.drop_path(x_attn)

        r = 0
        if info.get("r_queue"):
            r = int(info["r_queue"].pop(0))
        if r > 0 and metric is not None:
            n_before = x.shape[1]
            merge, unmerge = bipartite_soft_matching(
                metric,
                r,
                class_token=False,
                distill_token=False,
            )
            # Push unmerge so stage can restore length before window_reverse.
            info.setdefault("unmerge_stack", []).append(unmerge)
            x, info["size"] = merge_wavg(merge, x, info.get("size"))
            if info.get("trace_tokens"):
                info.setdefault("token_log", []).append((n_before, int(x.shape[1]), r))

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SharedStageBlock(nn.Module):
    def __init__(self, depth, dim, num_heads, **kwargs):
        super().__init__()
        self.depth = depth
        self.block = nn.Sequential(
            *[SharedBlock(dim=dim, num_heads=num_heads, **kwargs) for _ in range(depth)]
        )

    def forward(self, x):
        info = getattr(self, "_tome_info", None)
        if info is None or not info.get("enabled"):
            return self.block(x)

        # Per-stage schedule + restore full window length before caller reshape.
        info["r_queue"] = parse_r(self.depth, info.get("r", 0))
        info["size"] = None
        info["unmerge_stack"] = []

        x = self.block(x)

        for unmerge in reversed(info.get("unmerge_stack") or []):
            x = unmerge(x)
        info["unmerge_stack"] = []
        info["size"] = None
        return x


class Transformer(nn.Module):
    """Drop-in stem using SharedAttention blocks."""

    def __init__(
        self,
        args,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=10,
        embed_dim=384,
        depth=None,
        num_heads=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
        upsample=Upsample,
    ):
        super().__init__()
        self.args = args
        self.bottom_width = args.bottom_width
        self.embed_dim = embed_dim = args.gf_dim
        self.ch = embed_dim
        self.window_size = args.g_window_size
        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        act_layer = args.g_act
        num_heads = [int(i) for i in args.num_heads.split(",")]

        use_qsvd = getattr(args, "use_qsvd", False)
        qsvd_rank_ratio = getattr(args, "qsvd_rank_ratio", 1.0)
        qsvd_sigma_fuse = getattr(args, "qsvd_sigma_fuse", "UV")
        qsvd_act_aware = getattr(args, "qsvd_act_aware", False)
        qsvd_rank_align = getattr(args, "qsvd_rank_align", 1)
        use_factored_qk = getattr(args, "use_factored_qk", False)

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width ** 2, embed_dim * 2))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width * 2) ** 2, embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width * 4) ** 2, embed_dim // 2))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, (self.bottom_width * 8) ** 2, embed_dim // 4))
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4,
        ]

        stage_kwargs = dict(
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
            use_factored_qk=use_factored_qk,
        )

        self.blocks_1 = SharedStageBlock(
            depth=depth[0], dim=embed_dim * 2, num_heads=num_heads[0], **stage_kwargs
        )
        self.blocks_2 = SharedStageBlock(
            depth=depth[1], dim=embed_dim, num_heads=num_heads[1], **stage_kwargs
        )
        self.blocks_3 = SharedStageBlock(
            depth=depth[2], dim=embed_dim // 2, num_heads=num_heads[2], **stage_kwargs
        )
        self.blocks_4 = SharedStageBlock(
            depth=depth[3], dim=embed_dim // 4, num_heads=num_heads[3], **stage_kwargs
        )

        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=0.02)

        if args.datarange == "01":
            print("You choose sigmoid for range [0,1]")
        else:
            print("You choose tanh for range [-1,1]")

        self.padding3 = (3 + (3 - 1) * (1 - 1) - 1) // 2
        self.padding7 = (7 + (7 - 1) * (1 - 1) - 1) // 2
        self.to_rgb = nn.Sequential(
            nn.ReflectionPad2d(self.padding3),
            nn.Conv2d((embed_dim * 2) // (2 ** 3), (embed_dim * 2) // (2 ** 3), 3, 1, 0),
            nn.ReflectionPad2d(self.padding7),
            nn.Conv2d((embed_dim * 2) // (2 ** 3), 1, 7, 1, 0),
            nn.Tanh(),
        )
        self.apply(self._init_weights)
        print("apply init weith trunc_normal")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, gsfeatures, inirgb):
        # Identical topology to QSVD_Transformer64.Transformer.forward
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[0], x], 1)
        x, C, H, W = cnn2tf(x)
        x = x + self.pos_embed[0].to(x.device)
        x = self.blocks_1(x)

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


def collect_shared_attention_modules(model: nn.Module) -> List[SharedAttention]:
    core = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
    return [m for m in core.modules() if isinstance(m, SharedAttention)]


def set_use_factored_qk_on_model(model: nn.Module, enabled: bool) -> int:
    """Toggle Way-2 factored QK scores on all SharedAttention layers."""
    n = 0
    for attn in collect_shared_attention_modules(model):
        attn.use_factored_qk = bool(enabled)
        n += 1
    return n


def discard_original_qkv_in_shared_model(model: nn.Module) -> int:
    n = 0
    for attn in collect_shared_attention_modules(model):
        if attn.qkv is not None:
            attn.qkv = None
            n += 1
    print(f"Discarded original qkv weights from {n} SharedAttention layers.")
    return n


def apply_shared_qsvd_to_model(
    model: nn.Module,
    rank_ratio: float = 0.75,
    sigma_fuse: str = "UV",
    act_aware: bool = False,
    rank_align: int = 1,
):
    for attn in collect_shared_attention_modules(model):
        attn.apply_shared_qsvd(
            rank_ratio=rank_ratio,
            sigma_fuse=sigma_fuse,
            act_aware=act_aware,
            rank_align=rank_align,
        )


print("QSVD_Transformer64_shared.py loaded successfully.")
