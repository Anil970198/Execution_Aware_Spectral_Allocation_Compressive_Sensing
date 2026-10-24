import torch
import torch.nn as nn
import torch.nn.functional as F
import models.QSVD_Transformer64 as QSVD_Transformer64
import models.CNN64 as CNN64
import models.sib_acs as sib_acs

import numpy as np


class CSformer_QSVD(nn.Module):
    """
    CSFormer variant that uses the QSVD-aware transformer stem from
    models.QSVD_Transformer64.

    Use this model exactly like the original CSformer, but set the QSVD options
    in args:
        args.use_qsvd          (bool)   default False
        args.qsvd_rank_ratio   (float)  default 1.0
        args.qsvd_sigma_fuse   (str)    default 'UV'
        args.qsvd_act_aware    (bool)   default False
        args.qsvd_rank_align   (int)    default 1

    Optionally enable SIB-ACS adaptive sampling via args.use_sib_acs.
    With SIB-ACS:
      - Phi bank has headroom above n_input so hard blocks can oversample.
      - iniconv accepts the wide measurement tensor (inactive rows are zero).
      - PhiTb residual uses the adjoint init from masked measurements, not
        the auxiliary reconstructor output (aux drives ASA only).
    """
    def __init__(self, args):
        super(CSformer_QSVD, self).__init__()

        self.args = args
        self.n_input = args.n_input
        self.bottom_width = args.bottom_width
        self.embed_dim = args.gf_dim
        self.outdim = int(np.ceil((args.img_size**2)//(args.bottom_width ** 2)))

        self.use_sib_acs = getattr(args, 'use_sib_acs', False)
        self.sib_acs_phi_size = getattr(args, 'sib_acs_phi_size', 16)
        sr = float(getattr(args, 'cs_ratio', 10)) / 100.0

        if self.use_sib_acs:
            override = getattr(args, 'sib_acs_max_measurements', None)
            if override is not None and int(override) <= 0:
                override = None
            self.sib_max_measurements = sib_acs.compute_max_measurements(
                phi_size=self.sib_acs_phi_size,
                n_input=self.n_input,
                sr=sr,
                headroom=getattr(args, 'sib_acs_headroom', 4.0),
                override=override,
            )
            meas_channels = self.sib_max_measurements
        else:
            self.sib_max_measurements = self.n_input
            meas_channels = self.n_input

        self.iniconv = nn.Sequential(
            nn.Conv2d(meas_channels, 128, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

        if self.use_sib_acs:
            self.sampler = sib_acs.SIBACSSampler(
                phi_size=self.sib_acs_phi_size,
                max_measurements_per_block=self.sib_max_measurements,
                n_input=self.n_input,
                num_stages=getattr(args, 'sib_acs_num_stages', 3),
                base_ratio=getattr(args, 'sib_acs_base_ratio', 0.2),
                channels=getattr(args, 'sib_acs_channels', 32),
                aux_blocks=getattr(args, 'sib_acs_aux_blocks', 4),
                aux_layers=getattr(args, 'sib_acs_aux_layers', None),
                headroom=getattr(args, 'sib_acs_headroom', 4.0),
                default_sr=sr,
            )
            # Dummy Phi/PhiT kept for checkpoint key compatibility with
            # baselines that always store these parameters.
            self.Phi = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(self.n_input, 256)))
            self.PhiT = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(256, self.n_input)))
        else:
            self.Phi = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(self.n_input, 256)))
            self.PhiT = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(256, self.n_input)))

        # transformer branch (QSVD-aware)
        self.td = QSVD_Transformer64.Transformer(args)

        # cnn branch
        self.gs = CNN64.Generator(args)

        # Optional last aux recon for training losses (set during SIB forward).
        self.last_aux_recon = None
        self.last_aux_target = None
        self.last_sib_stats = None

    def together(self, inputs, S, H, L):
        inputs = inputs.squeeze(1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=H*S, dim=0), dim=2)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=S, dim=0), dim=1)
        inputs = inputs.unsqueeze(1)
        return inputs

    def forward(self, inputs):
        H = int(inputs.shape[2]/64)
        L = int(inputs.shape[3]/64)
        S = inputs.shape[0]
        inputs = torch.squeeze(inputs, dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=64, dim=1), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=64, dim=2), dim=0)
        inputs = torch.unsqueeze(inputs, dim=1)

        np.random.seed(12345)
        if self.use_sib_acs:
            sr = float(getattr(self.args, 'cs_ratio', 10)) / 100.0
            # y: wide masked measurements; PhiTb: adjoint init (NOT aux).
            y, PhiTb, x_aux, _mask = self.sampler(inputs, sr)
            self.last_aux_recon = x_aux
            self.last_aux_target = inputs.detach()
            self.last_sib_stats = getattr(self.sampler, 'last_budget_stats', None)
        else:
            self.last_aux_recon = None
            self.last_aux_target = None
            self.last_sib_stats = None
            PhiWeight = self.Phi.contiguous().view(self.n_input, 1, 16, 16)
            y = F.conv2d(inputs, PhiWeight, padding=0, stride=16, bias=None)

            # Initialization-subnet
            PhiTWeight = self.PhiT.contiguous().view(256, self.n_input, 1, 1)
            PhiTb = F.conv2d(y, PhiTWeight, padding=0, bias=None)
            PhiTb = torch.nn.PixelShuffle(16)(PhiTb)

        x = self.iniconv(y)
        x = torch.nn.PixelShuffle(2)(x)

        x = x.flatten(2).transpose(1, 2).contiguous()
        gsfeatures = self.gs(x)
        output = self.td(x, gsfeatures, PhiTb)
        merge_output = self.together(output, S, H, L)
        merge_PhiTb = self.together(PhiTb, S, H, L)

        return merge_output, merge_PhiTb, output, PhiTb
