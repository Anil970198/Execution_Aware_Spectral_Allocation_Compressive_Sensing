from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import torch
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from functions import train, validate, load_params, copy_params
import datasets
from utils.utils import set_log_dir, save_checkpoint, create_logger
from tensorboardX import SummaryWriter
from loss.loss import get_loss_dict
import time
import importlib

# NEW FILE: fine-tune / train for CSformer_QSVD_Shared (1-down + 3-up).
# Based on train.py; does not modify train.py.
from utils.qsvd_shared_metrics import print_shared_qsvd_metrics
from utils.sib_acs_train import resolve_sib_train_mode, apply_sib_train_mode
from utils.report_efficiency import (
    build_efficiency_report,
    print_efficiency_report,
    save_efficiency_report,
)
from models.QSVD_Transformer64_shared import (
    discard_original_qkv_in_shared_model,
    collect_shared_attention_modules,
)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _set_n_input(args):
    ratio_dict = {1: 3, 4: 11, 10: 26, 25: 64, 30: 77, 40: 103, 50: 128}
    args.n_input = ratio_dict[args.cs_ratio]
    return args


def main():
    args = cfg.parse_args()
    cli_use_factored_qk = bool(getattr(args, 'use_factored_qk', False))
    torch.manual_seed(args.random_seed)

    # ------------------------------------------------------------------
    # Optionally resume from a checkpoint first, so architecture args can
    # be restored before the model is built. This is essential for QSVD
    # checkpoints where the exact low-rank shapes are stored in the file.
    # ------------------------------------------------------------------
    checkpoint = None
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint_best.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        if 'args' in checkpoint:
            saved_args = checkpoint['args']
            architecture_keys = [
                'gen_model', 'cs_ratio', 'gf_dim', 'g_depth', 'num_heads',
                'g_window_size', 'bottom_width', 'img_size', 'datarange',
                'use_qsvd', 'qsvd_rank_ratio', 'qsvd_sigma_fuse',
                'qsvd_act_aware', 'qsvd_rank_align',
                'qsvd_adaptive_rank', 'qsvd_target_param_ratio',
                'qsvd_adaptive_rank_tau', 'qsvd_adaptive_rank_simple',
                'use_factored_qk',
                'use_sib_acs', 'sib_acs_phi_size', 'sib_acs_num_stages',
                'sib_acs_base_ratio', 'sib_acs_channels', 'sib_acs_aux_blocks',
                'sib_acs_aux_layers', 'sib_acs_headroom', 'sib_acs_max_measurements',
            ]
            for k in architecture_keys:
                if k in saved_args:
                    setattr(args, k, saved_args[k])
            print(f"    Restored architecture args from checkpoint (gen_model={args.gen_model}, "
                  f"qsvd_rank_ratio={getattr(args, 'qsvd_rank_ratio', None)}, "
                  f"qsvd_adaptive_rank={getattr(args, 'qsvd_adaptive_rank', False)})")

        if cli_use_factored_qk:
            args.use_factored_qk = True

    # Get Sensing_matrix
    args = _set_n_input(args)

    # import network
    model_module = importlib.import_module('models.' + args.gen_model)
    model_class = getattr(model_module, args.gen_model)
    gen_net = model_class(args).to(device)
    print(f'model: {args.gen_model}')
    print(f'model param {(sum(param.numel() for param in  gen_net.parameters()))/1e6}M')
    print(f'dataset: {args.dataset}')
    print(f'cs ratio: {args.cs_ratio}')
    print(f'windows size: {args.g_window_size}')
    print(f'transformer depth: {[int(i) for i in args.g_depth.split(",")]}')
    print(f'transformer num_heads: {[int(i) for i in args.num_heads.split(",")]}')
    print(f'dim: {args.gf_dim}')
    print(f'CNN Norm: {args.cnnnorm_type}')
    print(f'Training patch size: {args.train_patch_size}')

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    gpu_ids = [i for i in range(int(torch.cuda.device_count()))]
    print('gpu ids:', gpu_ids)
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        gen_net = torch.nn.DataParallel(gen_net.to(device), device_ids=gpu_ids)
    else:
        gen_net = gen_net.to(device)


    def build_optimizer(net):
        """Build the Adam optimizer matching the project's args."""
        if args.lr_multi is not None:
            print('multi lr is not None')
            gs_params_id = list(map(id, net.module.gs.parameters()))
            decoder_params = filter(lambda p: id(p) not in gs_params_id, net.parameters())
            return torch.optim.Adam([{'params': net.module.gs.parameters(), 'lr': args.g_lr},
                                     {'params': decoder_params, 'lr': args.decoder_lr}],
                                    betas=(args.beta1, args.beta2))
        else:
            return torch.optim.Adam(net.parameters(), args.g_lr, betas=(args.beta1, args.beta2))

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    # epoch number
    args.max_epoch = args.max_epoch
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter / len(train_loader))


    # ------------------------------  Cosine  LR ------------------------------
    t_max = args.max_epoch
    print(f'max epoch is {t_max}')

    # initial
    start_epoch = 0
    best_psnr =10
    best_ssim =0
    psnr_score = 8
    best_epoch = 0

    # set up loss
    loss_all = get_loss_dict(args)
    print('loss:',loss_all.keys())
    print(f'rec_loss type: {args.rec_loss_type}\ng_lr：{args.g_lr}')

    # set writer
    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        if getattr(args, 'reset_start_epoch', False):
            print(
                f"    reset_start_epoch: ignoring checkpoint epoch "
                f"{start_epoch} -> start from 0 for this run "
                f"(max_epoch={args.max_epoch})"
            )
            start_epoch = 0

        # If the checkpoint stores exact per-layer ranks (e.g. global allocation),
        # re-apply QSVD with those ranks so the shapes match before loading.
        if args.gen_model == 'CSformer_QSVD_Shared' and args.use_qsvd and 'qsvd_ranks' in checkpoint:
            ranks = checkpoint['qsvd_ranks']
            attentions = collect_shared_attention_modules(gen_net)
            if len(ranks) == len(attentions):
                print(f"    Re-applying SHARED QSVD with saved ranks: {ranks}")
                for attn, rank in zip(attentions, ranks):
                    attn.apply_shared_qsvd(
                        rank_or_ratio=int(rank), sigma_fuse=args.qsvd_sigma_fuse
                    )

        # Structured FFN prune: reshape MLPs after ranks, before load_state_dict.
        from utils.ffn_prune import (
            apply_ffn_prune_from_indices,
            load_indices_file,
            unwrap_model,
        )
        idx_path = getattr(args, 'ffn_prune_indices', '') or ''
        ffn_indices = None
        if idx_path:
            if not os.path.isfile(idx_path):
                raise FileNotFoundError(
                    f"--ffn_prune_indices missing (fail loud): {idx_path}"
                )
            ffn_indices = load_indices_file(idx_path)
            print(f"    Applying FFN prune indices from {idx_path} ({len(ffn_indices)} MLPs)")
        elif checkpoint.get('ffn_prune_indices'):
            ffn_indices = {
                k: v.long() if torch.is_tensor(v) else torch.tensor(v).long()
                for k, v in checkpoint['ffn_prune_indices'].items()
            }
            print(f"    Applying FFN prune indices from checkpoint ({len(ffn_indices)} MLPs)")
        elif checkpoint.get('ffn_prune_method'):
            raise RuntimeError(
                "Pruned FFN checkpoint requires --ffn_prune_indices "
                "(or embedded ffn_prune_indices); refusing to load."
            )
        if ffn_indices is not None:
            apply_ffn_prune_from_indices(unwrap_model(gen_net), ffn_indices)
            print(
                f"    FFN-pruned params: "
                f"{(sum(p.numel() for p in gen_net.parameters()))/1e6:.4f}M"
            )

        state_dict = checkpoint['gen_state_dict']
        # thop profile leaves total_ops/total_params in some pruned saves
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.endswith('total_ops') and not k.endswith('total_params')
        }
        if isinstance(gen_net, nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not isinstance(gen_net, nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        sd_keys = list(state_dict.keys())
        has_shared = any('shared_qkv.' in k for k in sd_keys)
        has_fused_qkv = any('attn.qkv.weight' in k for k in sd_keys)
        discarded_qkv = bool(
            checkpoint.get('qsvd_discard_original', False)
            or args.qsvd_discard_original
            or (has_shared and not has_fused_qkv)
        )
        if discarded_qkv:
            args.qsvd_discard_original = True
            print("    Detected discarded original qkv buffers; loading with strict=False.")

        strict_load = not discarded_qkv
        missing, unexpected = gen_net.load_state_dict(state_dict, strict=strict_load)
        if not strict_load and missing:
            print(f'    WARNING: missing keys (expected for discarded qkv): {missing[:5]}')

        if discarded_qkv:
            discard_original_qkv_in_shared_model(gen_net)
            gen_optimizer = build_optimizer(gen_net)
            if checkpoint.get('ffn_prune_method') or getattr(args, 'ffn_prune_indices', ''):
                print("    FFN prune: using fresh Adam (skip optimizer restore).")
            elif 'gen_optimizer' in checkpoint:
                print("    WARNING: skipping optimizer restore after qkv discard; rebuilt.")
        else:
            gen_optimizer = build_optimizer(gen_net)
            if checkpoint.get('ffn_prune_method') or getattr(args, 'ffn_prune_indices', ''):
                print("    FFN prune: using fresh Adam (skip optimizer restore).")
            elif 'gen_optimizer' in checkpoint:
                try:
                    gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
                except (ValueError, KeyError) as exc:
                    print(f"    WARNING: could not restore optimizer state ({exc}); rebuilt.")
            else:
                print("    WARNING: checkpoint has no gen_optimizer; rebuilt.")

        # FT writes into exp_name from CLI; ignore nested path_helper from compress.
        if args.exp_name:
            args.path_helper = set_log_dir('logs', args.exp_name)
        else:
            args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        if args.qsvd_discard_original:
            discard_original_qkv_in_shared_model(gen_net)
        gen_optimizer = build_optimizer(gen_net)

    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=t_max, eta_min=1e-6)

    # SIB-ACS curriculum (same modes as train.py).
    if getattr(args, 'use_sib_acs', False):
        sib_mode = resolve_sib_train_mode(args)
        if sib_mode != 'joint':
            gen_optimizer, sib_sum = apply_sib_train_mode(
                gen_net, sib_mode, args.g_lr, args.beta1, args.beta2
            )
            print(
                f"SIB-ACS train_mode={sib_sum['mode']}: "
                f"{sib_sum['frozen_m']:.2f}M frozen, "
                f"{sib_sum['trainable_m']:.2f}M trainable "
                f"({sib_sum['num_trainable_tensors']} tensors)."
            )
            scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
                gen_optimizer, T_max=t_max, eta_min=1e-6
            )
        else:
            print("SIB-ACS train_mode=joint (full model trainable).")

    if args.gen_model == 'CSformer_QSVD_Shared' and args.use_qsvd:
        print_shared_qsvd_metrics(gen_net, input_size=(1, 1, 64, 64))

    try:
        ckpt_for_size = None
        if args.load_path:
            cand = os.path.join(args.load_path, 'Model', 'checkpoint_best.pth')
            if os.path.isfile(cand):
                ckpt_for_size = cand
        eff = build_efficiency_report(
            gen_net, args=args, ckpt_path=ckpt_for_size,
            tag=f"train_start_shared_{getattr(args, 'sib_acs_train_mode', 'joint')}",
        )
        print_efficiency_report(eff)
        save_efficiency_report(
            eff, os.path.join(args.path_helper['log_path'], 'efficiency_train_start.json')
        )
    except Exception as exc:
        print(f"WARNING: efficiency report failed: {exc}")

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # train loop
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch)), desc='total progress'):
        #lr
        cur_lr = gen_optimizer.param_groups[0]['lr']
        print(f'epoch[{epoch}] lr: {cur_lr}')
        writer = writer_dict['writer']
        writer.add_scalar('LR/g_lr', cur_lr, epoch)
        #lr

        start_time = time.time()
        train(args, gen_net,gen_optimizer,train_loader, epoch, writer_dict,
              loss_all)
        print(f'training time is {time.time()-start_time}')

        if epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1:

            backup_param = copy_params(gen_net)
            psnr_score,PSNR_cross, ssim_score,SSIM_cross = validate(args, epoch, gen_net, writer_dict)
            logger.info(f'@ epoch {epoch} || PSNR score: {psnr_score:.2f} SSIM score: {ssim_score:.4f}.\t @ best epoch {best_epoch} || Best PSNR score: {best_psnr:.2f} SSIM score: {best_ssim:.4f} .')
            load_params(gen_net, backup_param)

            if psnr_score > best_psnr:
                best_epoch = epoch
                best_psnr = psnr_score
                is_best = True
                best_ssim = ssim_score
                logger.info(f'@ epoch {epoch} || Best PSNR score: {psnr_score:.2f} SSIM score: {ssim_score:.4f}.')
            else:
                is_best = False
        else:
            is_best = False

        save_dict = {
            'epoch': epoch + 1,
            'gen_state_dict': gen_net.state_dict(),
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'path_helper': args.path_helper,
            'args': vars(args),
            'qsvd_discard_original': bool(getattr(args, 'qsvd_discard_original', False)),
        }
        if checkpoint is not None and 'qsvd_ranks' in checkpoint:
            save_dict['qsvd_ranks'] = checkpoint['qsvd_ranks']
            save_dict['qsvd_shared'] = True
        else:
            ranks = [
                getattr(attn, 'qsvd_rank', None)
                for attn in collect_shared_attention_modules(gen_net)
            ]
            if ranks and all(r is not None for r in ranks):
                save_dict['qsvd_ranks'] = ranks
                save_dict['qsvd_shared'] = True
        save_checkpoint(save_dict, is_best, args.path_helper['ckpt_path'])
        scheduler_lr.step()


if __name__ == '__main__':
    main()
