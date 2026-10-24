import cfg
import torch
import importlib
from functions import test_overlap, test
from utils.utils import set_log_dir, create_logger
# NEW FILE: eval for CSformer_QSVD_Shared
from utils.qsvd_shared_metrics import print_shared_qsvd_metrics
import os
from tensorboardX import SummaryWriter


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
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
    assert args.load_path.endswith('.pth')
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir('logs_eval', args.exp_name)
    logger = create_logger(args.path_helper['log_path'], phase='test')

    logger.info(f'=> resuming from {args.load_path}')
    print(f'=> resuming from {args.load_path}')
    checkpoint_file = args.load_path
    checkpoint = torch.load(checkpoint_file, map_location=device)

    # Compressed QSVD checkpoints store the exact args used to create them.
    # Use those for architecture so the low-rank shapes match the state dict.
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
              f"qsvd_adaptive_rank={getattr(args, 'qsvd_adaptive_rank', False)}, "
              f"use_sib_acs={getattr(args, 'use_sib_acs', False)})")

    if cli_use_factored_qk:
        args.use_factored_qk = True

    args = _set_n_input(args)

    # Import model dynamically so both CSformer and CSformer_QSVD work.
    model_module = importlib.import_module('models.' + args.gen_model)
    model_class = getattr(model_module, args.gen_model)
    gen_net = model_class(args).to(device)

    print(f'gen_model: {args.gen_model}')
    print(f'Model params: {sum(param.numel() for param in gen_net.parameters())}')
    print(f'dataset: {args.dataset}')
    print(f'cs ratio: {args.cs_ratio}')

    # If the checkpoint stores exact per-layer ranks (e.g. global allocation),
    # re-apply QSVD with those ranks so the shapes match before loading.
    if args.gen_model == 'CSformer_QSVD_Shared' and args.use_qsvd and 'qsvd_ranks' in checkpoint:
        from models.QSVD_Transformer64_shared import collect_shared_attention_modules
        ranks = checkpoint['qsvd_ranks']
        attentions = collect_shared_attention_modules(gen_net)
        if len(ranks) == len(attentions):
            print(f"    Re-applying SHARED QSVD with saved ranks: {ranks}")
            for attn, rank in zip(attentions, ranks):
                attn.apply_shared_qsvd(
                    rank_or_ratio=int(rank), sigma_fuse=args.qsvd_sigma_fuse
                )

    # Remove 'module.' prefix if the model was saved with DataParallel
    state_dict = checkpoint.get('gen_state_dict') or checkpoint.get('avg_gen_state_dict')

    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]   # remove 'module.'
            new_state_dict[name] = v
        state_dict = new_state_dict

    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
    }
    writer = writer_dict['writer']

    # If the checkpoint was created with discarded original qkv weights,
    # use strict=False and then remove the unused qkv buffers. Also infer
    # discard from the state dict (same fix as train.py) so older saves work.
    sd_keys = list(state_dict.keys())
    has_shared = any('shared_qkv.' in k for k in sd_keys)
    has_fused_qkv = any('attn.qkv.weight' in k for k in sd_keys)
    discarded_qkv = bool(
        checkpoint.get('qsvd_discard_original', False)
        or (has_shared and not has_fused_qkv)
    )
    if discarded_qkv:
        print("    Detected discarded original qkv buffers; loading with strict=False.")
    strict_load = not discarded_qkv
    missing, unexpected = gen_net.load_state_dict(state_dict, strict=strict_load)
    if not strict_load and missing:
        print(f'    WARNING: missing keys (expected for discarded qkv): {missing[:5]}')
    if discarded_qkv:
        from models.QSVD_Transformer64_shared import discard_original_qkv_in_shared_model
        discard_original_qkv_in_shared_model(gen_net)

    if args.gen_model == 'CSformer_QSVD_Shared' and getattr(args, 'use_factored_qk', False):
        from models.QSVD_Transformer64_shared import set_use_factored_qk_on_model
        n = set_use_factored_qk_on_model(gen_net, True)
        print(f"    Way-2 factored QK enabled on {n} SharedAttention layers")

    if args.gen_model == 'CSformer_QSVD_Shared' and args.use_qsvd:
        print_shared_qsvd_metrics(gen_net, input_size=(1, 1, 64, 64))

    epoch = checkpoint.get('epoch', 0)
    logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
    print(f'=> loaded checkpoint {checkpoint_file}')

    if args.overlap == True:
        test_overlap(args, gen_net, logger)
    else:
        test(args, gen_net, logger)


if __name__ == '__main__':
    main()
