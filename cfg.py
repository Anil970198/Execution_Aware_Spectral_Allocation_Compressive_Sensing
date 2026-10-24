import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument(
        '-gen_bs',
        '--gen_batch_size',
        type=int,
        default=10,
        help='size of the batches')
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=200,
        help='number of epochs of training')
    parser.add_argument(
        '--max_iter',
        type=int,
        default=None,
        help='set the max iteration number')
    parser.add_argument(
        '--g_lr',
        type=float,
        default=0.0001,
        help='adam: gen learning rate')
    parser.add_argument(
        '--lr_decay',
        action='store_true',
        help='learning rate decay or not')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.0,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.9,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='number of cpu threads to use during batch generation')
    parser.add_argument(
        '--val_freq',
        type=int,
        default=1,
        help='interval between each validation')

    parser.add_argument(
        '--dataset',
        type=str,
        default='coco',
        help='dataset type')
    parser.add_argument(
        '--data_path',
        type=str,
        default="D:\database\package\coco\\unlabeled2017",
        help='The path of data set')
    parser.add_argument('--init_type', type=str, default='xavier_uniform',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='The init type')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='optimizer')
    parser.add_argument('--rec_w', type=int, default=1, help='penalty for the reconstruction loss')
    parser.add_argument('--rec_loss_type', type=str, default='l1', help='The type of reconstruction loss')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='penalty for the adam')
    parser.add_argument('--lr_multi', type=int, default=None)
    parser.add_argument('--train_patch_size',type=int,default=128,help='size of training input in dataloader')
    parser.add_argument('--random_seed', type=int, default=12345)

    # setting
    parser.add_argument(
        '--exp_name',
        type=str,
        help='The name of exp')
    parser.add_argument(
        '--load_path',
        type=str,
        help='The reload model path')
    parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')
    parser.add_argument('--datarange', type=str, default='-11',
                        help='input data norm to range')
    parser.add_argument('--torch_vision', action='store_true', default=False, help='Show intermediate results in tensorbard dir')
    parser.add_argument(
        '--print_freq',
        type=int,
        default=100,
        help='interval between each verbose')

    #test
    parser.add_argument(
        '--testdata_path',
        type=str,
        default=[r'C:\Users\HP\Documents\CSFormer\CSformer-main\data\Set11'],
        # default=['D:\database\dataset\\Urban100','C:\dataset\data\Set11','D:\database\dataset\BSD68','D:\database\dataset\Set14','D:\database\dataset\Set5\data'],
        help='The path of data set')
    parser.add_argument(
        '--valdata_path',
        type=str,
        default=r'C:\Users\HP\Documents\CSFormer\CSformer-main\data\Set11',
        help='The path of val set')
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--overlap',  action='store_true',help='overlap or not during testing')
    parser.add_argument('--overlapstep',type=int,default=8,help='the overlap step for testing')

    #model
    parser.add_argument(
        '--gen_model',
        type=str,
        default='CSformer',
        help='path of gen model')
    parser.add_argument(
        '--bottom_width',
        type=int,
        default=8)
    parser.add_argument(
        '--img_size',
        type=int,
        default=64,
        help='output size')
    parser.add_argument('--gf_dim', type=int, default=128,
                        help='The base channel num of model')

    parser.add_argument('--g_depth', type=str, default="5,5,5,5",
                        help='Generator Depth')
    parser.add_argument('--g_window_size', type=int, default=8,
                        help='generator mlp ratio')
    parser.add_argument('--num_heads', type=str, default="16,8,4,2",
                        help='num_head of transformer')
    parser.add_argument('--cnnnorm_type', type=str, default="BatchNorm",
                        help='norm type of cnn')
    parser.add_argument('--g_norm', type=str, default="ln",
                        help='Generator Normalization')
    parser.add_argument('--g_mlp', type=int, default=4,
                        help='generator mlp ratio')
    parser.add_argument('--g_act', type=str, default="gelu",
                        help='Generator activation Layer')
    parser.add_argument('--pretrained', type=str, default="church",
                        help='pretrained dataset')
    parser.add_argument('--seed', default=12345, type=int,
                        help='seed for initializing training. ')

    # QSVD options (used by models.CSformer_QSVD and models.QSVD_Transformer64)
    parser.add_argument('--use_qsvd', action='store_true', default=False,
                        help='Enable QSVD low-rank QKV compression in the transformer stem')
    parser.add_argument('--qsvd_rank_ratio', type=float, default=1.0,
                        help='QSVD parameter ratio (2*rank/dim); use 2.0 for full rank')
    parser.add_argument('--qsvd_sigma_fuse', type=str, default='UV',
                        help='QSVD singular-value split mode: UV, U, V, adaptive, or float in [0,1]')
    parser.add_argument('--qsvd_act_aware', action='store_true', default=False,
                        help='Use activation-aware SVD whitening')
    parser.add_argument('--qsvd_rank_align', type=int, default=1,
                        help='Align QSVD rank to a multiple of this value')
    parser.add_argument('--qsvd_discard_original', action='store_true', default=False,
                        help='Discard original full-rank qkv buffers before training '
                             '(saves memory when training CSformer_QSVD from scratch)')

    # Adaptive rank allocation options
    parser.add_argument('--qsvd_adaptive_rank', action='store_true', default=False,
                        help='Enable learnable adaptive rank allocation for QSVD')
    parser.add_argument('--qsvd_target_param_ratio', type=float, default=0.75,
                        help='Target QKV parameter ratio for adaptive rank allocation')
    parser.add_argument(
        '--qsvd_score_mode',
        type=str,
        default='ne',
        help='Knapsack importance density: ne (=σ|g|/(N·E), LOCKED default); '
             'classical_4e (σ|g|/(4E), failed ablation); '
             'sigma (|σ|); sigma_g (σ|g|, no denom).',
    )
    parser.add_argument('--qsvd_adaptive_rank_tau', type=float, default=0.4,
                        help='Gumbel-Sigmoid temperature for adaptive rank masks')
    parser.add_argument('--qsvd_adaptive_rank_lambda', type=float, default=1.0,
                        help='Weight for the adaptive rank compression loss')
    parser.add_argument('--qsvd_adaptive_rank_gamma', type=float, default=0.1,
                        help='Weight for the adaptive rank alignment loss')
    parser.add_argument('--qsvd_adaptive_rank_simple', action='store_true', default=True,
                        help='Use a simple learnable vector mask predictor '
                             '(else a small GRU hypernetwork)')
    parser.add_argument('--qsvd_adaptive_rank_epochs', type=int, default=-1,
                        help='Epochs after which to freeze rank masks '
                             '(-1 means train jointly until the end)')

    # SIB-ACS adaptive sampling options
    parser.add_argument('--use_sib_acs', action='store_true', default=False,
                        help='Enable SIB-ACS adaptive sampling front-end')
    parser.add_argument('--sib_acs_phi_size', type=int, default=16,
                        help='Block size for adaptive sampling (must match img_size factorization)')
    parser.add_argument('--sib_acs_num_stages', type=int, default=3,
                        help='Number of SIB-ACS sampling stages')
    parser.add_argument('--sib_acs_base_ratio', type=float, default=0.2,
                        help='Fraction of total sampling budget used for initial uniform sampling')
    parser.add_argument('--sib_acs_channels', type=int, default=32,
                        help='Channels in the SIB-ACS auxiliary reconstructor')
    parser.add_argument('--sib_acs_aux_blocks', type=int, default=4,
                        help='Legacy alias for aux depth; prefer --sib_acs_aux_layers')
    parser.add_argument('--sib_acs_aux_layers', type=int, default=4,
                        help='Number of mask-aware DC layers in the auxiliary reconstructor')
    parser.add_argument('--sib_acs_headroom', type=float, default=4.0,
                        help='Phi bank size as a multiple of the average CS budget '
                             '(hard blocks need headroom above n_input)')
    parser.add_argument('--sib_acs_max_measurements', type=int, default=-1,
                        help='Override Phi bank size (-1 = derive from headroom)')
    parser.add_argument('--sib_acs_freeze_backbone', action='store_true', default=False,
                        help='Deprecated alias: same as --sib_acs_train_mode sampler. '
                             'Train only sampler/aux (freeze CSFormer backbone).')
    parser.add_argument('--sib_acs_train_mode', type=str, default='joint',
                        choices=['sampler', 'sampler_iniconv', 'joint',
                                 'freeze_sampler', 'freeze_sensing'],
                        help='SIB trainable set: '
                             'sampler = only sampler.*; '
                             'sampler_iniconv = sampler.* + iniconv.*; '
                             'joint = train entire model (default); '
                             'freeze_sampler = freeze sampler.*, train rest; '
                             'freeze_sensing = freeze sampler.*+iniconv.*, '
                             'train recon/QKV (use after Shared QSVD attach).')
    parser.add_argument('--sib_acs_aux_loss_weight', type=float, default=0.1,
                        help='Weight for auxiliary recon L1 vs the input image '
                             '(0 disables). Helps ASA quality during training.')
    parser.add_argument('--reset_start_epoch', action='store_true', default=False,
                        help='After loading a checkpoint, start training from epoch 0 '
                             'instead of checkpoint[\"epoch\"]. Needed for curriculum '
                             'stages that each run their own max_epoch budget.')
    parser.add_argument('--use_factored_qk', action='store_true', default=False,
                        help='Way-2 shared QKV: factored QK^T scores via C @ G @ C^T '
                             '(default False = Way-1 q @ k^T).')
    parser.add_argument(
        '--ffn_prune_indices',
        type=str,
        default='',
        help='Path to prune_indices.pt for structured FFN prune. '
             'Required when loading a structurally pruned dense CSformer ckpt; '
             'applied after model build and before load_state_dict.',
    )
    parser.add_argument(
        '--use_tome',
        action='store_true',
        default=False,
        help='Enable Token Merging (ToMe) on shared/product CSformer stem '
             '(in-window merge between attn and MLP; unmerge before window_reverse).',
    )
    parser.add_argument(
        '--tome-r', '--tome_r',
        type=int,
        default=4,
        dest='tome_r',
        help='ToMe merges per block (per window). Fixed for batching. Start at 4.',
    )
    parser.add_argument(
        '--tome-prop-attn', '--tome_prop_attn',
        type=int,
        default=1,
        dest='tome_prop_attn',
        help='Proportional attention with token size (1=on, 0=off).',
    )

    opt = parser.parse_args(args)

    return opt
