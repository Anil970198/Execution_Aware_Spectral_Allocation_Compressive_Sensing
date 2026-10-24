# Paper jobs (WACV 3314)

Six launchers. Historical SLURM scripts stay in `right_scripts/` (do not delete).
Previous worker-wrapper ablation jobs: `repro/legacy/` (copy back to undo).

| Job | Train (writes checkpoints) | Infer (reads checkpoints) |
|-----|----------------------------|---------------------------|
| Ours-QKV | `repro/train_qkv.sh` | `repro/infer_qkv.sh` |
| Ours-QKV+FFN | `repro/train_product.sh` | `repro/infer_product.sh` |
| Ablations | `repro/train_ablation.sh` | `repro/infer_ablation.sh` |

Train never evals (COCO + Set11 val). Infer never trains.

Table 1 defaults: CS `1,4,10,25,50`, datasets Set11/Set14/BSD68/Urban100/Set5, α=`0.75`.
Ablations default CS `25`, `--jobs table2`, datasets Set11+Urban100.

```bash
bash repro/train_qkv.sh
bash repro/infer_qkv.sh --ckpt-root runs/paper/qkv

bash repro/train_product.sh --qkv-root runs/paper/qkv
bash repro/infer_product.sh --ckpt-root runs/paper/product

bash repro/train_ablation.sh --jobs table2 --cs 25 --alpha 0.75
bash repro/infer_ablation.sh --jobs table2 --ckpt-root runs/paper/ablation
bash repro/infer_ablation.sh --ckpt checkpoints/ablations/ne/checkpoint.pth --cs 25
```

`--jobs table2` = uniform, sigma, sigma_g, ne, matched_topr, classical_4e, rn.
FFN criterion (Taylor / Mag / Random) is `train_product.sh --ffn`, not the allocation ladder.

```bash
DRY_RUN=1 bash repro/train_ablation.sh --jobs uniform --cs 25
bash repro/infer_ablation.sh --datasets all --jobs ne --cs 25
```

Env: `COCO_DIR`, `DATA_ROOT`, `EXP_ROOT`, `SCRATCH`, `FT_MAX_EPOCH`. Set11 must be exactly 11 images.
