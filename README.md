# Execution-Aware Structured Compression for CSformer

Code for **WACV 2027** paper ID 3314:

**Execution-Aware Structured Compression of Progressive Hybrid Transformer Reconstructors for Compressive Sensing**

Backbone: [CSformer (Ye et al., TIP 2023)](https://ieeexplore.ieee.org/document/10124835/).

## Layout

```
scripts/          paper jobs (train + infer)
models/           CSformer and shared-QKV networks
utils/            metrics and helpers
data/             put Set11 / Set14 / Set5 / BSD68 / Urban100 here
checkpoints/      unzip Release weights here (see MANIFEST.txt)
```

Python entry points at the repo root (`train.py`, `eval.py`, `train_qsvd_shared.py`, `compress_checkpoint_shared_*.py`) are what those jobs call. Do not move them.

All paper commands live in `scripts/`:

| Table | Train | Infer |
|-------|-------|-------|
| Table 1 Ours-QKV | `scripts/train_qkv.sh` | `scripts/infer_qkv.sh` |
| Table 1 Ours-QKV+FFN | `scripts/train_product.sh` | `scripts/infer_product.sh` |
| Table 2 ablations | `scripts/train_ablation.sh` | `scripts/infer_ablation.sh` |

Train never evals. Infer never trains. `DRY_RUN=1` prints commands.

Each job script has `#SBATCH` headers (`--job-name`, `--partition`, `--gres`, `--time`, `--cpus-per-task`, `--nodes`, `--ntasks`, logs). **Edit those for your cluster** before `sbatch scripts/<job>.sh ...`.

## Paths (change these)

Test images and COCO are not in git. Point the jobs at **your** copies:

```bash
export DATA_ROOT=/path/to/your/test_sets          # Set11, Set14, Set5, BSD68, Urban100
export COCO_DIR=/path/to/coco/unlabeled2017       # train / calib only
```

If you instead drop the five test folders under `data/` in this repo, `DATA_ROOT` can stay at the default (`./data`). Set11 must contain **exactly 11** images.

## Cluster submit

```bash
mkdir -p logs
export DATA_ROOT=/path/to/your/test_sets
export COCO_DIR=/path/to/coco/unlabeled2017

sbatch scripts/train_qkv.sh --cs 1,4,10,25,50 --alpha 0.75
sbatch scripts/infer_qkv.sh --ckpt checkpoints/ours-qkv/CS25/checkpoint.pth --cs 25
sbatch scripts/train_product.sh --cs 1,4,10,25,50 --alpha 0.75 --ffn taylor --keep 0.5
sbatch scripts/infer_product.sh --ckpt checkpoints/ours-qkv-ffn/CS25/checkpoint.pth \
  --ffn-idx checkpoints/ours-qkv-ffn/CS25/prune_indices.pt --cs 25
sbatch scripts/train_ablation.sh --jobs table2 --cs 25 --alpha 0.75
sbatch scripts/infer_ablation.sh --ckpt-root checkpoints/ablations --jobs table2 --cs 25
```

`--partition=gpu` and `--gres=gpu:1` are placeholders. Replace them (and `--time`) with whatever your site uses.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

See **Paths** above and `data/README.md`.

## Checkpoints

Weights are **not** in git. Download the four zips from
[Releases](https://github.com/Anil970198/Execution_Aware_Spectral_Allocation_Compressive_Sensing/releases)
and unzip **into** `checkpoints/` (so the zip folders `csformer/`, `ours-qkv/`, … sit next to `MANIFEST.txt`):

```bash
unzip csformer-coco-cs1-50.zip -d checkpoints
unzip ours-qkv-cs1-50.zip -d checkpoints
unzip ours-qkv-ffn-cs1-50.zip -d checkpoints
unzip ablations-cs25.zip -d checkpoints
```

You should then have:

```
checkpoints/csformer/CS{1,4,10,25,50}/checkpoint.pth
checkpoints/ours-qkv/CS{r}/checkpoint.pth
checkpoints/ours-qkv-ffn/CS{r}/checkpoint.pth
checkpoints/ours-qkv-ffn/CS{r}/prune_indices.pt
checkpoints/ablations/<name>/checkpoint.pth
```

Details: `checkpoints/MANIFEST.txt`. Do not `git add` the `.pth` files.

## Table 1 (infer saved weights)

```bash
sbatch scripts/infer_qkv.sh --ckpt checkpoints/ours-qkv/CS25/checkpoint.pth --cs 25
sbatch scripts/infer_product.sh --ckpt checkpoints/ours-qkv-ffn/CS25/checkpoint.pth \
  --ffn-idx checkpoints/ours-qkv-ffn/CS25/prune_indices.pt --cs 25
```

Train (needs `COCO_DIR` on your machine):

```bash
sbatch scripts/train_qkv.sh --cs 1,4,10,25,50 --alpha 0.75
sbatch scripts/train_product.sh --cs 1,4,10,25,50 --alpha 0.75 --ffn taylor --keep 0.5
```

## Table 2 / ablations (CS 25)

```bash
sbatch scripts/infer_ablation.sh --ckpt-root checkpoints/ablations --jobs table2 --cs 25
sbatch scripts/train_ablation.sh --jobs table2 --cs 25 --alpha 0.75
```

`--jobs table2` = uniform, |σ|, R, Ours NE, matched top-r, R/(4E), R/N.  
`--datasets` defaults to Set11+Urban100 (`--datasets all` for five sets).  
FFN Mag/Random: `bash scripts/train_product.sh --cs 25 --ffn magnitude` or `--ffn random`.
