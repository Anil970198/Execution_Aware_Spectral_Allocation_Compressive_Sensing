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

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place test images under `data/` (see `data/README.md`). Set11 must contain **exactly 11** images.

## Checkpoints

Download the `v1.0` Release zips and unzip so you have:

```
checkpoints/csformer/CS{1,4,10,25,50}/checkpoint.pth
checkpoints/ours-qkv/CS{r}/checkpoint.pth
checkpoints/ours-qkv-ffn/CS{r}/checkpoint.pth
checkpoints/ours-qkv-ffn/CS{r}/prune_indices.pt
checkpoints/ablations/<name>/checkpoint.pth
```

Details: `checkpoints/MANIFEST.txt`.

## Table 1 (infer saved weights)

```bash
bash scripts/infer_qkv.sh --ckpt checkpoints/ours-qkv/CS25/checkpoint.pth --cs 25
bash scripts/infer_product.sh --ckpt checkpoints/ours-qkv-ffn/CS25/checkpoint.pth \
  --ffn-idx checkpoints/ours-qkv-ffn/CS25/prune_indices.pt --cs 25
```

Train (COCO unlabeled2017; set `COCO_DIR`):

```bash
bash scripts/train_qkv.sh --cs 1,4,10,25,50 --alpha 0.75
bash scripts/train_product.sh --cs 1,4,10,25,50 --alpha 0.75 --ffn taylor --keep 0.5
```

## Table 2 / ablations (CS 25)

```bash
bash scripts/infer_ablation.sh --ckpt-root checkpoints/ablations --jobs table2 --cs 25
bash scripts/train_ablation.sh --jobs table2 --cs 25 --alpha 0.75
```

`--jobs table2` = uniform, |σ|, R, Ours NE, matched top-r, R/(4E), R/N.  
`--datasets` defaults to Set11+Urban100 (`--datasets all` for five sets).  
FFN Mag/Random: `bash scripts/train_product.sh --cs 25 --ffn magnitude` or `--ffn random`.
