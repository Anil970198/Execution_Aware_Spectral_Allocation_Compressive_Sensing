# Execution-Aware Structured Compression for CSformer

Code to reproduce **WACV 2027** paper ID 3314:

**Execution-Aware Structured Compression of Progressive Hybrid Transformer Reconstructors for Compressive Sensing**

This is a **paper archive**, not the full lab disk. Backbone: [CSformer (Ye et al., TIP 2023)](https://ieeexplore.ieee.org/document/10124835/).

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place test images under `data/` (see `data/README.md`). Set11 must contain **exactly 11** images.

## Checkpoints (GitHub Release, not git)

Download the `v1.0` Release zips and unzip so you have:

```
checkpoints/csformer/CS{1,4,10,25,50}/checkpoint.pth
checkpoints/ours-qkv/CS{r}/checkpoint.pth
checkpoints/ours-qkv-ffn/CS{r}/checkpoint.pth
checkpoints/ours-qkv-ffn/CS{r}/prune_indices.pt
checkpoints/ablations/<name>/checkpoint.pth
```

See `checkpoints/MANIFEST.txt`.

## Reproduce Table 1 (infer saved weights)

```bash
bash repro/infer_qkv.sh --ckpt checkpoints/ours-qkv/CS25/checkpoint.pth --cs 25
bash repro/infer_product.sh --ckpt checkpoints/ours-qkv-ffn/CS25/checkpoint.pth \
  --ffn-idx checkpoints/ours-qkv-ffn/CS25/prune_indices.pt --cs 25
```

Train (COCO unlabeled2017; needs `COCO_DIR`):

```bash
bash repro/train_qkv.sh --cs 1,4,10,25,50 --alpha 0.75
bash repro/train_product.sh --cs 1,4,10,25,50 --alpha 0.75 --ffn taylor --keep 0.5
```

## Reproduce Table 2 / supp ablations (CS 25)

```bash
bash repro/infer_ablation.sh --ckpt-root checkpoints/ablations --jobs table2 --cs 25
bash repro/train_ablation.sh --jobs table2 --cs 25 --alpha 0.75
```

`--jobs table2` = uniform, |σ|, R, Ours NE, matched top-r, R/(4E), R/N.  
`--datasets` defaults to Set11+Urban100 (`--datasets all` for five sets).  
FFN Mag/Random: `repro/train_product.sh --cs 25 --ffn magnitude|random`.

Train never evals; infer never trains. `DRY_RUN=1` prints commands.

## Citation

Please cite this paper (camera-ready author list) and CSformer (Ye et al., TIP 2023). See `CITATION.cff`.

## License

MIT for new compression / repro code. CSformer files remain Ye et al.'s. See `LICENSE`.

Collaborators: `GITHUB.md`.
