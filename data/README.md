# Data (not in git, not in the Release)

Same rule as checkpoints: this repo does not ship images. Put **your** copies here, or point `DATA_ROOT` / `COCO_DIR` at folders you already have.

## Infer (test sets)

Jobs look for these folder names under `DATA_ROOT` (default: this `data/` directory):

| Folder | Images | Used in |
|--------|--------|---------|
| `Set11` | 11 | Table 1, Table 2 |
| `Set14` | 14 | Table 1 |
| `Set5` | 5 | Table 1 |
| `BSD68` | 68 | Table 1 |
| `Urban100` | 100 | Table 1, Table 2 |

Table 2 defaults to Set11 + Urban100. Table 1 infer uses all five (`--datasets all`).

These are standard CS / SR test sets. Use the copies you already evaluate with. Do not upload them to this GitHub repo.

```bash
export DATA_ROOT=/path/to/your/test_sets   # contains Set11, Set14, Set5, BSD68, Urban100
```

## Train / calibration (COCO)

Train jobs need **COCO unlabeled2017**. That set is large and is not shipped.

```bash
export COCO_DIR=/path/to/coco/unlabeled2017
```

Infer jobs do not need COCO.

## Layout example

```
data/Set11/*.tif
data/Set14/...
data/Set5/...
data/BSD68/...
data/Urban100/...
```

or keep them anywhere else and set `DATA_ROOT` to the parent of those five folders.
