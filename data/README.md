Put test sets here **or** set `DATA_ROOT` to wherever they already live on your machine.

Expected names (case as below):

- `Set11` — exactly 11 images (remove `house (1).tif` if present)
- `Set14`
- `BSD68`
- `Urban100`
- `Set5`

Training / calibration is COCO unlabeled2017. That is **not** shipped. Set `COCO_DIR` to that folder on your cluster, for example:

```bash
export DATA_ROOT=/path/to/your/test_sets
export COCO_DIR=/path/to/coco/unlabeled2017
```

Do not assume the paper-lab paths. Change them to your setup before `sbatch` or `bash`.
