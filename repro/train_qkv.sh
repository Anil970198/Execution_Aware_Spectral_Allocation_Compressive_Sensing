#!/bin/bash
# Train Ours-QKV only (compress + FT30). No eval.
#
#   bash repro/train_qkv.sh
#   bash repro/train_qkv.sh --cs 1,4,10,25,50 --alpha 0.75 --score ne

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

CS_LIST="${TABLE1_CS}"
ALPHAS="0.75"
SCORE="ne"
OUT_OVERRIDE=""

usage() {
  cat <<'EOF'
repro/train_qkv.sh — train Ours-QKV (compress + recover FT). No inference.

  --cs LIST       Sensing ratios (default 1,4,10,25,50)
  --alpha LIST    Rank budget α (default 0.75). Example: 0.60,0.75,0.90
  --score MODE    ne | sigma | sigma_g | classical_4e | rn | re | ne_l025 | ...
  --out DIR       Run root (default $EXP_ROOT/qkv)
  --dense PATH    Dense CSformer ckpt (single CS only)
  --dry-run
  -h
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cs) CS_LIST="$(expand_cs "${2:-all}")"; shift 2 ;;
    --alpha) ALPHAS="${2:-0.75}"; shift 2 ;;
    --score) SCORE="${2:-ne}"; shift 2 ;;
    --out) OUT_OVERRIDE="${2:-}"; shift 2 ;;
    --dense) DENSE_CKPT="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

OUT_ROOT="${OUT_OVERRIDE:-${EXP_ROOT}/qkv}"
mkdir -p logs "${OUT_ROOT}"
check_set11
apply_dry_run

run_one() {
  local cs="$1" alpha="$2"
  local tag_dir compressed ft_out dense
  tag_dir="$(qkv_dir "${OUT_ROOT}" "${cs}" "${SCORE}" "${alpha}")"
  compressed="${tag_dir}/compressed"
  ft_out="${tag_dir}/finetuned"
  dense="$(dense_ckpt "${cs}")"
  mkdir -p "${compressed}" "${ft_out}"

  echo "============================================================"
  echo "TRAIN QKV  CS=${cs}  α=${alpha}  score=${SCORE}"
  echo "dense=${dense}"
  echo "out=${tag_dir}"
  echo "============================================================"

  require_file "${dense}"
  require_dir "${COCO_DIR}"
  run_cmd "${PYTHON}" compress_checkpoint_shared_grad_info.py \
    --load_path "${dense}" \
    --save_dir "${compressed}" \
    --cs_ratio "${cs}" \
    --data_path "${COCO_DIR}" \
    --dataset coco \
    --qsvd_target_param_ratio "${alpha}" \
    --qsvd_grad_alpha "${GRAD_ALPHA}" \
    --qsvd_score_mode "${SCORE}" \
    --qsvd_calib_nsamples "${CALIB_NSAMPLES}" \
    --qsvd_calib_batch_size "${CALIB_BATCH}" \
    --qsvd_sigma_fuse "${SIGMA_FUSE}" \
    "${ARCH[@]}" \
    2>&1 | tee "${compressed}/compression_stdout.log"

  run_cmd "${PYTHON}" train_qsvd_shared.py \
    --load_path "${compressed}" \
    --exp_name "${ft_out}" \
    --data_path "${COCO_DIR}" \
    --dataset coco \
    --gen_model CSformer_QSVD_Shared \
    --use_qsvd \
    --qsvd_sigma_fuse "${SIGMA_FUSE}" \
    --qsvd_discard_original \
    --cs_ratio "${cs}" \
    --max_epoch "${FT_MAX_EPOCH}" \
    --gen_batch_size "${FT_BATCH_SIZE}" \
    --g_lr "${FT_LR}" \
    --num_workers "${FT_NUM_WORKERS}" \
    --valdata_path "$(dataset_dir set11)" \
    --testdata_path "$(dataset_dir set11)" \
    "${ARCH[@]}" \
    --val_freq 1 \
    2>&1 | tee "${ft_out}/finetune_stdout.log"
}

for alpha in $(split_csv "${ALPHAS}"); do
  for cs in $(split_csv "${CS_LIST}"); do
    run_one "${cs}" "${alpha}"
  done
done

echo "TRAIN QKV done. Checkpoints under ${OUT_ROOT}"
echo "Infer: bash repro/infer_qkv.sh --ckpt-root ${OUT_ROOT}"
