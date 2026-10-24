#!/bin/bash
#SBATCH --job-name=train_product
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Train Ours-QKV+FFN only (structured FFN prune + FT30). No eval.
# Edit --partition / --gres / --time / --cpus-per-task for your cluster.
# Point datasets at YOUR paths before submit:
#   export COCO_DIR=/path/to/coco/unlabeled2017
#   export DATA_ROOT=/path/to/test_sets
#   sbatch scripts/train_product.sh --cs 1,4,10,25,50 --alpha 0.75 --ffn taylor --keep 0.5
#
#   bash scripts/train_product.sh
#   bash scripts/train_product.sh --cs 1,4,10,25,50 --alpha 0.75 --ffn taylor --keep 0.5

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

CS_LIST="${TABLE1_CS}"
ALPHAS="0.75"
FFN_LIST="taylor"
QKV_ROOT=""
QKV_CKPT=""
SCORE="ne"
RANDOM_SEED="12345"
OUT_OVERRIDE=""

usage() {
  cat <<'EOF'
scripts/train_product.sh — train Ours-QKV+FFN (prune + recover FT). No inference.

  --cs LIST        Sensing ratios (default 1,4,10,25,50)
  --alpha LIST     Parent QKV rank budget α (default 0.75)
  --ffn LIST       taylor | magnitude | random  (default taylor)
  --keep R         FFN keep ratio ρ (default 0.5)
  --score MODE     Parent QKV score tag (default ne)
  --qkv-root DIR   Parent train root (default $EXP_ROOT/qkv)
  --qkv-ckpt PATH  Single parent checkpoint (one CS)
  --seed N         Random FFN seed (default 12345)
  --out DIR        Run root (default $EXP_ROOT/product)
  --dry-run
  -h
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cs) CS_LIST="$(expand_cs "${2:-all}")"; shift 2 ;;
    --alpha) ALPHAS="${2:-0.75}"; shift 2 ;;
    --ffn) FFN_LIST="${2:-taylor}"; shift 2 ;;
    --keep) KEEP_RATIO="${2:-0.5}"; shift 2 ;;
    --score) SCORE="${2:-ne}"; shift 2 ;;
    --qkv-root) QKV_ROOT="${2:-}"; shift 2 ;;
    --qkv-ckpt) QKV_CKPT="${2:-}"; shift 2 ;;
    --seed) RANDOM_SEED="${2:-12345}"; shift 2 ;;
    --out) OUT_OVERRIDE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

OUT_ROOT="${OUT_OVERRIDE:-${EXP_ROOT}/product}"
QKV_ROOT="${QKV_ROOT:-${EXP_ROOT}/qkv}"
mkdir -p logs "${OUT_ROOT}"
check_set11
apply_dry_run

find_parent() {
  local cs="$1" alpha="$2"
  if [[ -n "${QKV_CKPT}" ]]; then
    echo "${QKV_CKPT}"
    return 0
  fi
  echo "$(qkv_ckpt "${QKV_ROOT}" "${cs}" "${SCORE}" "${alpha}")"
}

run_one() {
  local cs="$1" alpha="$2" ffn="$3"
  local parent tag_dir pruned ft_out idx
  parent="$(find_parent "${cs}" "${alpha}")"
  tag_dir="$(product_dir "${OUT_ROOT}" "${cs}" "${ffn}" "${KEEP_RATIO}" "${alpha}")"
  pruned="${tag_dir}/pruned"
  ft_out="${tag_dir}/finetuned"
  idx="${pruned}/prune_indices.pt"
  mkdir -p "${pruned}" "${ft_out}"

  echo "============================================================"
  echo "TRAIN QKV+FFN  CS=${cs}  α=${alpha}  ffn=${ffn}  keep=${KEEP_RATIO}"
  echo "parent=${parent}"
  echo "out=${tag_dir}"
  echo "============================================================"

  require_file "${parent}"
  require_dir "${COCO_DIR}"
  local prune_args=(
    --checkpoint "${parent}"
    --gen_model CSformer_QSVD_Shared
    --method "${ffn}"
    --keep_ratio "${KEEP_RATIO}"
    --out_dir "${pruned}"
    --cs_ratio "${cs}"
    --device cuda
  )
  if [[ "${ffn}" == "taylor" ]]; then
    prune_args+=(--data_path "${COCO_DIR}" --calib_batches "${CALIB_BATCHES_FFN}" --calib_batch_size 4)
  fi
  if [[ "${ffn}" == "random" ]]; then
    prune_args+=(--random_seed "${RANDOM_SEED}")
  fi
  run_cmd "${PYTHON}" scripts/prune_ffn_structured.py "${prune_args[@]}"

  require_file "${idx}"
  run_cmd "${PYTHON}" train_qsvd_shared.py \
    --load_path "${pruned}" \
    --ffn_prune_indices "${idx}" \
    --exp_name "${ft_out}" \
    --reset_start_epoch \
    --data_path "${COCO_DIR}" \
    --dataset coco \
    --gen_model CSformer_QSVD_Shared \
    --use_qsvd \
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
  for ffn in $(split_csv "${FFN_LIST}"); do
    for cs in $(split_csv "${CS_LIST}"); do
      run_one "${cs}" "${alpha}" "${ffn}"
    done
  done
done

echo "TRAIN QKV+FFN done. Checkpoints under ${OUT_ROOT}"
echo "Infer: bash scripts/infer_product.sh --ckpt-root ${OUT_ROOT}"
