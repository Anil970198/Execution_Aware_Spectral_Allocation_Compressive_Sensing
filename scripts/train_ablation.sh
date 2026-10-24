#!/bin/bash
#SBATCH --job-name=train_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Train allocation ablations (compress + FT30). No eval.
# Edit --partition / --gres / --time / --cpus-per-task for your cluster.
# Point datasets at YOUR paths before submit:
#   export COCO_DIR=/path/to/coco/unlabeled2017
#   export DATA_ROOT=/path/to/test_sets
#   sbatch scripts/train_ablation.sh --jobs table2 --cs 25 --alpha 0.75
#
#   bash scripts/train_ablation.sh --jobs table2 --cs 25 --alpha 0.75
#   bash scripts/train_ablation.sh --jobs ne_a060,ne,ne_a090 --cs 25
#   DRY_RUN=1 bash scripts/train_ablation.sh --jobs uniform --cs 25

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

JOBS="table2"
CS_LIST="25"
ALPHAS="0.75"
OUT_OVERRIDE=""
RANKS_CKPT=""

usage() {
  cat <<'EOF'
scripts/train_ablation.sh — train score-ladder / matched-rank ablations. No inference.

  --jobs LIST     table2 | core | ladder | all | lambda_sweep | denom_split |
                  alpha_sweep | uniform,sigma,ne,...  (default table2)
  --cs LIST       default 25 (paper Table 2)
  --alpha LIST    Rank budget for jobs that do not bake α into the id (default 0.75)
  --out DIR       Run root (default $EXP_ROOT/ablation)
  --dense PATH    Dense CSformer ckpt (single CS)
  --ranks-ckpt PATH   For matched_topr: override NE compressed ckpt
  --dry-run
  --list
  -h

Job ids: uniform, sigma, sigma_g, classical_4e, ne, matched_topr,
         ne_a090, ne_a060, ne_l025, ne_l05, ne_l075, rn, re
table2 = uniform,sigma,sigma_g,ne,matched_topr,classical_4e,rn
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="${2:-table2}"; shift 2 ;;
    --cs) CS_LIST="$(expand_cs "${2:-25}")"; shift 2 ;;
    --alpha) ALPHAS="${2:-0.75}"; shift 2 ;;
    --out) OUT_OVERRIDE="${2:-}"; shift 2 ;;
    --dense) DENSE_CKPT="${2:-}"; shift 2 ;;
    --ranks-ckpt) RANKS_CKPT="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) usage; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

OUT_ROOT="${OUT_OVERRIDE:-${EXP_ROOT}/ablation}"
mkdir -p logs "${OUT_ROOT}"
check_set11
apply_dry_run

mapfile -t JOB_LIST < <(expand_ablation_jobs "${JOBS}")
if [[ "${#JOB_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: empty --jobs"; exit 1
fi

ne_compressed() {
  local cs="$1" alpha="$2"
  echo "$(ablation_dir "${OUT_ROOT}" "${cs}" "ne" "${alpha}")/compressed/Model/checkpoint_best.pth"
}

run_ft() {
  local compressed="$1" ft_out="$2" cs="$3"
  require_dir "${COCO_DIR}"
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

run_one() {
  local cs="$1" job="$2" alpha="$3"
  local tag_dir compressed ft_out dense mode ranks
  tag_dir="$(ablation_dir "${OUT_ROOT}" "${cs}" "${job}" "${alpha}")"
  compressed="${tag_dir}/compressed"
  ft_out="${tag_dir}/finetuned"
  dense="$(dense_ckpt "${cs}")"
  mkdir -p "${compressed}" "${ft_out}"
  mode="$(ablation_score_mode "${job}")"

  echo "============================================================"
  echo "TRAIN ABLATION  job=${job}  CS=${cs}  α=${alpha}"
  echo "dense=${dense}"
  echo "out=${tag_dir}"
  echo "============================================================"

  require_file "${dense}"

  case "${job}" in
    uniform)
      run_cmd "${PYTHON}" compress_checkpoint_shared_uniform.py \
        --load_path "${dense}" \
        --save_dir "${compressed}" \
        --cs_ratio "${cs}" \
        --qsvd_target_alpha_ratio "${alpha}" \
        --qsvd_sigma_fuse "${SIGMA_FUSE}" \
        "${ARCH[@]}" \
        2>&1 | tee "${compressed}/compression_stdout.log"
      ;;
    matched_topr)
      ranks="${RANKS_CKPT}"
      if [[ -z "${ranks}" ]]; then
        ranks="$(ne_compressed "${cs}" "${alpha}")"
      fi
      require_file "${ranks}"
      run_cmd "${PYTHON}" compress_checkpoint_shared_matched_topr.py \
        --load_path "${dense}" \
        --ranks_from_ckpt "${ranks}" \
        --save_dir "${compressed}" \
        --cs_ratio "${cs}" \
        --qsvd_target_param_ratio "${alpha}" \
        --qsvd_sigma_fuse "${SIGMA_FUSE}" \
        "${ARCH[@]}" \
        2>&1 | tee "${compressed}/compression_stdout.log"
      ;;
    sigma|sigma_g|classical_4e|ne|ne_a090|ne_a060|ne_l025|ne_l05|ne_l075|rn|re)
      require_dir "${COCO_DIR}"
      run_cmd "${PYTHON}" compress_checkpoint_shared_grad_info.py \
        --load_path "${dense}" \
        --save_dir "${compressed}" \
        --cs_ratio "${cs}" \
        --data_path "${COCO_DIR}" \
        --dataset coco \
        --qsvd_target_param_ratio "${alpha}" \
        --qsvd_grad_alpha "${GRAD_ALPHA}" \
        --qsvd_score_mode "${mode}" \
        --qsvd_calib_nsamples "${CALIB_NSAMPLES}" \
        --qsvd_calib_batch_size "${CALIB_BATCH}" \
        --qsvd_sigma_fuse "${SIGMA_FUSE}" \
        "${ARCH[@]}" \
        2>&1 | tee "${compressed}/compression_stdout.log"
      ;;
    *)
      echo "ERROR: unknown ablation job '${job}'"; exit 1
      ;;
  esac

  run_ft "${compressed}" "${ft_out}" "${cs}"
}

for job in "${JOB_LIST[@]}"; do
  case "${job}" in
    ne_a060|ne_a090)
      a="$(ablation_alpha "${job}" "0.75")"
      for cs in $(split_csv "${CS_LIST}"); do
        run_one "${cs}" "${job}" "${a}"
      done
      ;;
    *)
      for alpha in $(split_csv "${ALPHAS}"); do
        a="$(ablation_alpha "${job}" "${alpha}")"
        for cs in $(split_csv "${CS_LIST}"); do
          run_one "${cs}" "${job}" "${a}"
        done
      done
      ;;
  esac
done

echo "TRAIN ABLATION done. Checkpoints under ${OUT_ROOT}"
echo "Infer: bash scripts/infer_ablation.sh --ckpt-root ${OUT_ROOT} --jobs ${JOBS}"
