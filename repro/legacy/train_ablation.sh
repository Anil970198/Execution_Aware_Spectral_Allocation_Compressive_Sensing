#!/bin/bash
# Train allocation ablations (compress + FT30). No eval.
# Same Python as right_scripts/run_ablation_clean_local.sh
#
#   bash repro/train_ablation.sh --jobs core
#   bash repro/train_ablation.sh --jobs ne,uniform --cs 25 --alpha 0.75

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

WORKER="${PROJECT_DIR}/run_ablation_clean_local.sh"
if [[ ! -f "${WORKER}" ]]; then
  WORKER="${PROJECT_DIR}/right_scripts/run_ablation_clean_local.sh"
fi
if [[ ! -f "${WORKER}" ]]; then
  echo "ERROR: missing run_ablation_clean_local.sh"; exit 1
fi

JOBS="core"
CS_LIST="25"
ALPHAS="0.75"
OUT_OVERRIDE=""

usage() {
  cat <<'EOF'
repro/train_ablation.sh — train score-ladder / matched-rank ablations. No inference.

  --jobs LIST     core | ladder | all | uniform,sigma,ne,...  (default core)
  --cs LIST       default 25 (paper Table 2). Can pass 1,4,10,25,50
  --alpha LIST    Rank budget for jobs that do not bake α into the id (default 0.75)
  --out DIR       EXP_ROOT (default $SCRATCH/ablation)
  --dry-run
  --list          Print worker job ids
  -h

Job ids (Table 2 / supp): uniform, sigma, sigma_g, classical_4e, ne,
matched_topr, ne_a090, ne_a060, ne_l025, ne_l05, ne_l075, rn, re.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="${2:-core}"; shift 2 ;;
    --cs) CS_LIST="$(expand_cs "${2:-25}")"; shift 2 ;;
    --alpha) ALPHAS="${2:-0.75}"; shift 2 ;;
    --out) OUT_OVERRIDE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) exec bash "${WORKER}" --list ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

OUT_ROOT="${OUT_OVERRIDE:-${SCRATCH}/ablation}"
mkdir -p logs "${OUT_ROOT}"
check_set11

export PROJECT_DIR DATA_ROOT COCO_DIR
export SET11_DIR URBAN100_DIR
SET11_DIR="$(dataset_dir set11)"
URBAN100_DIR="$(dataset_dir urban100)"

for alpha in $(split_csv "${ALPHAS}"); do
  for cs in $(split_csv "${CS_LIST}"); do
    export CS_RATIO="${cs}"
    export TARGET_ALPHA="${alpha}"
    export BASE_CKPT="$(dense_ckpt "${cs}")"
    export EXP_ROOT="${OUT_ROOT}"
    echo "============================================================"
    echo "TRAIN ABLATION  jobs=${JOBS}  CS=${cs}  α=${alpha}"
    echo "EXP_ROOT=${EXP_ROOT}  BASE_CKPT=${BASE_CKPT}"
    echo "============================================================"
    echo "+ bash ${WORKER} --jobs ${JOBS} --alpha ${alpha} --skip-eval"
    if [[ "${DRY_RUN}" != "1" ]]; then
      bash "${WORKER}" --jobs "${JOBS}" --alpha "${alpha}" --skip-eval
    fi
  done
done

echo "TRAIN ABLATION done. Checkpoints under ${OUT_ROOT}"
echo "Infer: bash repro/infer_ablation.sh --ckpt-root ${OUT_ROOT} --jobs ${JOBS}"
