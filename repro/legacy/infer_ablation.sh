#!/bin/bash
# Infer allocation ablations from saved FT checkpoints. No training.
#
#   bash repro/infer_ablation.sh --jobs core
#   bash repro/infer_ablation.sh --ckpt-root runs/ablation --jobs ne --cs 25 --datasets set11,urban100

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

JOBS="core"
CS_LIST="25"
ALPHAS="0.75"
DATASETS="set11,urban100"
CKPT_ROOT=""
OUT_OVERRIDE=""

usage() {
  cat <<'EOF'
repro/infer_ablation.sh — overlap eval of saved ablation checkpoints (stride 8)

  --jobs LIST       Same ids as train_ablation.sh (default core)
  --ckpt-root DIR   Train EXP_ROOT (default $SCRATCH/ablation)
  --cs LIST         default 25
  --alpha LIST      default 0.75
  --datasets LIST   default set11,urban100
  --out DIR         Eval parent override
  --dry-run
  -h
EOF
}

expand_ablation_jobs() {
  local raw="$1" p
  local out=()
  IFS=',' read -ra parts <<< "${raw}"
  for p in "${parts[@]}"; do
    p="$(echo "${p}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    case "${p}" in
      ladder) out+=(uniform sigma sigma_g classical_4e ne) ;;
      all) out+=(uniform sigma sigma_g classical_4e ne matched_topr) ;;
      core) out+=(uniform sigma sigma_g ne matched_topr classical_4e) ;;
      lambda_sweep) out+=(ne_l025 ne_l05 ne_l075) ;;
      denom_split) out+=(rn re) ;;
      h100_extra) out+=(ne_l025 ne_l05 ne_l075 rn re) ;;
      "") ;;
      *) out+=("${p}") ;;
    esac
  done
  printf '%s\n' "${out[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="${2:-core}"; shift 2 ;;
    --ckpt-root) CKPT_ROOT="${2:-}"; shift 2 ;;
    --cs) CS_LIST="$(expand_cs "${2:-25}")"; shift 2 ;;
    --alpha) ALPHAS="${2:-0.75}"; shift 2 ;;
    --datasets) DATASETS="$(expand_datasets "${2:-set11,urban100}")"; shift 2 ;;
    --out) OUT_OVERRIDE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

CKPT_ROOT="${CKPT_ROOT:-${SCRATCH}/ablation}"
maybe_check_set11 "${DATASETS}"

mapfile -t JOB_LIST < <(expand_ablation_jobs "${JOBS}")

for default_alpha in $(split_csv "${ALPHAS}"); do
  for job in "${JOB_LIST[@]}"; do
    alpha="$(ablation_alpha "${job}" "${default_alpha}")"
    tag="$(ablation_tag "${job}" "${alpha}")"
    if [[ "${tag}" == "UNKNOWN" ]]; then
      echo "ERROR: unknown ablation job '${job}'"; exit 1
    fi
    for cs in $(split_csv "${CS_LIST}"); do
      ckpt="${CKPT_ROOT}/finetuned/CS${cs}_${tag}/Model/checkpoint_best.pth"
      run_dir="${CKPT_ROOT}/finetuned/CS${cs}_${tag}"
      eval_root="${OUT_OVERRIDE:-${run_dir}}"
      require_file "${ckpt}"
      echo "INFER ABLATION  job=${job}  CS=${cs}  α=${alpha}  ckpt=${ckpt}"
      for ds in $(split_csv "${DATASETS}"); do
        ddir="$(dataset_dir "${ds}")"
        tagname="$(eval_tag "${ds}")"
        require_dir "${ddir}"
        eval_shared "${ckpt}" "${eval_root}/eval_${tagname}" "${ddir}" "${cs}"
      done
    done
  done
done

echo "INFER ABLATION done."
