#!/bin/bash
# Infer allocation ablations from saved FT checkpoints. No training.
#
#   bash scripts/infer_ablation.sh --jobs table2 --cs 25 --datasets set11,urban100
#   bash scripts/infer_ablation.sh --ckpt checkpoints/ablations/ne/checkpoint.pth --cs 25
#   bash scripts/infer_ablation.sh --ckpt-root checkpoints/ablations --jobs uniform,sigma

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

JOBS="table2"
CS_LIST="25"
ALPHAS="0.75"
DATASETS="set11,urban100"
CKPT_ROOT=""
CKPT=""
OUT_OVERRIDE=""

usage() {
  cat <<'EOF'
scripts/infer_ablation.sh — overlap eval of saved ablation checkpoints (stride 8)

  --jobs LIST       table2 | core | ... or explicit ids (default table2)
  --ckpt-root DIR   Train root or checkpoints/ablations (default $EXP_ROOT/ablation)
  --ckpt PATH       Single checkpoint (use with --cs)
  --cs LIST         default 25
  --alpha LIST      default 0.75
  --datasets LIST   default set11,urban100;  --datasets all  = five test sets
  --out DIR         Eval parent override
  --dry-run
  -h
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="${2:-table2}"; shift 2 ;;
    --ckpt-root) CKPT_ROOT="${2:-}"; shift 2 ;;
    --ckpt) CKPT="${2:-}"; shift 2 ;;
    --cs) CS_LIST="$(expand_cs "${2:-25}")"; shift 2 ;;
    --alpha) ALPHAS="${2:-0.75}"; shift 2 ;;
    --datasets) DATASETS="$(expand_datasets "${2:-set11,urban100}")"; shift 2 ;;
    --out) OUT_OVERRIDE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

CKPT_ROOT="${CKPT_ROOT:-${EXP_ROOT}/ablation}"
maybe_check_set11 "${DATASETS}"

run_eval_ckpt() {
  local ckpt="$1" cs="$2" run_dir="$3"
  local eval_root ds ddir tagname
  eval_root="${OUT_OVERRIDE:-${run_dir}/eval}"
  require_file "${ckpt}"
  echo "INFER ABLATION  CS=${cs}  ckpt=${ckpt}"
  for ds in $(split_csv "${DATASETS}"); do
    ddir="$(dataset_dir "${ds}")"
    tagname="$(eval_tag "${ds}")"
    require_dir "${ddir}"
    eval_shared "${ckpt}" "${eval_root}/eval_${tagname}" "${ddir}" "${cs}"
  done
}

if [[ -n "${CKPT}" ]]; then
  one_cs="$(echo "${CS_LIST}" | tr ',' ' ' | awk '{print $1}')"
  run_eval_ckpt "${CKPT}" "${one_cs}" "$(dirname "${CKPT}")"
  echo "INFER ABLATION done."
  exit 0
fi

mapfile -t JOB_LIST < <(expand_ablation_jobs "${JOBS}")

for job in "${JOB_LIST[@]}"; do
  case "${job}" in
    ne_a060|ne_a090)
      a="$(ablation_alpha "${job}" "0.75")"
      for cs in $(split_csv "${CS_LIST}"); do
        ckpt="$(find_ablation_ckpt "${CKPT_ROOT}" "${cs}" "${job}" "${a}")"
        run_dir="$(dirname "${ckpt}")"
        run_eval_ckpt "${ckpt}" "${cs}" "${run_dir}"
      done
      ;;
    *)
      for alpha in $(split_csv "${ALPHAS}"); do
        a="$(ablation_alpha "${job}" "${alpha}")"
        if [[ "$(ablation_tag "${job}" "${a}")" == "UNKNOWN" ]]; then
          echo "ERROR: unknown ablation job '${job}'"; exit 1
        fi
        for cs in $(split_csv "${CS_LIST}"); do
          ckpt="$(find_ablation_ckpt "${CKPT_ROOT}" "${cs}" "${job}" "${a}")"
          run_dir="$(dirname "${ckpt}")"
          run_eval_ckpt "${ckpt}" "${cs}" "${run_dir}"
        done
      done
      ;;
  esac
done

echo "INFER ABLATION done."
