#!/bin/bash
# Infer Ours-QKV from saved FT checkpoints. No training.
#
#   bash scripts/infer_qkv.sh
#   bash scripts/infer_qkv.sh --ckpt-root runs/paper/qkv --cs 25 --alpha 0.75 --datasets set11

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

CS_LIST="${TABLE1_CS}"
ALPHAS="0.75"
SCORE="ne"
DATASETS="${TABLE1_DATASETS}"
CKPT_ROOT=""
CKPT=""
OUT_OVERRIDE=""

usage() {
  cat <<'EOF'
scripts/infer_qkv.sh — overlap eval of saved Ours-QKV checkpoints (stride 8)

  --ckpt-root DIR   Train output root (default $EXP_ROOT/qkv)
  --ckpt PATH       Single checkpoint_best.pth (use with --cs)
  --cs LIST         default 1,4,10,25,50
  --alpha LIST      Rank budget used at train time (default 0.75)
  --score MODE      Must match train (default ne)
  --datasets LIST   default set11,set14,bsd68,urban100,set5
  --out DIR         Eval parent (default <run>/eval)
  --dry-run
  -h
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-root) CKPT_ROOT="${2:-}"; shift 2 ;;
    --ckpt) CKPT="${2:-}"; shift 2 ;;
    --cs) CS_LIST="$(expand_cs "${2:-all}")"; shift 2 ;;
    --alpha) ALPHAS="${2:-0.75}"; shift 2 ;;
    --score) SCORE="${2:-ne}"; shift 2 ;;
    --datasets) DATASETS="$(expand_datasets "${2:-all}")"; shift 2 ;;
    --out) OUT_OVERRIDE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

CKPT_ROOT="${CKPT_ROOT:-${EXP_ROOT}/qkv}"
maybe_check_set11 "${DATASETS}"

run_eval_ckpt() {
  local ckpt="$1" cs="$2" run_dir="$3"
  local eval_root ds ddir tagname
  eval_root="${OUT_OVERRIDE:-${run_dir}/eval}"
  require_file "${ckpt}"
  echo "INFER QKV  CS=${cs}  ckpt=${ckpt}"
  for ds in $(split_csv "${DATASETS}"); do
    ddir="$(dataset_dir "${ds}")"
    tagname="$(eval_tag "${ds}")"
    require_dir "${ddir}"
    eval_shared "${ckpt}" "${eval_root}/eval_${tagname}" "${ddir}" "${cs}"
  done
}

if [[ -n "${CKPT}" ]]; then
  one_cs="$(echo "${CS_LIST}" | tr ',' ' ' | awk '{print $1}')"
  run_eval_ckpt "${CKPT}" "${one_cs}" "$(dirname "$(dirname "${CKPT}")")"
else
  for alpha in $(split_csv "${ALPHAS}"); do
    for cs in $(split_csv "${CS_LIST}"); do
      run_dir="$(qkv_dir "${CKPT_ROOT}" "${cs}" "${SCORE}" "${alpha}")"
      run_eval_ckpt "$(qkv_ckpt "${CKPT_ROOT}" "${cs}" "${SCORE}" "${alpha}")" "${cs}" "${run_dir}"
    done
  done
fi

echo "INFER QKV done."
