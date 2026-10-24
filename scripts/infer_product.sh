#!/bin/bash
# Infer Ours-QKV+FFN from saved FT checkpoints + prune_indices.pt. No training.
#
#   bash scripts/infer_product.sh
#   bash scripts/infer_product.sh --ckpt-root runs/paper/product --cs 25 --ffn taylor

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

CS_LIST="${TABLE1_CS}"
ALPHAS="0.75"
FFN_LIST="taylor"
DATASETS="${TABLE1_DATASETS}"
CKPT_ROOT=""
CKPT=""
FFN_IDX=""
OUT_OVERRIDE=""

usage() {
  cat <<'EOF'
scripts/infer_product.sh — overlap eval of saved Ours-QKV+FFN checkpoints (stride 8)

  --ckpt-root DIR   Train output root (default $EXP_ROOT/product)
  --ckpt PATH       Single checkpoint_best.pth (use with --cs --ffn-idx)
  --ffn-idx PATH    prune_indices.pt (required with --ckpt)
  --cs LIST         default 1,4,10,25,50
  --alpha LIST      Rank budget used at train time (default 0.75)
  --ffn LIST        default taylor
  --keep R          Must match train (default 0.5)
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
    --ffn-idx) FFN_IDX="${2:-}"; shift 2 ;;
    --cs) CS_LIST="$(expand_cs "${2:-all}")"; shift 2 ;;
    --alpha) ALPHAS="${2:-0.75}"; shift 2 ;;
    --ffn) FFN_LIST="${2:-taylor}"; shift 2 ;;
    --keep) KEEP_RATIO="${2:-0.5}"; shift 2 ;;
    --datasets) DATASETS="$(expand_datasets "${2:-all}")"; shift 2 ;;
    --out) OUT_OVERRIDE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

CKPT_ROOT="${CKPT_ROOT:-${EXP_ROOT}/product}"
maybe_check_set11 "${DATASETS}"

run_eval_ckpt() {
  local ckpt="$1" idx="$2" cs="$3" run_dir="$4"
  local eval_root ds ddir tagname
  eval_root="${OUT_OVERRIDE:-${run_dir}/eval}"
  require_file "${ckpt}"
  require_file "${idx}"
  echo "INFER QKV+FFN  CS=${cs}  ckpt=${ckpt}"
  for ds in $(split_csv "${DATASETS}"); do
    ddir="$(dataset_dir "${ds}")"
    tagname="$(eval_tag "${ds}")"
    require_dir "${ddir}"
    eval_shared "${ckpt}" "${eval_root}/eval_${tagname}" "${ddir}" "${cs}" "${idx}"
  done
}

if [[ -n "${CKPT}" ]]; then
  if [[ -z "${FFN_IDX}" ]]; then
    echo "ERROR: --ckpt requires --ffn-idx"; exit 1
  fi
  one_cs="$(echo "${CS_LIST}" | tr ',' ' ' | awk '{print $1}')"
  run_eval_ckpt "${CKPT}" "${FFN_IDX}" "${one_cs}" "$(dirname "$(dirname "${CKPT}")")"
else
  for alpha in $(split_csv "${ALPHAS}"); do
    for ffn in $(split_csv "${FFN_LIST}"); do
      for cs in $(split_csv "${CS_LIST}"); do
        run_dir="$(product_dir "${CKPT_ROOT}" "${cs}" "${ffn}" "${KEEP_RATIO}" "${alpha}")"
        run_eval_ckpt \
          "$(product_ckpt "${CKPT_ROOT}" "${cs}" "${ffn}" "${KEEP_RATIO}" "${alpha}")" \
          "$(product_idx "${CKPT_ROOT}" "${cs}" "${ffn}" "${KEEP_RATIO}" "${alpha}")" \
          "${cs}" "${run_dir}"
      done
    done
  done
fi

echo "INFER QKV+FFN done."
