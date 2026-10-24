#!/bin/bash
# Shared helpers for scripts/*.sh (source this; do not run it).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
  PYTHON="${REPO_ROOT}/.venv/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/Scripts/python.exe" ]]; then
  PYTHON="${REPO_ROOT}/.venv/Scripts/python.exe"
fi

PROJECT_DIR="${PROJECT_DIR:-${REPO_ROOT}}"
SCRATCH="${SCRATCH:-${SCRATCH_ROOT:-${REPO_ROOT}/runs}}"
EXP_ROOT="${EXP_ROOT:-${SCRATCH}/paper}"
# Change these to your machine. Test sets default to $REPO_ROOT/data/{Set11,...}.
# Train/calib uses COCO unlabeled2017 via COCO_DIR.
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
COCO_DIR="${COCO_DIR:-${SCRATCH}/datasets/coco/unlabeled2017}"
if [[ ! -d "${COCO_DIR}" && -d "${PROJECT_DIR}/data/coco/unlabeled2017" ]]; then
  COCO_DIR="${PROJECT_DIR}/data/coco/unlabeled2017"
fi

SIGMA_FUSE="${SIGMA_FUSE:-UV}"
GRAD_ALPHA="${GRAD_ALPHA:-1.0}"
CALIB_NSAMPLES="${CALIB_NSAMPLES:-64}"
CALIB_BATCH="${CALIB_BATCH:-2}"
FT_MAX_EPOCH="${FT_MAX_EPOCH:-30}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-10}"
FT_LR="${FT_LR:-1e-4}"
FT_NUM_WORKERS="${FT_NUM_WORKERS:-8}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
CALIB_BATCHES_FFN="${CALIB_BATCHES_FFN:-64}"
DRY_RUN="${DRY_RUN:-0}"
# Main Table 1 grid (Dense / Ours-QKV / Ours-QKV+FFN Taylor)
TABLE1_CS="${TABLE1_CS:-1,4,10,25,50}"
TABLE1_DATASETS="${TABLE1_DATASETS:-set11,set14,bsd68,urban100,set5}"

ARCH=(
  --gf_dim 128
  --g_depth 5,5,5,5
  --num_heads 16,8,4,2
  --g_window_size 8
  --bottom_width 8
  --img_size 64
  --datarange -11
)

run_cmd() {
  echo "+" "$@"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

dataset_dir() {
  local name
  name="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  local candidates=()
  case "${name}" in
    set11) candidates=("${DATA_ROOT}/Set11" "${PROJECT_DIR}/data/Set11") ;;
    set14) candidates=("${DATA_ROOT}/Set14" "${SCRATCH}/datasets/Set14" "${PROJECT_DIR}/data/Set14") ;;
    set5) candidates=("${DATA_ROOT}/Set5" "${SCRATCH}/datasets/Set5" "${PROJECT_DIR}/data/Set5") ;;
    bsd68) candidates=("${DATA_ROOT}/BSD68" "${SCRATCH}/datasets/BSD68" "${PROJECT_DIR}/data/BSD68") ;;
    urban100|urban) candidates=("${DATA_ROOT}/Urban100" "${SCRATCH}/datasets/Urban100" "${PROJECT_DIR}/data/Urban100") ;;
    *) echo "ERROR: unknown dataset ${1}" >&2; return 1 ;;
  esac
  local d
  for d in "${candidates[@]}"; do
    if [[ -d "${d}" ]]; then
      echo "${d}"
      return 0
    fi
  done
  echo "${candidates[0]}"
}

eval_tag() {
  local name
  name="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${name}" in
    set11) echo "Set11" ;;
    set14) echo "Set14" ;;
    set5) echo "Set5" ;;
    bsd68) echo "BSD68" ;;
    urban100|urban) echo "Urban100" ;;
    *) echo "$1" ;;
  esac
}

split_csv() {
  local raw="${1// /}"
  echo "${raw}" | tr ',' ' '
}

expand_cs() {
  local v="${1:-all}"
  case "${v}" in
    all|table1) echo "${TABLE1_CS}" ;;
    *) echo "${v}" ;;
  esac
}

expand_datasets() {
  local v="${1:-all}"
  case "${v}" in
    all|table1) echo "${TABLE1_DATASETS}" ;;
    *) echo "${v}" ;;
  esac
}

require_file() {
  local p="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "${p}" ]]; then
    echo "ERROR: missing file: ${p}" >&2
    exit 1
  fi
}

require_dir() {
  local p="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ ! -d "${p}" ]]; then
    echo "ERROR: missing directory: ${p}" >&2
    echo "Set DATA_ROOT (Set11/Set14/Set5/BSD68/Urban100) and COCO_DIR (COCO unlabeled2017) to your paths." >&2
    exit 1
  fi
}

check_set11() {
  local d
  d="$(dataset_dir set11)"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  require_dir "${d}"
  local n
  n="$(find "${d}" -maxdepth 1 \( -iname "*.tif" -o -iname "*.png" -o -iname "*.jpg" \) | wc -l | tr -d ' ')"
  echo "Set11 image count: ${n}"
  if [[ "${n}" -ne 11 ]]; then
    echo "ERROR: Set11 has ${n} images (expected 11)" >&2
    exit 1
  fi
}

# Infer: only enforce the 11-image rule if Set11 is in --datasets.
maybe_check_set11() {
  local list="${1:-}"
  local ds
  for ds in $(split_csv "${list}"); do
    ds="$(echo "${ds}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${ds}" == "set11" ]]; then
      check_set11
      return 0
    fi
  done
}

dense_ckpt() {
  local cs="$1"
  if [[ -n "${DENSE_CKPT:-}" ]]; then
    echo "${DENSE_CKPT}"
    return 0
  fi
  local cands=(
    "${PROJECT_DIR}/checkpoints/csformer/CS${cs}/checkpoint.pth"
    "${PROJECT_DIR}/logs/checkpoint_coco/checkpoint_CS${cs}.pth"
  )
  local p
  for p in "${cands[@]}"; do
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  echo "${cands[0]}"
}

qkv_dir() {
  local root="$1" cs="$2" score="$3" alpha="$4"
  echo "${root}/CS${cs}_${score}_a${alpha}"
}

qkv_ckpt() {
  echo "$(qkv_dir "$1" "$2" "$3" "$4")/finetuned/Model/checkpoint_best.pth"
}

product_dir() {
  local root="$1" cs="$2" ffn="$3" keep="$4" alpha="$5"
  echo "${root}/CS${cs}_${ffn}_k${keep}_a${alpha}"
}

product_ckpt() {
  echo "$(product_dir "$1" "$2" "$3" "$4" "$5")/finetuned/Model/checkpoint_best.pth"
}

product_idx() {
  echo "$(product_dir "$1" "$2" "$3" "$4" "$5")/pruned/prune_indices.pt"
}

ablation_tag() {
  local job="$1" alpha="$2"
  case "${job}" in
    uniform) echo "uniform_a${alpha}" ;;
    sigma) echo "score_sigma_a${alpha}" ;;
    sigma_g) echo "score_sigma_g_a${alpha}" ;;
    classical_4e) echo "score_classical4e_a${alpha}" ;;
    ne) echo "score_ne_a${alpha}" ;;
    ne_l025) echo "score_ne_l025_a${alpha}" ;;
    ne_l05) echo "score_ne_l05_a${alpha}" ;;
    ne_l075) echo "score_ne_l075_a${alpha}" ;;
    rn) echo "score_rn_a${alpha}" ;;
    re) echo "score_re_a${alpha}" ;;
    matched_topr) echo "matched_topr_from_ne_a${alpha}" ;;
    ne_a090) echo "score_ne_a0.90" ;;
    ne_a060) echo "score_ne_a0.60" ;;
    *) echo "UNKNOWN" ;;
  esac
}

ablation_alpha() {
  local job="$1" default_alpha="$2"
  case "${job}" in
    ne_a090) echo "0.90" ;;
    ne_a060) echo "0.60" ;;
    *) echo "${default_alpha}" ;;
  esac
}

ablation_score_mode() {
  local job="$1"
  case "${job}" in
    sigma) echo "sigma" ;;
    sigma_g) echo "sigma_g" ;;
    classical_4e) echo "classical_4e" ;;
    ne|ne_a090|ne_a060) echo "ne" ;;
    ne_l025) echo "ne_l025" ;;
    ne_l05) echo "ne_l05" ;;
    ne_l075) echo "ne_l075" ;;
    rn) echo "rn" ;;
    re) echo "re" ;;
    *) echo "" ;;
  esac
}

ablation_dir() {
  local root="$1" cs="$2" job="$3" alpha="$4"
  echo "${root}/CS${cs}_${job}_a${alpha}"
}

ablation_ckpt() {
  echo "$(ablation_dir "$1" "$2" "$3" "$4")/finetuned/Model/checkpoint_best.pth"
}

expand_ablation_jobs() {
  local raw="$1" p
  local out=()
  IFS=',' read -ra parts <<< "${raw}"
  for p in "${parts[@]}"; do
    p="$(echo "${p}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    case "${p}" in
      table2)
        # Main Table 2, including R/N. ne before matched_topr.
        out+=(uniform sigma sigma_g ne matched_topr classical_4e rn)
        ;;
      ladder) out+=(uniform sigma sigma_g classical_4e ne) ;;
      all) out+=(uniform sigma sigma_g classical_4e ne matched_topr) ;;
      core) out+=(uniform sigma sigma_g ne matched_topr classical_4e) ;;
      lambda_sweep) out+=(ne_l025 ne_l05 ne_l075) ;;
      denom_split) out+=(rn re) ;;
      h100_extra) out+=(ne_l025 ne_l05 ne_l075 rn re) ;;
      alpha_sweep) out+=(ne_a060 ne ne_a090) ;;
      "") ;;
      *) out+=("${p}") ;;
    esac
  done
  printf '%s\n' "${out[@]}"
}

find_ablation_ckpt() {
  local root="$1" cs="$2" job="$3" alpha="$4"
  local tag
  tag="$(ablation_tag "${job}" "${alpha}")"
  local cands=(
    "$(ablation_ckpt "${root}" "${cs}" "${job}" "${alpha}")"
    "${root}/finetuned/CS${cs}_${tag}/Model/checkpoint_best.pth"
    "${root}/${job}/checkpoint.pth"
    "${root}/ablations/${job}/checkpoint.pth"
    "${PROJECT_DIR}/checkpoints/ablations/${job}/checkpoint.pth"
  )
  local p
  for p in "${cands[@]}"; do
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  echo "${cands[0]}"
}

apply_dry_run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_cmd() { echo "+" "$@"; }
  fi
}

eval_shared() {
  local ckpt="$1"
  local out="$2"
  local data_dir="$3"
  local cs="$4"
  local ffn_idx="${5:-}"
  local dense="${6:-0}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${out}"
  fi
  local extra=()
  if [[ "${dense}" == "1" ]]; then
    extra+=(--gen_model CSformer)
  else
    extra+=(--gen_model CSformer_QSVD_Shared --use_qsvd)
  fi
  if [[ -n "${ffn_idx}" ]]; then
    extra+=(--ffn_prune_indices "${ffn_idx}")
  fi
  local cmd=(
    "${PYTHON}" eval.py
    --load_path "${ckpt}"
    --exp_name "${out}"
    --testdata_path "${data_dir}"
    --valdata_path "${data_dir}"
    --overlap --overlapstep 8
    --cs_ratio "${cs}"
    "${ARCH[@]}"
    "${extra[@]}"
  )
  echo "+" "${cmd[@]}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}" 2>&1 | tee "${out}/eval_stdout.log"
  fi
}
