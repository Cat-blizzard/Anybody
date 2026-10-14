#!/usr/bin/env bash
# Run batch_csv_to_npz.py on CSV shards csv_by_gpu/0 .. csv_by_gpu/7 (one Isaac process per GPU).
#
# Usage:
#   ./scripts/run_batch_csv_to_npz_shards.sh
#   MOSAIC_ROOT=... DATA_ROOT=... ./scripts/run_batch_csv_to_npz_shards.sh
#   SEQUENTIAL=1 ./scripts/run_batch_csv_to_npz_shards.sh    # one GPU job at a time
#   MAX_PARALLEL=4 ./scripts/run_batch_csv_to_npz_shards.sh  # cap concurrent Isaac instances
#   DRY_RUN=1 ./scripts/run_batch_csv_to_npz_shards.sh       # print commands; no output/native startup
#   ROOT_VELOCITY_MODE=legacy_com_v1 OUT_DIR=/new/path ./scripts/run_batch_csv_to_npz_shards.sh
#
# Each worker sets CUDA_VISIBLE_DEVICES to its shard index. Ensure csv_by_gpu/N contains *.csv
# (optionally in subdirs); empty shards are skipped with a warning.

set -euo pipefail

CONVERTER_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOSAIC_ROOT="${MOSAIC_ROOT:-$(cd -- "${CONVERTER_SCRIPT_DIR}/.." && pwd)}"
MOSAIC_ROOT="$(cd -- "${MOSAIC_ROOT}" && pwd)"
CONVERTER_WORKER_PYTHONPATH="${MOSAIC_ROOT}/source/whole_body_tracking:${MOSAIC_ROOT}/source/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
DATA_ROOT="${DATA_ROOT:-/home/lsn/Datasets/bones-seed/g1}"
CSV_PARENT="${CSV_PARENT:-${DATA_ROOT}/csv_by_gpu}"
ROOT_VELOCITY_MODE="${ROOT_VELOCITY_MODE:-link_v2}"
CONVERSION_RUN_ID="${CONVERSION_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/npz_${ROOT_VELOCITY_MODE}_${CONVERSION_RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-bones_seed_g1}"
INPUT_FPS="${INPUT_FPS:-120}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
FIRST_SHARD="${FIRST_SHARD:-0}"
LAST_SHARD="${LAST_SHARD:-7}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SEQUENTIAL="${SEQUENTIAL:-0}"

SCRIPT="${MOSAIC_ROOT}/scripts/batch_csv_to_npz.py"
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: batch script not found: ${SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${MOSAIC_ROOT}/scripts/root_velocity_contract.py" ]]; then
  echo "ERROR: versioned converter helper missing; refusing a historical converter: ${MOSAIC_ROOT}" >&2
  exit 1
fi
if [[ "${ROOT_VELOCITY_MODE}" != "link_v2" && "${ROOT_VELOCITY_MODE}" != "legacy_com_v1" ]]; then
  echo "ERROR: unsupported ROOT_VELOCITY_MODE: ${ROOT_VELOCITY_MODE}" >&2
  exit 1
fi
if [[ -e "${OUT_DIR}" || -L "${OUT_DIR}" ]]; then
  echo "ERROR: output campaign directory must be new: ${OUT_DIR}" >&2
  exit 1
fi
if [[ ! "${FIRST_SHARD}" =~ ^[0-9]+$ || ! "${LAST_SHARD}" =~ ^[0-9]+$ ||
      ! "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ || "${FIRST_SHARD}" -gt "${LAST_SHARD}" ]]; then
  echo "ERROR: shard range and positive MAX_PARALLEL must be integers" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "ERROR: DRY_RUN must be 0 or 1" >&2
  exit 1
fi

has_csv_under() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 1
  find "${dir}" -type f -name '*.csv' -print -quit | grep -q .
}

run_shard() {
  local i="$1"
  local indir="${CSV_PARENT}/${i}"
  if ! has_csv_under "${indir}"; then
    echo "[WARN] No CSV files under ${indir} — skip shard ${i}" >&2
    return 0
  fi
  # Each worker owns a distinct, nonexistent directory. Never share batch output roots.
  local shard_output="${OUT_DIR}/shard_${i}"
  local command=("${PYTHON_BIN}" "${SCRIPT}"
    --input_dir "${indir}" --output_dir "${shard_output}"
    --output_prefix "${OUTPUT_PREFIX}" --input_fps "${INPUT_FPS}"
    --output_fps "${OUTPUT_FPS}" --root_velocity_mode "${ROOT_VELOCITY_MODE}"
    --bones_seed_g1_csv --headless)
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q ' "${i}" "${CONVERTER_WORKER_PYTHONPATH}"
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi
  echo "[INFO] Shard ${i}: input=${indir} -> ${shard_output}"
  CUDA_VISIBLE_DEVICES="${i}" PYTHONPATH="${CONVERTER_WORKER_PYTHONPATH}" "${command[@]}"
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for i in $(seq "${FIRST_SHARD}" "${LAST_SHARD}"); do
    run_shard "${i}"
  done
  exit 0
fi
# Exclusively reserve the campaign parent; Python reserves fresh shard children.
mkdir -- "${OUT_DIR}"

if [[ "${SEQUENTIAL}" == "1" ]]; then
  for i in $(seq "${FIRST_SHARD}" "${LAST_SHARD}"); do
    run_shard "${i}"
  done
  echo "All shard jobs finished."
  exit 0
fi

# Parallel batches (portable; does not rely on bash 5.1+ wait -n)
i="${FIRST_SHARD}"
while [[ "${i}" -le "${LAST_SHARD}" ]]; do
  pids=()
  for ((k = 0; k < MAX_PARALLEL && i <= LAST_SHARD; k++)); do
    run_shard "${i}" &
    pids+=("$!")
    i=$((i + 1))
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

echo "All shard jobs finished."
