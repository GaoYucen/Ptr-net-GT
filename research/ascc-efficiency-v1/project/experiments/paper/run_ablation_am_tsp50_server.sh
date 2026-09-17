#!/usr/bin/env bash

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/groupopt-paper-canonical}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/groupopt-data/paper_ablation_am_tsp50_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/paper_ablation_am_tsp50_v1}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${PROJECT_ROOT}/artifacts/paper_final_tsp50_v1}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/groupopt-venv/bin/python}"
EXIT_FILE="${EXIT_FILE:-${DATA_ROOT}/paper_ablation_am_tsp50_v1.exit}"
MAIN_LOG="${MAIN_LOG:-${DATA_ROOT}/paper_ablation_am_tsp50_v1.log}"

mkdir -p "${DATA_ROOT}"
rm -f "${EXIT_FILE}"
finish() {
  local status=$?
  trap - EXIT
  printf '%s\n' "${status}" >"${EXIT_FILE}.tmp"
  mv "${EXIT_FILE}.tmp" "${EXIT_FILE}"
  exit "${status}"
}
trap finish EXIT

OUTPUT_ROOT="${OUTPUT_ROOT}" REFERENCE_ROOT="${REFERENCE_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/experiments/paper/run_ablation_am_tsp50.sh" \
  >"${MAIN_LOG}" 2>&1
