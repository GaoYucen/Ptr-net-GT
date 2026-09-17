#!/usr/bin/env bash

# AM TSP50 正式信息消融；Full 直接复用正式主表结果。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/paper_ablation_am_tsp50_v1}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${PROJECT_ROOT}/artifacts/paper_final_tsp50_v1}"
TEST_SEED="${TEST_SEED:-20260904}"
TEST_SIZE="${TEST_SIZE:-10000}"
SEEDS=(1234 4321 2468)
MODES=(
  native_random_tail
  native_free_no_head_summary
  native_free_no_path_state
)

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/summary"
cp "${SCRIPT_DIR}/protocol_ablation_am_tsp50_v1.json" \
  "${OUTPUT_ROOT}/protocol_snapshot.json"

"${PYTHON_BIN}" experiments/paper/validate_ablation_am.py \
  --output "${OUTPUT_ROOT}/summary/implementation_validation.json" \
  --device cuda \
  >"${OUTPUT_ROOT}/logs/validate_implementation.log" 2>&1

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="${OUTPUT_ROOT}/train/am_tsp50_${mode}_seed${seed}"
    checkpoint="${run_dir}/checkpoints/step-010000.pt"
    if [[ ! -f "${checkpoint}" ]]; then
      resume_args=()
      if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
        resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
      fi
      "${PYTHON_BIN}" experiments/train.py \
        --problem tsp \
        --base-mode "${mode}" \
        --output-dir "${run_dir}" \
        "${resume_args[@]}" \
        --graph-size 50 \
        --steps 10000 \
        --batch-size 512 \
        --validation-size 1024 \
        --embedding-dim 128 \
        --heads 8 \
        --encoder-layers 3 \
        --feed-forward-dim 512 \
        --normalization batch \
        --learning-rate 1e-4 \
        --eval-every 100 \
        --checkpoint-every 1000 \
        --seed "${seed}" \
        --device cuda \
        >"${OUTPUT_ROOT}/logs/train_${mode}_seed${seed}.log" 2>&1
    fi

    eval_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/am/${mode}/seed${seed}"
    if [[ ! -f "${eval_dir}/summary.json" || ! -f "${eval_dir}/costs.pt" ]]; then
      "${PYTHON_BIN}" experiments/evaluate.py \
        --checkpoint "${checkpoint}" \
        --config "${run_dir}/config.json" \
        --base-mode-override "${mode}" \
        --output-dir "${eval_dir}" \
        --test-size "${TEST_SIZE}" \
        --test-seed "${TEST_SEED}" \
        --graph-size 50 \
        --batch-size 64 \
        --device cuda \
        >"${OUTPUT_ROOT}/logs/eval_${mode}_seed${seed}.log" 2>&1
    fi
  done
done

"${PYTHON_BIN}" experiments/paper/summarize_ablation_am.py \
  --ablation-root "${OUTPUT_ROOT}" \
  --reference-root "${REFERENCE_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/summary" \
  --test-seed "${TEST_SEED}" \
  >"${OUTPUT_ROOT}/logs/summarize.log" 2>&1

checkpoints=$(find "${OUTPUT_ROOT}/train" -path '*/checkpoints/step-010000.pt' | wc -l)
summaries=$(find "${OUTPUT_ROOT}/eval" -name summary.json | wc -l)
costs=$(find "${OUTPUT_ROOT}/eval" -name costs.pt | wc -l)
if [[ "${checkpoints}" -ne 9 || "${summaries}" -ne 9 || "${costs}" -ne 9 ]]; then
  echo "消融实验产物不完整: checkpoints=${checkpoints}, summaries=${summaries}, costs=${costs}" >&2
  exit 20
fi
test -f "${OUTPUT_ROOT}/summary/implementation_validation.json"
test -f "${OUTPUT_ROOT}/summary/ablation_summary.json"
test -f "${OUTPUT_ROOT}/summary/mode_results.csv"
test -f "${OUTPUT_ROOT}/summary/paired_comparisons.csv"
