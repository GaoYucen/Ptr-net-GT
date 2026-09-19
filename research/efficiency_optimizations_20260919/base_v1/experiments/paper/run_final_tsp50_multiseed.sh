#!/usr/bin/env bash

# 论文 TSP50 主表：四个宿主、Original/GroupOpt、三个训练种子的长程收敛结果。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/paper_final_tsp50_v1}"
REFERENCE_ROOT="${REFERENCE_ROOT:-}"
REFERENCE_SEED="${REFERENCE_SEED:-2468}"
TEST_SEED="${TEST_SEED:-20260904}"
TEST_SIZE="${TEST_SIZE:-10000}"
SEEDS=(1234 4321 2468)
FAMILIES=(am ptrnet gpn pomo)
MODES=(native_original native_conditional_free)

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/summary" "${OUTPUT_ROOT}/train"
cp "${SCRIPT_DIR}/protocol_paper_suite_v1.json" "${OUTPUT_ROOT}/protocol_snapshot.json"

"${PYTHON_BIN}" experiments/validate_native_original.py \
  --output "${OUTPUT_ROOT}/summary/native_original_validation.json" \
  --device cuda

configure_family() {
  case "$1" in
    am)
      TRAINER="experiments/train.py"
      FINAL_STEP=10000
      BATCH_SIZE=512
      VALIDATION_SIZE=1024
      CHECKPOINT_EVERY=1000
      EVAL_BATCH=64
      MODEL_ARGS=(--heads 8 --encoder-layers 3 --feed-forward-dim 512 --normalization batch)
      ;;
    ptrnet)
      TRAINER="experiments/train_ptrnet.py"
      FINAL_STEP=10000
      BATCH_SIZE=128
      VALIDATION_SIZE=1024
      CHECKPOINT_EVERY=1000
      EVAL_BATCH=64
      MODEL_ARGS=(--encoder-layers 1)
      ;;
    gpn)
      TRAINER="experiments/train_gpn.py"
      FINAL_STEP=10000
      BATCH_SIZE=128
      VALIDATION_SIZE=1024
      CHECKPOINT_EVERY=1000
      EVAL_BATCH=64
      MODEL_ARGS=(--encoder-layers 3)
      ;;
    pomo)
      TRAINER="experiments/train_pomo.py"
      FINAL_STEP=5000
      BATCH_SIZE=64
      VALIDATION_SIZE=64
      CHECKPOINT_EVERY=500
      EVAL_BATCH=8
      MODEL_ARGS=(--training-scheme pomo --pomo-size 8 --qkv-dim 16 --heads 8 --encoder-layers 6 --feed-forward-dim 512)
      ;;
    *)
      echo "不支持的宿主：$1" >&2
      return 2
      ;;
  esac
}

import_reference() {
  local family="$1"
  local mode="$2"
  local seed="$3"
  [[ -n "${REFERENCE_ROOT}" && "${seed}" == "${REFERENCE_SEED}" ]] || return 1
  configure_family "${family}"
  local source_dir="${REFERENCE_ROOT}/train/${family}_tsp50_${mode}_seed${seed}"
  local checkpoint="${source_dir}/checkpoints/step-$(printf '%06d' "${FINAL_STEP}").pt"
  [[ -f "${source_dir}/config.json" && -f "${checkpoint}" ]] || return 1

  local run_dir="${OUTPUT_ROOT}/train/${family}_tsp50_${mode}_seed${seed}"
  mkdir -p "${run_dir}/checkpoints"
  cp "${source_dir}/config.json" "${run_dir}/config.json"
  [[ ! -f "${source_dir}/metrics.jsonl" ]] || cp "${source_dir}/metrics.jsonl" "${run_dir}/metrics.jsonl"
  cp "${checkpoint}" "${run_dir}/checkpoints/step-$(printf '%06d' "${FINAL_STEP}").pt"
  printf '%s\n' "${source_dir}" > "${run_dir}/imported_from.txt"
  return 0
}

train_one() {
  local family="$1"
  local mode="$2"
  local seed="$3"
  configure_family "${family}"
  local run_dir="${OUTPUT_ROOT}/train/${family}_tsp50_${mode}_seed${seed}"
  local checkpoint="${run_dir}/checkpoints/step-$(printf '%06d' "${FINAL_STEP}").pt"
  [[ ! -f "${checkpoint}" ]] || return 0
  if import_reference "${family}" "${mode}" "${seed}"; then
    return 0
  fi
  local resume_args=()
  if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
  fi

  "${PYTHON_BIN}" "${TRAINER}" \
    --base-mode "${mode}" \
    --output-dir "${run_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps "${FINAL_STEP}" \
    --batch-size "${BATCH_SIZE}" \
    --validation-size "${VALIDATION_SIZE}" \
    --embedding-dim 128 \
    "${MODEL_ARGS[@]}" \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --seed "${seed}" \
    --device cuda
}

wait_wave() {
  local failed=0
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  [[ "${failed}" -eq 0 ]]
}

# 每个宿主内并行 Original 和 GroupOpt；种子与宿主分波，不超过 24GB 显存。
for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    pids=()
    for mode in "${MODES[@]}"; do
      train_one "${family}" "${mode}" "${seed}" \
        >"${OUTPUT_ROOT}/logs/train_${family}_${mode}_seed${seed}.log" 2>&1 &
      pids+=("$!")
    done
    if ! wait_wave "${pids[@]}"; then
      echo "${family} seed=${seed} 训练失败；保留日志和 checkpoint。" >&2
      exit 10
    fi
  done
done

for family in "${FAMILIES[@]}"; do
  configure_family "${family}"
  for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
      run_dir="${OUTPUT_ROOT}/train/${family}_tsp50_${mode}_seed${seed}"
      checkpoint="${run_dir}/checkpoints/step-$(printf '%06d' "${FINAL_STEP}").pt"
      evaluation_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/${family}/${mode}/seed${seed}"
      if [[ ! -f "${checkpoint}" || ! -f "${run_dir}/config.json" ]]; then
        echo "缺少 checkpoint 或配置：${run_dir}" >&2
        exit 20
      fi
      if [[ -f "${evaluation_dir}/summary.json" && -f "${evaluation_dir}/costs.pt" ]]; then
        continue
      fi
      "${PYTHON_BIN}" experiments/evaluate.py \
        --checkpoint "${checkpoint}" \
        --config "${run_dir}/config.json" \
        --base-mode-override "${mode}" \
        --output-dir "${evaluation_dir}" \
        --test-size "${TEST_SIZE}" \
        --test-seed "${TEST_SEED}" \
        --graph-size 50 \
        --batch-size "${EVAL_BATCH}" \
        --device cuda \
        >"${OUTPUT_ROOT}/logs/eval_${family}_${mode}_seed${seed}.log" 2>&1
    done
  done
done

"${PYTHON_BIN}" experiments/paper/summarize_main_tsp50.py \
  --root "${OUTPUT_ROOT}/eval" \
  --output-dir "${OUTPUT_ROOT}/summary" \
  --families "${FAMILIES[@]}" \
  --seeds "${SEEDS[@]}" \
  --test-seed "${TEST_SEED}" \
  --experiment-id paper_final_tsp50_v1

expected_long_checkpoints=$(find "${OUTPUT_ROOT}/train" -type f \
  \( -path '*/am_tsp50_*/checkpoints/step-010000.pt' \
     -o -path '*/ptrnet_tsp50_*/checkpoints/step-010000.pt' \
     -o -path '*/gpn_tsp50_*/checkpoints/step-010000.pt' \) | wc -l)
expected_pomo_checkpoints=$(find "${OUTPUT_ROOT}/train" -type f \
  -path '*/pomo_tsp50_*/checkpoints/step-005000.pt' | wc -l)
expected_summaries=$(find "${OUTPUT_ROOT}/eval" -name summary.json | wc -l)
expected_costs=$(find "${OUTPUT_ROOT}/eval" -name costs.pt | wc -l)
if [[ "${expected_long_checkpoints}" -ne 18 || "${expected_pomo_checkpoints}" -ne 6 \
   || "${expected_summaries}" -ne 24 || "${expected_costs}" -ne 24 ]]; then
  echo "正式主表产物不完整: long_checkpoints=${expected_long_checkpoints}, pomo_checkpoints=${expected_pomo_checkpoints}, summaries=${expected_summaries}, costs=${expected_costs}" >&2
  exit 30
fi
