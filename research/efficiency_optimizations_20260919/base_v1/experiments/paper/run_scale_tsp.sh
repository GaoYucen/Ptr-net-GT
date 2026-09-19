#!/usr/bin/env bash

# TSP20/100 分别训练；TSP50 从已完成的正式主表读取。
# TSP200 曾完成资源探针及部分训练，但本轮论文实验暂不纳入正式汇总。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/paper_scale_effect_tsp_v1}"
TSP50_ROOT="${TSP50_ROOT:?must point to paper_final_tsp50_v1}"
TEST_SEED="${TEST_SEED:-20260904}"
TEST_SIZE="${TEST_SIZE:-10000}"
SIZES=(20 100)
FAMILIES=(am pomo)
MODES=(native_original native_conditional_free)
SEEDS=(1234 4321 2468)

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/summary"
cp "${SCRIPT_DIR}/protocol_scale_tsp_v1.json" "${OUTPUT_ROOT}/protocol_snapshot.json"

configure() {
  local family="$1"
  local size="$2"
  case "${family}" in
    am)
      TRAINER="experiments/train.py"
      FINAL_STEP=10000
      MODEL_ARGS=(--heads 8 --encoder-layers 3 --feed-forward-dim 512 --normalization batch)
      case "${size}" in
        20) BATCH_SIZE=512; VALIDATION_SIZE=1024; EVAL_EVERY=200; EVAL_BATCH=256 ;;
        100) BATCH_SIZE=64; VALIDATION_SIZE=64; EVAL_EVERY=250; EVAL_BATCH=16 ;;
        200) BATCH_SIZE=16; VALIDATION_SIZE=16; EVAL_EVERY=500; EVAL_BATCH=4 ;;
      esac
      ;;
    pomo)
      TRAINER="experiments/train_pomo.py"
      FINAL_STEP=5000
      MODEL_ARGS=(--training-scheme pomo --pomo-size 8 --qkv-dim 16 --heads 8 --encoder-layers 6 --feed-forward-dim 512)
      case "${size}" in
        20) BATCH_SIZE=64; VALIDATION_SIZE=64; EVAL_EVERY=200; EVAL_BATCH=32 ;;
        100) BATCH_SIZE=8; VALIDATION_SIZE=4; EVAL_EVERY=250; EVAL_BATCH=4 ;;
        200) BATCH_SIZE=2; VALIDATION_SIZE=1; EVAL_EVERY=500; EVAL_BATCH=1 ;;
      esac
      ;;
    *) echo "unsupported family: ${family}" >&2; return 2 ;;
  esac
}

train_one() {
  local family="$1" size="$2" mode="$3" seed="$4"
  configure "${family}" "${size}"
  local run_dir="${OUTPUT_ROOT}/train/${family}_tsp${size}_${mode}_seed${seed}"
  local checkpoint="${run_dir}/checkpoints/step-$(printf '%06d' "${FINAL_STEP}").pt"
  [[ ! -f "${checkpoint}" ]] || return 0
  local resume_args=()
  if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
  fi
  "${PYTHON_BIN}" "${TRAINER}" \
    --base-mode "${mode}" \
    --output-dir "${run_dir}" \
    "${resume_args[@]}" \
    --graph-size "${size}" \
    --steps "${FINAL_STEP}" \
    --batch-size "${BATCH_SIZE}" \
    --validation-size "${VALIDATION_SIZE}" \
    --embedding-dim 128 \
    "${MODEL_ARGS[@]}" \
    --learning-rate 1e-4 \
    --eval-every "${EVAL_EVERY}" \
    --checkpoint-every "${FINAL_STEP}" \
    --seed "${seed}" \
    --device cuda
}

evaluate_one() {
  local family="$1" size="$2" mode="$3" seed="$4"
  configure "${family}" "${size}"
  local run_dir="${OUTPUT_ROOT}/train/${family}_tsp${size}_${mode}_seed${seed}"
  local checkpoint="${run_dir}/checkpoints/step-$(printf '%06d' "${FINAL_STEP}").pt"
  local eval_dir="${OUTPUT_ROOT}/eval/iid_tsp${size}_seed${TEST_SEED}/${family}/${mode}/seed${seed}"
  if [[ -f "${eval_dir}/summary.json" && -f "${eval_dir}/costs.pt" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" experiments/evaluate.py \
    --checkpoint "${checkpoint}" \
    --config "${run_dir}/config.json" \
    --base-mode-override "${mode}" \
    --output-dir "${eval_dir}" \
    --test-size "${TEST_SIZE}" \
    --test-seed "${TEST_SEED}" \
    --graph-size "${size}" \
    --batch-size "${EVAL_BATCH}" \
    --device cuda
}

wait_wave() {
  local failed=0
  for pid in "$@"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  [[ "${failed}" -eq 0 ]]
}

for size in "${SIZES[@]}"; do
  for family in "${FAMILIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      pids=()
      for mode in "${MODES[@]}"; do
        train_one "${family}" "${size}" "${mode}" "${seed}" \
          >"${OUTPUT_ROOT}/logs/train_${family}_tsp${size}_${mode}_seed${seed}.log" 2>&1 &
        pids+=("$!")
      done
      if ! wait_wave "${pids[@]}"; then
        echo "training failed: family=${family} size=${size} seed=${seed}" >&2
        exit 10
      fi
    done
  done
done

for size in "${SIZES[@]}"; do
  for family in "${FAMILIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for mode in "${MODES[@]}"; do
        evaluate_one "${family}" "${size}" "${mode}" "${seed}" \
          >"${OUTPUT_ROOT}/logs/eval_${family}_tsp${size}_${mode}_seed${seed}.log" 2>&1
      done
    done
  done
done

"${PYTHON_BIN}" experiments/paper/summarize_scale_tsp.py \
  --scale-root "${OUTPUT_ROOT}" \
  --tsp50-root "${TSP50_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/summary" \
  --sizes 20 50 100 \
  --families am pomo \
  --seeds 1234 4321 2468 \
  --test-seed "${TEST_SEED}" \
  >"${OUTPUT_ROOT}/logs/summarize.log" 2>&1

final_checkpoints=0
for size in "${SIZES[@]}"; do
  for family in "${FAMILIES[@]}"; do
    configure "${family}" "${size}"
    for seed in "${SEEDS[@]}"; do
      for mode in "${MODES[@]}"; do
        checkpoint="${OUTPUT_ROOT}/train/${family}_tsp${size}_${mode}_seed${seed}/checkpoints/step-$(printf '%06d' "${FINAL_STEP}").pt"
        [[ -f "${checkpoint}" ]] && final_checkpoints=$((final_checkpoints + 1))
      done
    done
  done
done
eval_summaries=$(find "${OUTPUT_ROOT}/eval" -name summary.json | wc -l)
eval_costs=$(find "${OUTPUT_ROOT}/eval" -name costs.pt | wc -l)
if [[ "${final_checkpoints}" -ne 24 || "${eval_summaries}" -ne 24 || "${eval_costs}" -ne 24 ]]; then
  echo "incomplete scale artifacts: checkpoints=${final_checkpoints}, summaries=${eval_summaries}, costs=${eval_costs}" >&2
  exit 20
fi
test -f "${OUTPUT_ROOT}/summary/scale_effect_summary.json"
test -f "${OUTPUT_ROOT}/summary/scale_effect_per_seed.csv"
