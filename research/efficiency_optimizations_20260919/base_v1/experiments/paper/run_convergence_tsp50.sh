#!/usr/bin/env bash

# 从主实验 seed=2468 checkpoint 续训，比较 Original/GroupOpt 的收敛曲线。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_ROOT="${SOURCE_ROOT:?必须提供 paper_main_v1/train 的绝对路径}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/paper_convergence_tsp50_v1}"
SEED="${SEED:-2468}"
TEST_SEED="${TEST_SEED:-20260902}"
TEST_SIZE="${TEST_SIZE:-10000}"
FAMILIES=(am ptrnet gpn pomo)
MODES=(native_original native_conditional_free)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/summary"

configure_family() {
  case "$1" in
    am)
      TRAINER="experiments/train.py"
      START_STEP=2000
      FINAL_STEP=10000
      BATCH_SIZE=512
      VALIDATION_SIZE=1024
      CHECKPOINT_EVERY=250
      EVAL_BATCH=64
      EVAL_STEPS=(2000 5000 10000)
      MODEL_ARGS=(--heads 8 --encoder-layers 3 --feed-forward-dim 512 --normalization batch)
      ;;
    ptrnet)
      TRAINER="experiments/train_ptrnet.py"
      START_STEP=2000
      FINAL_STEP=10000
      BATCH_SIZE=128
      VALIDATION_SIZE=1024
      CHECKPOINT_EVERY=250
      EVAL_BATCH=64
      EVAL_STEPS=(2000 5000 10000)
      MODEL_ARGS=(--encoder-layers 1)
      ;;
    gpn)
      TRAINER="experiments/train_gpn.py"
      START_STEP=2000
      FINAL_STEP=10000
      BATCH_SIZE=128
      VALIDATION_SIZE=1024
      CHECKPOINT_EVERY=250
      EVAL_BATCH=64
      EVAL_STEPS=(2000 5000 10000)
      MODEL_ARGS=(--encoder-layers 3)
      ;;
    pomo)
      TRAINER="experiments/train_pomo.py"
      START_STEP=1000
      FINAL_STEP=5000
      BATCH_SIZE=64
      VALIDATION_SIZE=64
      CHECKPOINT_EVERY=100
      EVAL_BATCH=8
      EVAL_STEPS=(1000 2500 5000)
      MODEL_ARGS=(--training-scheme pomo --pomo-size 8 --qkv-dim 16 --heads 8 --encoder-layers 6 --feed-forward-dim 512)
      ;;
    *)
      echo "不支持的宿主：$1" >&2
      return 2
      ;;
  esac
}

prepare_run() {
  local family="$1"
  local mode="$2"
  local source_dir="${SOURCE_ROOT}/${family}_tsp50_${mode}_seed${SEED}"
  local run_dir="${OUTPUT_ROOT}/train/${family}_tsp50_${mode}_seed${SEED}"
  if [[ ! -f "${source_dir}/config.json" || ! -f "${source_dir}/checkpoints/step-$(printf '%06d' "${START_STEP}").pt" ]]; then
    echo "缺少起始实验：${source_dir}" >&2
    return 20
  fi
  mkdir -p "${run_dir}/checkpoints"
  if [[ ! -f "${run_dir}/config.json" ]]; then
    cp "${source_dir}/config.json" "${run_dir}/config.json"
  fi
  if [[ ! -f "${run_dir}/metrics.jsonl" && -f "${source_dir}/metrics.jsonl" ]]; then
    cp "${source_dir}/metrics.jsonl" "${run_dir}/metrics.jsonl"
  fi
  for name in latest.pt best.pt "step-$(printf '%06d' "${START_STEP}").pt"; do
    if [[ ! -f "${run_dir}/checkpoints/${name}" ]]; then
      cp "${source_dir}/checkpoints/${name}" "${run_dir}/checkpoints/${name}"
    fi
  done
}

train_one() {
  local family="$1"
  local mode="$2"
  configure_family "${family}"
  prepare_run "${family}" "${mode}"
  local run_dir="${OUTPUT_ROOT}/train/${family}_tsp50_${mode}_seed${SEED}"
  local final_checkpoint
  final_checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${FINAL_STEP}")"
  if [[ -f "${final_checkpoint}" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" "${TRAINER}" \
    --base-mode "${mode}" \
    --output-dir "${run_dir}" \
    --resume "${run_dir}/checkpoints/latest.pt" \
    --graph-size 50 \
    --steps "${FINAL_STEP}" \
    --batch-size "${BATCH_SIZE}" \
    --validation-size "${VALIDATION_SIZE}" \
    --embedding-dim 128 \
    "${MODEL_ARGS[@]}" \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --seed "${SEED}" \
    --device cuda
}

for family in "${FAMILIES[@]}"; do
  pids=()
  for mode in "${MODES[@]}"; do
    train_one "${family}" "${mode}" \
      >"${OUTPUT_ROOT}/logs/train_${family}_${mode}_seed${SEED}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "${family} 收敛续训失败；保留日志与 checkpoint。" >&2
    exit 30
  fi

  configure_family "${family}"
  for mode in "${MODES[@]}"; do
    run_dir="${OUTPUT_ROOT}/train/${family}_tsp50_${mode}_seed${SEED}"
    for step in "${EVAL_STEPS[@]}"; do
      checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${step}")"
      evaluation_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/${family}/step${step}/${mode}"
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
        >"${OUTPUT_ROOT}/logs/eval_${family}_step${step}_${mode}.log" 2>&1
    done
  done
done

"${PYTHON_BIN}" experiments/paper/summarize_convergence_tsp50.py \
  --root "${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}" \
  --output-dir "${OUTPUT_ROOT}/summary" \
  --test-seed "${TEST_SEED}"
