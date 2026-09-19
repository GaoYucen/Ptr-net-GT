#!/usr/bin/env bash
# Pre-registered TSP50 screening queue.  One GPU index is required.
set -euo pipefail

gpu=${1:?usage: run_pair_validation_queue.sh GPU_INDEX OUTPUT_ROOT}
out_root=${2:?usage: run_pair_validation_queue.sh GPU_INDEX OUTPUT_ROOT}
repo_root=$(cd "$(dirname "$0")/.." && pwd)
python_bin=${PAIR_VALIDATION_PYTHON:-python}
cuda_driver_path=${PAIR_VALIDATION_CUDA_DRIVER_PATH:-/cuda_fix}

for method in single_opt seq2 pair_add pair_interaction; do
  for seed in 12031 12037 12041 12043 12047 12053 12059 12071 12077 12083; do
    out="$out_root/$method-seed$seed"
    test -e "$out" && continue
    LD_LIBRARY_PATH="$cuda_driver_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo_root/src" \
      "$python_bin" "$repo_root/experiments/train_pair_screen.py" \
      --out "$out" --method "$method" --seed "$seed" --nodes 50 --steps 2000 \
      --batch 8 --rollouts 8 --eval-rollouts 1 --eval-batch 64 \
      --validation-size 512 --test-size 10000 --distribution uniform \
      --eval-every 200 --log-every 25 --lr 1e-4 --max-seconds 14400 --device cuda:0 \
      --memory-fraction .30
  done
done
