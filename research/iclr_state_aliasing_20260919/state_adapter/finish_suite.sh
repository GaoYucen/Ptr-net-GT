#!/usr/bin/env bash
set -euo pipefail
driver="$1"
host="$2"
output="$3"
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="${GROUP_OPT_ROOT:-/workspace/计算群论}/environments/cuda-driver-535"
python_bin="${GROUP_OPT_ROOT:-/workspace/计算群论}/environments/groupopt-official-am-kool/bin/python"
for attempt in $(seq 1 360); do
    count=$(find "$output" -name training_summary.json | wc -l)
    if [ "$count" -eq 12 ]; then
        "$python_bin" -u "$driver" finalize --host "$host" --out "$output"
        "$python_bin" summarize_pilot.py --out "$output" >"$output/aggregate.log"
        exit 0
    fi
    sleep 10
done
echo 'Timed out waiting for all 12 training arms' >&2
exit 1
