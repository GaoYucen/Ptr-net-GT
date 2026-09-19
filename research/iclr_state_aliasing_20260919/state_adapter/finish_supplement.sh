#!/usr/bin/env bash
set -euo pipefail
host="$1"
output="$2"
decoder="${3:-false}"
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="${GROUP_OPT_ROOT:-/workspace/计算群论}/environments/cuda-driver-535"
python_bin="${GROUP_OPT_ROOT:-/workspace/计算群论}/environments/groupopt-official-am-kool/bin/python"
extra=()
if [ "$decoder" = true ]; then extra+=(--decoder); fi
for attempt in $(seq 1 360); do
    if [ -f "$output/aggregate.json" ]; then
        "$python_bin" source_variability.py --host "$host" --out "$output" "${extra[@]}"
        "$python_bin" posthoc_diagnostics.py --host "$host" --out "$output" "${extra[@]}"
        exit 0
    fi
    sleep 10
done
echo 'Timed out waiting for frozen final-test aggregate' >&2
exit 1
