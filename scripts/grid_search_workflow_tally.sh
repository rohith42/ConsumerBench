#!/bin/bash

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
RUN_BENCHMARK="$SCRIPTS_DIR/run_benchmark.sh"

BASE_CONFIG="${1:-$REPO_DIR/configs/workflow_tally_concurrent.yml}"
NSIGHT="${2:-0}"

if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "Base config not found: $BASE_CONFIG"
    exit 1
fi

if [[ ! -x "$RUN_BENCHMARK" ]]; then
    echo "Benchmark runner not executable: $RUN_BENCHMARK"
    echo "Run: chmod +x $RUN_BENCHMARK"
    exit 1
fi

if [[ "$NSIGHT" != "0" && "$NSIGHT" != "1" ]]; then
    echo "Invalid nsight flag: $NSIGHT (must be 0 or 1)"
    exit 1
fi

schedulers=("tally" "tgs")
priorities=(1 2)

RUN_TS="$(date +%Y%m%d_%H%M%S)"
GENERATED_CONFIG_DIR="$REPO_DIR/configs/generated/grid_search_workflow_tally_3_concurrent_$RUN_TS"
mkdir -p "$GENERATED_CONFIG_DIR"

TOTAL_RUNS=$(( ${#schedulers[@]} * ${#priorities[@]} * ${#priorities[@]} * ${#priorities[@]} ))
RUN_INDEX=0
FAILED_RUNS=()

START_EPOCH="$(date +%s)"

echo "Starting grid search for workflow_tally_3_concurrent"
echo "Base config: $BASE_CONFIG"
echo "Generated configs: $GENERATED_CONFIG_DIR"
echo "Schedulers: ${schedulers[*]}"
echo "Priorities: ${priorities[*]}"
echo "Total runs: $TOTAL_RUNS"
echo

for scheduler in "${schedulers[@]}"; do
    for imagegen_priority in "${priorities[@]}"; do
        for livecaption_priority in "${priorities[@]}"; do
            for chatbot_priority in "${priorities[@]}"; do
                RUN_INDEX=$((RUN_INDEX + 1))

                CONFIG_NAME="workflow_${scheduler}_img${imagegen_priority}_lv${livecaption_priority}_chat${chatbot_priority}.yml"
                CONFIG_PATH="$GENERATED_CONFIG_DIR/$CONFIG_NAME"

                python3 - "$BASE_CONFIG" "$CONFIG_PATH" "$scheduler" "$imagegen_priority" "$livecaption_priority" "$chatbot_priority" <<'PY'
import sys
import yaml

base_config, output_config, scheduler, imagegen_priority, livecaption_priority, chatbot_priority = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

if not isinstance(data, dict):
    raise SystemExit("Base config must be a top-level YAML mapping")

required_apps = ("imagegen1", "lv1", "chat1")
for app in required_apps:
    if app not in data or not isinstance(data[app], dict):
        raise SystemExit(f"Missing or invalid app section: {app}")

required_fields = {
    "imagegen1": ("model", "api_port"),
    "lv1": ("model", "api_port", "client_command_file"),
    "chat1": ("model", "api_port"),
}

for app, fields in required_fields.items():
    for field in fields:
        value = data[app].get(field)
        if value is None or value == "":
            raise SystemExit(f"Missing required field '{field}' in app section '{app}'")

data["scheduler"] = scheduler
data["imagegen1"]["priority"] = int(imagegen_priority)
data["lv1"]["priority"] = int(livecaption_priority)
data["chat1"]["priority"] = int(chatbot_priority)

with open(output_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY

                echo "[$RUN_INDEX/$TOTAL_RUNS] scheduler=$scheduler imagegen1=$imagegen_priority lv1=$livecaption_priority chat1=$chatbot_priority"
                if ! bash "$RUN_BENCHMARK" "$CONFIG_PATH" "$NSIGHT"; then
                    echo "Run failed: $CONFIG_NAME"
                    FAILED_RUNS+=("$CONFIG_NAME")
                fi
                echo
            done
        done
    done
done

END_EPOCH="$(date +%s)"
ELAPSED_SECONDS=$((END_EPOCH - START_EPOCH))

echo "Grid search finished in ${ELAPSED_SECONDS}s"
echo "Generated configs: $GENERATED_CONFIG_DIR"

if (( ${#FAILED_RUNS[@]} > 0 )); then
    echo "Failed runs (${#FAILED_RUNS[@]}):"
    for failed in "${FAILED_RUNS[@]}"; do
        echo "  - $failed"
    done
    exit 1
fi

echo "All runs completed successfully"
