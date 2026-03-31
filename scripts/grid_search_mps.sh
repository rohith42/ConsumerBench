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

RUN_TS="$(date +%Y%m%d_%H%M%S)"
GENERATED_CONFIG_DIR="$REPO_DIR/configs/generated/grid_search_mps_workflow_tally_concurrent_$RUN_TS"
RUN_LOG_DIR="$GENERATED_CONFIG_DIR/run_logs"
SUMMARY_CSV="$GENERATED_CONFIG_DIR/summary.csv"
mkdir -p "$GENERATED_CONFIG_DIR" "$RUN_LOG_DIR"

imagegen_mps_start=75
imagegen_mps_end=85
livecaption_mps_start=40
livecaption_mps_end=45
chatbot_mps=6

total_imagegen=$((imagegen_mps_end - imagegen_mps_start + 1))
total_livecaption=$((livecaption_mps_end - livecaption_mps_start + 1))
TOTAL_RUNS=$((total_imagegen * total_livecaption))
RUN_INDEX=0

BEST_TOTAL_TIME=""
BEST_IMAGEGEN_MPS=""
BEST_LIVECAPTION_MPS=""
BEST_CONFIG=""
BEST_RESULTS_DIR=""

FAILED_RUNS=()

START_EPOCH="$(date +%s)"

echo "Starting brute-force MPS search for workflow_tally_3_concurrent"
echo "Base config: $BASE_CONFIG"
echo "Generated configs: $GENERATED_CONFIG_DIR"
echo "ImageGen mps range: [$imagegen_mps_start, $imagegen_mps_end]"
echo "LiveCaptions mps range: [$livecaption_mps_start, $livecaption_mps_end]"
echo "Chatbot mps (fixed): $chatbot_mps"
echo "Total runs: $TOTAL_RUNS"
echo

echo "run_index,imagegen_mps,livecaptions_mps,chatbot_mps,total_execution_time_seconds,results_dir,config_path,status" > "$SUMMARY_CSV"

for imagegen_mps in $(seq "$imagegen_mps_start" "$imagegen_mps_end"); do
    for livecaption_mps in $(seq "$livecaption_mps_start" "$livecaption_mps_end"); do
        RUN_INDEX=$((RUN_INDEX + 1))

        CONFIG_NAME="workflow_mps_img${imagegen_mps}_lv${livecaption_mps}.yml"
        CONFIG_PATH="$GENERATED_CONFIG_DIR/$CONFIG_NAME"
        RUN_LOG_PATH="$RUN_LOG_DIR/${CONFIG_NAME%.yml}.log"

        python3 - "$BASE_CONFIG" "$CONFIG_PATH" "$imagegen_mps" "$livecaption_mps" "$chatbot_mps" <<'PY'
import sys
import yaml

base_config, output_config, imagegen_mps, livecaption_mps, chatbot_mps = sys.argv[1:]

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

# Explicitly keep scheduler out of generated configs.
data.pop("scheduler", None)
data["imagegen1"]["type"] = "ImageGenServer"
data["lv1"]["type"] = "LiveCaptionsHF"
data["chat1"]["type"] = "ChatbotHF"
data["imagegen1"]["device"] = "gpu"
data["lv1"]["device"] = "gpu"
data["chat1"]["device"] = "gpu"
data["imagegen1"]["mps"] = int(imagegen_mps)
data["lv1"]["mps"] = int(livecaption_mps)
data["chat1"]["mps"] = int(chatbot_mps)

with open(output_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY

        echo "[$RUN_INDEX/$TOTAL_RUNS] imagegen1.mps=$imagegen_mps lv1.mps=$livecaption_mps chat1.mps=$chatbot_mps"

        if ! bash "$RUN_BENCHMARK" "$CONFIG_PATH" "$NSIGHT" | tee "$RUN_LOG_PATH"; then
            echo "Run failed: $CONFIG_NAME"
            FAILED_RUNS+=("$CONFIG_NAME")
            echo "${RUN_INDEX},${imagegen_mps},${livecaption_mps},${chatbot_mps},,,$CONFIG_PATH,failed" >> "$SUMMARY_CSV"
            echo
            continue
        fi

        parse_output="$(python3 - "$RUN_LOG_PATH" <<'PY'
import re
import sys

log_path = sys.argv[1]

total_time = ""
results_dir = ""

with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m_time = re.search(r"Total execution time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds", line)
        if m_time:
            total_time = m_time.group(1)
        m_dir = re.search(r"Results are saved in\s+(.+)\s*$", line)
        if m_dir:
            results_dir = m_dir.group(1).strip()

print(total_time)
print(results_dir)
PY
)"

        total_time="$(printf "%s\n" "$parse_output" | sed -n '1p')"
        results_dir="$(printf "%s\n" "$parse_output" | sed -n '2p')"

        status="ok"
        if [[ -z "$total_time" ]]; then
            status="parse_failed"
            FAILED_RUNS+=("${CONFIG_NAME}(missing_total_execution_time)")
            echo "Warning: could not parse total execution time from $RUN_LOG_PATH"
        fi

        echo "${RUN_INDEX},${imagegen_mps},${livecaption_mps},${chatbot_mps},${total_time},${results_dir},${CONFIG_PATH},${status}" >> "$SUMMARY_CSV"

        if [[ "$status" == "ok" ]]; then
            if [[ -z "$BEST_TOTAL_TIME" || "$(awk "BEGIN {print ($total_time < $BEST_TOTAL_TIME)}")" -eq 1 ]]; then
                BEST_TOTAL_TIME="$total_time"
                BEST_IMAGEGEN_MPS="$imagegen_mps"
                BEST_LIVECAPTION_MPS="$livecaption_mps"
                BEST_CONFIG="$CONFIG_PATH"
                BEST_RESULTS_DIR="$results_dir"
            fi
        fi

        echo
    done
done

END_EPOCH="$(date +%s)"
ELAPSED_SECONDS=$((END_EPOCH - START_EPOCH))

echo "Brute-force MPS search finished in ${ELAPSED_SECONDS}s"
echo "Generated configs: $GENERATED_CONFIG_DIR"
echo "Summary CSV: $SUMMARY_CSV"

if [[ -n "$BEST_TOTAL_TIME" ]]; then
    echo
    echo "Best setting found (minimum total execution time):"
    echo "  imagegen1.mps=$BEST_IMAGEGEN_MPS"
    echo "  lv1.mps=$BEST_LIVECAPTION_MPS"
    echo "  total_execution_time=${BEST_TOTAL_TIME}s"
    echo "  config=$BEST_CONFIG"
    if [[ -n "$BEST_RESULTS_DIR" ]]; then
        echo "  results=$BEST_RESULTS_DIR"
    fi
else
    echo
    echo "No successful runs with parseable total execution time were found."
fi

if (( ${#FAILED_RUNS[@]} > 0 )); then
    echo
    echo "Failed runs (${#FAILED_RUNS[@]}):"
    for failed in "${FAILED_RUNS[@]}"; do
        echo "  - $failed"
    done
    exit 1
fi

echo
echo "All runs completed successfully"