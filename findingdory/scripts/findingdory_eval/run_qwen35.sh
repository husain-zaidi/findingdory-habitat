#!/bin/bash

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

REPORT_STAMP="${REPORT_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
report_dir="findingdory/data/findingdory_outputs/qwen_mapper/reports/${REPORT_STAMP}"
EPISODE_START="${EPISODE_START:-1}"
EPISODE_END="${EPISODE_END:-$EPISODE_START}"
INSTRUCTIONS_TO_EVALUATE="${INSTRUCTIONS_TO_EVALUATE:--1}"

all_ep_ids=($(seq "$EPISODE_START" "$EPISODE_END"))
all_ep_ids_str=""
for j in $(seq 0 $((${#all_ep_ids[@]} - 1))); do
    all_ep_ids_str="${all_ep_ids_str}${all_ep_ids[$j]},"
done
all_ep_ids_str=${all_ep_ids_str%,}

mkdir -p "$report_dir"

# Create symlink for data folder if it doesn't exist
if [ ! -L "data" ]; then
    ln -s findingdory/data data
fi

echo "Episode IDs: [${all_ep_ids_str}]"
echo "Evaluation task queue: [${INSTRUCTIONS_TO_EVALUATE}]"

python findingdory/run_findingdory_eval.py \
  --config-name=baseline/qwen35_mapper.yaml \
  "habitat.dataset.episode_ids=[${all_ep_ids_str}]" \
  "habitat.task.instructions_to_evaluate=[${INSTRUCTIONS_TO_EVALUATE}]"

python findingdory/scripts/findingdory_eval/view_metrics.py \
    findingdory/data/findingdory_outputs/qwen35_mapper/logs \
    --output "${report_dir}/findingdory_metrics_view.html"

echo "Reports written to ${report_dir}"
