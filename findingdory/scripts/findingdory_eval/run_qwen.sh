#!/bin/bash

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

# Create symlink for data folder if it doesn't exist
if [ ! -L "data" ]; then
    ln -s findingdory/data data
fi

all_ep_ids=($(seq 1 100))
# chunk_size=768
chunk_size=250

# Get all episode IDs as comma-separated string
all_ep_ids_str=""
for j in $(seq 0 $((${#all_ep_ids[@]} - 1))); do
    all_ep_ids_str="${all_ep_ids_str}${all_ep_ids[$j]},"
done
all_ep_ids_str=${all_ep_ids_str%,} # Remove trailing comma

# Create base output folder
base_folder="findingdory/data/findingdory_outputs/qwen_mapper/slurm_logs"
folder_name="${base_folder}/batch_0"
report_dir="findingdory/data/findingdory_outputs/qwen_mapper/reports"
mkdir -p "$folder_name"
mkdir -p "$report_dir"

export HABITAT_SIM_LOG=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

# python -m debugpy --listen 5678 --wait-for-client findingdory/run_findingdory_eval.py \
python findingdory/run_findingdory_eval.py \
    --config-name=baseline/qwen_mapper.yaml \
    habitat.dataset.episode_ids=[1] \
    habitat.task.selected_instruction=0 \
    habitat_baselines.agent.config.vlm_temp_folder="data/findingdory_outputs/qwen_mapper/temp" \
    habitat_baselines.agent.config.output_folder="data/findingdory_outputs/qwen_mapper/logs" \
    habitat_baselines.agent.config.chunk_size=${chunk_size} \
    habitat_baselines.agent.config.subsample_frames=True

micromamba run -n findingdory python findingdory/scripts/findingdory_eval/view_metrics.py \
    findingdory/data/findingdory_outputs/qwen_mapper/logs \
    --output "${report_dir}/findingdory_metrics_view.html"
