#!/bin/bash

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

# Create symlink for data folder if it doesn't exist
if [ ! -L "data" ]; then
    ln -s findingdory/data data
fi

DATASET_PATH="findingdory/data/datasets/findingdory-habitat/findingdory/val/episodes.json.gz"

if [ ! -f "$DATASET_PATH" ]; then
    echo "Dataset not found at $DATASET_PATH"
    exit 1
fi

num_episodes=$(python - <<'PY'
import gzip
import json

dataset_path = "findingdory/data/datasets/findingdory-habitat/findingdory/val/episodes.json.gz"
with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
    payload = json.load(f)
print(len(payload.get("episodes", [])))
PY
)
if [ "$num_episodes" -le 0 ]; then
    echo "Could not determine number of episodes from $DATASET_PATH"
    exit 1
fi

EPISODE_START="${EPISODE_START:-1}"
EPISODE_END="${EPISODE_END:-$EPISODE_START}"
INSTRUCTIONS_TO_EVALUATE="${INSTRUCTIONS_TO_EVALUATE:--1}"
REPORT_STAMP="${REPORT_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
all_ep_ids=($(seq "$EPISODE_START" "$EPISODE_END"))
# chunk_size=768
chunk_size=250

# Get all episode IDs as comma-separated string
all_ep_ids_str=""
for j in $(seq 0 $((${#all_ep_ids[@]} - 1))); do
    all_ep_ids_str="${all_ep_ids_str}${all_ep_ids[$j]},"
done
all_ep_ids_str=${all_ep_ids_str%,} # Remove trailing comma

echo "Running Qwen mapper for ${num_episodes} dataset episodes"
echo "Episode IDs: [${all_ep_ids_str}]"
echo "Evaluation task queue: [${INSTRUCTIONS_TO_EVALUATE}]"

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA build: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    raise SystemExit("CUDA is not available to PyTorch in this environment.")
PY

# Create base output folder
base_folder="findingdory/data/findingdory_outputs/qwen_mapper/slurm_logs"
folder_name="${base_folder}/batch_0"
report_dir="findingdory/data/findingdory_outputs/qwen_mapper/reports/${REPORT_STAMP}"
mkdir -p "$folder_name"
mkdir -p "$report_dir"

export HABITAT_SIM_LOG=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

# python -m debugpy --listen 5678 --wait-for-client findingdory/run_findingdory_eval.py \
python findingdory/run_findingdory_eval.py \
    --config-name=baseline/qwen_mapper.yaml \
    "habitat.dataset.episode_ids=[${all_ep_ids_str}]" \
    "habitat.task.instructions_to_evaluate=[${INSTRUCTIONS_TO_EVALUATE}]" \
    habitat_baselines.agent.config.vlm_temp_folder="data/findingdory_outputs/qwen_mapper/temp" \
    habitat_baselines.agent.config.output_folder="data/findingdory_outputs/qwen_mapper/logs" \
    habitat_baselines.agent.config.chunk_size=${chunk_size} \
    habitat_baselines.agent.config.subsample_frames=True \
    "$@"

python findingdory/scripts/findingdory_eval/view_metrics.py \
    findingdory/data/findingdory_outputs/qwen_mapper/logs \
    --output "${report_dir}/findingdory_metrics_view.html"

echo "Reports written to ${report_dir}"
