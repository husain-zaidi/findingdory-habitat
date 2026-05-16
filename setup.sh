# Run from root directory of findingdory project folder
conda_env_name=${CONDA_ENV_NAME:-python310}
python_version=${PYTHON_VERSION:-3.10}
habitat_sim_spec=${HABITAT_SIM_SPEC:-habitat-sim=0.3.3.2026.05.16}
torch_version=${TORCH_VERSION:-2.8.0}
torchvision_version=${TORCHVISION_VERSION:-0.23.0}
torch_index_url=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
flash_attn_wheel=${FLASH_ATTN_WHEEL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.8-cp310-cp310-linux_x86_64.whl}

# Create conda env and install habitat-sim
if ! micromamba env list | awk '{print $1}' | grep -qx "$conda_env_name"; then
    micromamba create -n "$conda_env_name" "python=${python_version}" cmake=3.14.0 -y
fi
micromamba install -n "$conda_env_name" "$habitat_sim_spec" withbullet headless -c conda-forge -c aihabitat-nightly -c aihabitat -y

micromamba activate $conda_env_name

pip install \
    "torch==${torch_version}" \
    "torchvision==${torchvision_version}" \
    --index-url "$torch_index_url"

python - <<'PY'
import torch
import torchvision

print(f"Using torch {torch.__version__}")
print(f"Using torchvision {torchvision.__version__}")
PY

git submodule update --init --recursive
pip install -e third_party/habitat-lab/habitat-lab
pip install -e third_party/habitat-lab/habitat-baselines

# Install vc_models
pip install git+https://github.com/facebookresearch/eai-vc.git@main#subdirectory=vc_models

# Install the repo as a package
pip install -e .

# install dependencies for evaluating VLMs
pip install -e .[vlm_baseline]

# Install the flash-attn wheel that matches the Torch 2.8.0 + CUDA 12.8 stack above.
pip install --no-deps "$flash_attn_wheel"

python - <<'PY'
import flash_attn

print(f"Using flash_attn {flash_attn.__version__}")
PY

# install dependencies for the mapping baseline
pip install -e .[mapping_baseline]
