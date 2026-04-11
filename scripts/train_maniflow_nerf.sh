#!/usr/bin/env bash
# Train ManiFlow_BC WITH Gaussian Splatting neural rendering auxiliary loss.
#
# Requires NeRF multi-view data stored alongside RLBench demos.
# Run INSIDE the neuralgrasp Docker container:
#   docker exec -it neuralgrasp bash
#   cd /app && bash scripts/train_maniflow_nerf.sh [GPU_IDS] [PORT] [EXP_NAME]
#
# Examples:
#   bash scripts/train_maniflow_nerf.sh 0     12345 maniflow_nerf_run1
#   bash scripts/train_maniflow_nerf.sh 0,1   12345 maniflow_nerf_run1

method="ManiFlow_BC"
seed="0"

train_gpu=${1:-"0"}
train_gpu_list=(${train_gpu//,/ })
port=${2:-"12345"}
use_wandb=True

cur_dir=$(pwd)
train_demo_path="${cur_dir}/data/train_data"
test_demo_path="${cur_dir}/data/test_data"

addition_info="$(date +%Y%m%d)"
exp_name=${3:-"${method}_nerf_${addition_info}"}
replay_dir="${cur_dir}/replay/${exp_name}"

log_dir="${cur_dir}/logs/runs"
mkdir -p "${log_dir}"
log_file="${log_dir}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: ${log_file}"

# ---- Hyperparameters -------------------------------------------------------
# NOTE: batch_size=1 is intentional for the nerf variant.
# The GS rasterizer processes one scene per GPU call; larger batches require
# the reshape(-1, 3) fix in models_embed.py (already applied).
batch_size=1
tasks=[put_money_in_safe]
demo=100
training_iterations=200010

# Flow-matching specific
denoise_timesteps=100
lv2_batch_size=4
flow_context_dim=256
flow_hidden_dim=512
flow_num_layers=4

# Neural rendering — ENABLED
use_neural_rendering=True
num_view_for_nerf=20
lambda_nerf=0.02          # outer weight on full nerf loss (see ManiFlow_BC.yaml)
lambda_bc=1.0             # weight on flow-matching BC loss
lambda_embed=0.5          # semantic embedding loss weight
lambda_dyna=0.1           # dynamic field loss weight
lambda_reg=0.0            # regularisation loss weight
render_freq=2000          # log a render image every N steps
foundation_model=diffusion  # 'diffusion' | 'none'
use_dynamic_field=True
# ----------------------------------------------------------------------------

echo "Starting ManiFlow+NeRF training: method=${method}, devices=${#train_gpu_list[@]}, exp=${exp_name}" | tee -a "${log_file}"

train_cmd=(
    "/app/.venv/bin/python" "train.py"
    "method=${method}"
    "rlbench.task_name=${exp_name}"
    "rlbench.demo_path=${train_demo_path}"
    "replay.path=${replay_dir}"
    "framework.start_seed=${seed}"
    "framework.use_wandb=${use_wandb}"
    "method.use_wandb=${use_wandb}"
    "framework.wandb_group=${exp_name}"
    "framework.wandb_name=${exp_name}"
    "ddp.num_devices=${#train_gpu_list[@]}"
    "replay.batch_size=${batch_size}"
    "ddp.master_port=${port}"
    "rlbench.tasks=${tasks}"
    "rlbench.demos=${demo}"
    "framework.training_iterations=${training_iterations}"
    # --- flow-matching head ---
    "method.denoise_timesteps=${denoise_timesteps}"
    "method.lv2_batch_size=${lv2_batch_size}"
    "method.flow_context_dim=${flow_context_dim}"
    "method.flow_hidden_dim=${flow_hidden_dim}"
    "method.flow_num_layers=${flow_num_layers}"
    # --- neural rendering ---
    "method.use_neural_rendering=${use_neural_rendering}"
    "method.num_view_for_nerf=${num_view_for_nerf}"
    "method.lambda_bc=${lambda_bc}"
    "method.neural_renderer.lambda_nerf=${lambda_nerf}"
    "method.neural_renderer.lambda_embed=${lambda_embed}"
    "method.neural_renderer.lambda_dyna=${lambda_dyna}"
    "method.neural_renderer.lambda_reg=${lambda_reg}"
    "method.neural_renderer.render_freq=${render_freq}"
    "method.neural_renderer.foundation_model_name=${foundation_model}"
    "method.neural_renderer.use_dynamic_field=${use_dynamic_field}"
)

echo "CUDA_VISIBLE_DEVICES=${train_gpu} ${train_cmd[*]}" | tee -a "${log_file}"

CUDA_VISIBLE_DEVICES="${train_gpu}" "${train_cmd[@]}" 2>&1 | tee -a "${log_file}"
train_exit=${PIPESTATUS[0]}

# Remove the step-0 checkpoint (random init, not useful)
rm -rf "logs/${exp_name}/seed${seed}/weights/0"

exit ${train_exit}
