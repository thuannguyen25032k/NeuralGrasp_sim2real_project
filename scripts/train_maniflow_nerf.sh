#!/usr/bin/env bash
# Train ManiFlow_BC WITH neural rendering (Gaussian Splatting auxiliary loss).
#
# Enables the Gaussian Splatting renderer (method.use_neural_rendering=True)
# so that both the rectified-flow BC loss and the NeRF reconstruction loss
# are active.  Run train_maniflow_no_nerf.sh first to verify the flow head
# trains correctly, then use this script for full training.
#
# Run INSIDE the neuralgrasp Docker container:
#   docker exec -it neuralgrasp bash
#   cd /app && bash scripts/train_maniflow_nerf.sh [GPU_IDS] [PORT] [EXP_NAME]
#
# Usage:
#   bash scripts/train_maniflow_nerf.sh [GPU_IDS] [PORT] [EXP_NAME]
#
# Examples:
#   bash scripts/train_maniflow_nerf.sh 0 12346 maniflow_nerf_debug
#   bash scripts/train_maniflow_nerf.sh 0,1 12346

method="ManiFlow_BC"
seed="0"

train_gpu=${1:-"0"}
train_gpu_list=(${train_gpu//,/ })
port=${2:-"12346"}
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
batch_size=8
tasks=[put_money_in_safe]
demo=100

lr=0.0001
optimizer=adam
lr_scheduler=False

# lv2_batch_size: inner noise re-sampling loop — 3 gives 3× gradient diversity
# with negligible cost (scene encoding is reused).
lv2_batch_size=3

training_iterations=200010
save_freq=10000
# ----------------------------------------------------------------------------

echo "Starting ManiFlow (NeRF) training: method=${method}, seed=${seed}, num_devices=${#train_gpu_list[@]}, port=${port}, exp=${exp_name}" | tee -a "${log_file}"

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
    # ---- Enable neural rendering -------------------------------------------
    "method.use_neural_rendering=True"
    # ---- Optimizer ---------------------------------------------------------
    "method.lr=${lr}"
    "method.optimizer=${optimizer}"
    "method.lr_scheduler=${lr_scheduler}"
    "method.num_warmup_steps=0"
    # ---- Flow-matching hyperparameters -------------------------------------
    "method.lv2_batch_size=${lv2_batch_size}"
    # ---- Training schedule -------------------------------------------------
    "framework.training_iterations=${training_iterations}"
    "framework.save_freq=${save_freq}"
)

echo "Running ManiFlow (NeRF) training in foreground." | tee -a "${log_file}"
echo "CUDA_VISIBLE_DEVICES=${train_gpu} ${train_cmd[*]}" | tee -a "${log_file}"

CUDA_VISIBLE_DEVICES="${train_gpu}" "${train_cmd[@]}" 2>&1 | tee -a "${log_file}"
