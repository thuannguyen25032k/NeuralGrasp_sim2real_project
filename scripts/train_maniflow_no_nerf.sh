#!/usr/bin/env bash
# Train ManiFlow_BC WITHOUT neural rendering — for fast flow-matching sanity check.
#
# Disables the Gaussian Splatting renderer (method.use_neural_rendering=False)
# so that only the rectified-flow BC loss is active.  This lets you verify
# that the flow-matching head trains correctly before adding the NeRF cost.
#
# Run INSIDE the neuralgrasp Docker container:
#   docker exec -it neuralgrasp bash
#   cd /app && bash scripts/train_maniflow_no_nerf.sh [GPU_IDS] [PORT] [EXP_NAME]
#
# Usage:
#   bash scripts/train_maniflow_no_nerf.sh [GPU_IDS] [PORT] [EXP_NAME]
#
# Examples:
#   bash scripts/train_maniflow_no_nerf.sh 0 12346 maniflow_no_nerf_debug
#   bash scripts/train_maniflow_no_nerf.sh 0,1 12346

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
exp_name=${3:-"${method}_no_nerf_${addition_info}"}
replay_dir="${cur_dir}/replay/${exp_name}"

log_dir="${cur_dir}/logs/runs"
mkdir -p "${log_dir}"
log_file="${log_dir}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: ${log_file}"

# ---- Hyperparameters -------------------------------------------------------
# batch_size=8: the previous batch_size=4 gave only 16 BCE samples per step
# for the gripper head (B=4 × T=1 × nhand=1 × lv2=4).  Across 9 diverse
# multi-task training tasks that is < 2 grip samples per task per step,
# causing the logged L_grip to oscillate wildly (0.07 → 1.44 swings).
# batch_size=8 doubles per-step grip coverage and halves SGD variance on
# pos/rot velocity predictions — the main driver of late-training SR.
# VRAM usage: ~12 GB on RTX 6000 Ada (48 GB available) for the no-NeRF path.
batch_size=8
# Single easy task for a quick correctness check — change as needed.
tasks=[light_bulb_in,put_money_in_safe,place_wine_at_rack_location,put_groceries_in_cupboard,place_shape_in_shape_sorter,push_buttons,insert_onto_square_peg,stack_cups,place_cups]
demo=100

# Optimizer — reference 3D FlowMatch Actor: Adam lr=1e-4, constant schedule.
# LAMB + lr=0.005 (PerAct defaults) is 50× too high for a flow-matching
# Transformer and causes the loss to plateau after ~2k steps.
lr=0.0001
optimizer=adam
# CRITICAL: keep lr_scheduler=False and num_warmup_steps=0.
# Using lr_scheduler=True with num_warmup_steps=0 causes cosine decay from
# step 0 — by step 100k (33%) the LR is already halved, effectively
# under-training the entire run.  Constant LR matches the reference exactly.
lr_scheduler=False
num_warmup_steps=0

# Run for 300K steps (a reasonable check run; reference uses 300K for multi-task)
training_iterations=400010
save_freq=10000
precision='bf16'
# ----------------------------------------------------------------------------

echo "Starting ManiFlow (no-NeRF) training: method=${method}, seed=${seed}, num_devices=${#train_gpu_list[@]}, port=${port}, exp=${exp_name}" | tee -a "${log_file}"

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
    # ---- Disable neural rendering ------------------------------------------
    "method.use_neural_rendering=False"
    # ---- Optimizer (must match reference: Adam + constant lr=1e-4) ---------
    "method.lr=${lr}"
    "method.optimizer=${optimizer}"
    "method.lr_scheduler=${lr_scheduler}"
    "method.num_warmup_steps=${num_warmup_steps}"
    # ---- Training schedule -------------------------------------------------
    "framework.training_iterations=${training_iterations}"
    "framework.save_freq=${save_freq}"
    "framework.precision=${precision}"
)

echo "Running ManiFlow (no-NeRF) training in foreground." | tee -a "${log_file}"
echo "CUDA_VISIBLE_DEVICES=${train_gpu} ${train_cmd[*]}" | tee -a "${log_file}"

CUDA_VISIBLE_DEVICES="${train_gpu}" "${train_cmd[@]}" 2>&1 | tee -a "${log_file}"
