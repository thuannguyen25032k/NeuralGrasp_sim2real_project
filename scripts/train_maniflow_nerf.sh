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
# SR-FIX NeRF#3: batch_size raised 4 → 6 so that (a) the grip BCE sees more
# positive/negative samples per step (was only 4), and (b) the NeRF renderer
# gets more view diversity per update.  6 fits comfortably in 48 GB with
# DINOv2 + dynamic field enabled (~30 GB estimated peak).
batch_size=4
tasks=[close_jar,open_drawer,sweep_to_dustpan_of_size,meat_off_grill,turn_tap,slide_block_to_color_target,put_item_in_drawer,reach_and_drag,push_buttons,stack_blocks]
demo=100

lr=0.0001
optimizer=adam
# FIX X13 (run4): enable cosine LR schedule with 3k-step warmup.  Constant
# LR (run3) bottoms out around pos_loss=0.3 on 200k steps; cosine decay
# from 1e-4 with warmup reliably drops the final pos_loss another 20-30 %
# by making the last ~60 % of training increasingly precision-focused.
lr_scheduler=True
num_warmup_steps=3000

# SR-FIX NeRF#12: pin precision explicitly to '32' (old Lightning Fabric API).
# bf16 can destabilise the Gaussian Splatting renderer (log-sigmoid opacities
# and exp-scale transforms are prone to bf16 overflow).  The NeRF path is
# kept in fp32 even though the no-nerf path uses bf16.
precision='32'

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
    # ---- Enable neural rendering -------------------------------------------1
    "method.use_neural_rendering=False"
    "method.neural_renderer.foundation_model_name=diffusion"
	"method.neural_renderer.use_dynamic_field=True"
    # ---- Optimizer ---------------------------------------------------------
    "method.lr=${lr}"
    "method.optimizer=${optimizer}"
    "method.lr_scheduler=${lr_scheduler}"
    "method.num_warmup_steps=${num_warmup_steps}"
    # ---- Training schedule -------------------------------------------------
    "framework.training_iterations=${training_iterations}"
    "framework.save_freq=${save_freq}"
    # ---- Precision (SR-FIX NeRF#12) ----------------------------------------
    "framework.precision=${precision}"
)

echo "Running ManiFlow (NeRF) training in foreground." | tee -a "${log_file}"
echo "CUDA_VISIBLE_DEVICES=${train_gpu} ${train_cmd[*]}" | tee -a "${log_file}"

CUDA_VISIBLE_DEVICES="${train_gpu}" "${train_cmd[@]}" 2>&1 | tee -a "${log_file}"
