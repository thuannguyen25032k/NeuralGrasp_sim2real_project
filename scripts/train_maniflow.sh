#!/usr/bin/env bash
# Train ManiFlow_BC (flow-matching policy + Gaussian Splatting renderer)
#
# Usage:
#   bash scripts/train_maniflow.sh [GPU_IDS] [PORT] [EXP_NAME]
#
# Examples:
#   bash scripts/train_maniflow.sh 0 12345 maniflow_run1
#   bash scripts/train_maniflow.sh 0   12345

method="ManiFlow_BC"
seed="0"

train_gpu=${1:-"0,1"}
train_gpu_list=(${train_gpu//,/ })
port=${2:-"12345"}
use_wandb=True

cur_dir=$(pwd)
train_demo_path="${cur_dir}/data/train_data"
test_demo_path="${cur_dir}/data/test_data"

addition_info="$(date +%Y%m%d)"
exp_name=${3:-"${method}_${addition_info}"}
replay_dir="${cur_dir}/replay/${exp_name}"

log_dir="${cur_dir}/logs/runs"
mkdir -p "${log_dir}"
log_file="${log_dir}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: ${log_file}"

# ---- Hyperparameters -------------------------------------------------------
batch_size=1
tasks=[light_bulb_in,put_money_in_safe,place_wine_at_rack_location,put_groceries_in_cupboard,place_shape_in_shape_sorter,push_buttons,insert_onto_square_peg,stack_cups,place_cups]
demo=20
lambda_dyna=0.1
lambda_reg=0.0
render_freq=2000

# Flow-matching specific
denoise_timesteps=100
flow_context_dim=256
flow_hidden_dim=512
flow_num_layers=4
# ----------------------------------------------------------------------------

# python_exec="/app/.venv/bin/python"
# if [ ! -x "${python_exec}" ]; then
# 	python_exec="python"
# fi

train_cmd=(
	"uv" "run" "train.py"
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
    "method.neural_renderer.render_freq=${render_freq}"
    "method.neural_renderer.lambda_embed=0.0"
    "method.neural_renderer.lambda_dyna=${lambda_dyna}"
    "method.neural_renderer.lambda_reg=${lambda_reg}"
    "method.neural_renderer.foundation_model_name=null"
    "method.neural_renderer.use_dynamic_field=True"
    "method.denoise_timesteps=${denoise_timesteps}"
    "method.flow_context_dim=${flow_context_dim}"
    "method.flow_hidden_dim=${flow_hidden_dim}"
    "method.flow_num_layers=${flow_num_layers}"
)

echo "Running ManiFlow training in foreground." | tee -a "${log_file}"
echo "CUDA_VISIBLE_DEVICES=${train_gpu} ${train_cmd[*]}" | tee -a "${log_file}"

CUDA_VISIBLE_DEVICES="${train_gpu}" "${train_cmd[@]}" 2>&1 | tee -a "${log_file}"
train_exit=${PIPESTATUS[0]}

# Remove initial checkpoint
rm -rf "logs/${exp_name}/seed${seed}/weights/0"

exit ${train_exit}
