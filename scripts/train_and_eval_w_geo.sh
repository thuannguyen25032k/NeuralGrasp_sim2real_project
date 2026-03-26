# example to run our ManiGaussian:
#       bash scripts/train_and_eval_w_geo.sh ManiGaussian_BC 0,1 12345 ${exp_name}
# Other examples:
#       bash scripts/train_and_eval_w_geo.sh GNFACTOR_BC 0,1 12345 ${exp_name}
#       bash scripts/train_and_eval_w_geo.sh PERACT_BC 0,1 12345 ${exp_name}

# set the method name
method=${1}
echo "method: ${method}"
# set the seed number
seed="0"
# set the gpu id for training. we use two gpus for training. you could also use one gpu.
train_gpu=${2:-"0,1"}
train_gpu_list=(${train_gpu//,/ })

# set the port for ddp training.
port=${3:-"12345"}
# you could enable/disable wandb by this.
use_wandb=True

cur_dir=$(pwd)
train_demo_path="${cur_dir}/data/train_data"
test_demo_path="${cur_dir}/data/test_data"

# we set experiment name as method+date. you could specify it as you like.
addition_info="$(date +%Y%m%d)"
exp_name=${4:-"${method}_${addition_info}"}
replay_dir="${cur_dir}/replay/${exp_name}"

# Log output to a file (no tmux). This keeps behavior similar: you still get a persistent log.
log_dir="${cur_dir}/logs/runs"
mkdir -p "${log_dir}"
log_file="${log_dir}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
echo "log file: ${log_file}"

#######
# override hyper-params in config.yaml
#######
batch_size=1
tasks=[close_jar,open_drawer,sweep_to_dustpan_of_size,meat_off_grill,turn_tap,slide_block_to_color_target,put_item_in_drawer,reach_and_drag,push_buttons,stack_blocks]
demo=20
render_freq=2000

python_exec="/app/.venv/bin/python"
if [ ! -x "${python_exec}" ]; then
	python_exec="python"
fi

train_cmd=(
	"${python_exec}" train.py
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
)

echo "Running training in foreground (no tmux)." | tee -a "${log_file}"
echo "CUDA_VISIBLE_DEVICES=${train_gpu} ${train_cmd[*]}" | tee -a "${log_file}"

# Run in foreground so Docker/process managers track status; also stream to log.
CUDA_VISIBLE_DEVICES="${train_gpu}" "${train_cmd[@]}" 2>&1 | tee -a "${log_file}"
train_exit=${PIPESTATUS[0]}

# remove 0.ckpt
rm -rf logs/${exp_name}/seed${seed}/weights/0

exit ${train_exit}
