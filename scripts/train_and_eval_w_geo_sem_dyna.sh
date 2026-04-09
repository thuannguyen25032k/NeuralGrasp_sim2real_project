# example to run our ManiGaussian:
#       bash scripts/train_and_eval_w_geo_sem_dyna.sh ManiGaussian_BC 0 12345 manigaussian_run1
# this file does not support other examples.

# set the method name
method=${1}

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

#######
# override hyper-params in config.yaml
#######
batch_size=1
tasks=[light_bulb_in,put_money_in_safe,place_wine_at_rack_location,put_groceries_in_cupboard,place_shape_in_shape_sorter,push_buttons,insert_onto_square_peg,stack_cups,place_cups]
demo=20
lambda_embed=0.01
lambda_dyna=0.1
lambda_reg=0.0
render_freq=5000

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
	"method.neural_renderer.lambda_embed=${lambda_embed}"
	"method.neural_renderer.lambda_dyna=${lambda_dyna}"
	"method.neural_renderer.lambda_reg=${lambda_reg}"
	"method.neural_renderer.foundation_model_name=diffusion"
	"method.neural_renderer.use_dynamic_field=True"
	"framework.num_workers=20"
	"replay.max_parallel_processes=20"
)

echo "Running training in foreground (no tmux)." 
echo "CUDA_VISIBLE_DEVICES=${train_gpu} ${train_cmd[*]}" 

CUDA_VISIBLE_DEVICES="${train_gpu}" "${train_cmd[@]}"
train_exit=${PIPESTATUS[0]}

# remove 0.ckpt
rm -rf logs/${exp_name}/seed${seed}/weights/0

exit ${train_exit}
