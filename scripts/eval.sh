# this script is for evaluating a given checkpoint.
# NOTE: method is determined automatically from logs/<exp_name>/seed0/config.yaml —
#       method_name below is only passed to satisfy Hydra; eval.py ignores it.
#
# example to evaluate ManiFlow (no-NeRF, 9 tasks):
#       bash scripts/eval.sh ManiFlow_BC ${exp_name} 0
# example to evaluate ManiFlow (NeRF, 9 tasks):
#       bash scripts/eval.sh ManiFlow_BC ${exp_name} 0
# Other examples:
#       bash scripts/eval.sh ManiGaussian_BC ${exp_name} 0
#       bash scripts/eval.sh GNFACTOR_BC ${exp_name} 0
#       bash scripts/eval.sh PERACT_BC ${exp_name} 0

# some params specified by user
method_name=$1
exp_name=$2

# set the seed number
seed="0"
# set the gpu id for evaluation. we use one gpu for parallel evaluation.
eval_gpu=${3:-"0"}

cur_dir=$(pwd)
train_demo_path="${cur_dir}/data/train_data"
test_demo_path="${cur_dir}/data/test_data"
tasks="[close_jar,open_drawer,sweep_to_dustpan_of_size,meat_off_grill,turn_tap,slide_block_to_color_target,put_item_in_drawer,reach_and_drag,push_buttons,stack_blocks]"   # specify the task(s) to evaluate on; e.g. "[put_money_in_safe,place_cups]" or "all" for all tasks in the demo path
use_split='test'    # or 'train' for debugging

starttime=`date +'%Y-%m-%d %H:%M:%S'`

if [ "${use_split}" == "train" ]; then
    echo "eval on train set"
    # eval on train set
    CUDA_VISIBLE_DEVICES=${eval_gpu} xvfb-run -a /app/.venv/bin/python eval.py \
        method.name=${method_name} \
        rlbench.task_name=${exp_name} \
        rlbench.tasks=${tasks} \
        rlbench.demo_path=${train_demo_path} \
        framework.start_seed=${seed} \
        framework.eval_episodes=25

else
    echo "eval on test set"
    # eval on test set
    CUDA_VISIBLE_DEVICES=${eval_gpu} xvfb-run -a /app/.venv/bin/python eval.py \
        method.name=${method_name} \
        rlbench.task_name=${exp_name} \
        rlbench.tasks=${tasks} \
        rlbench.demo_path=${test_demo_path} \
        framework.start_seed=${seed} \
        framework.eval_episodes=25
fi

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "eclipsed time "$((end_seconds-start_seconds))"s"
