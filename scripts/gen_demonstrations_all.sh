# this script generates demonstrations for multiple tasks in parallel batches.
# example:
#       bash scripts/gen_demonstrations_all.sh
#
# Set PARALLEL_TASKS to control how many tasks run simultaneously.
# Each task already spawns --processes=4 workers internally, so:
#   total CoppeliaSim instances = PARALLEL_TASKS * 4
# Tune PARALLEL_TASKS based on available CPU cores and RAM.

# The recommended 10 tasks
ALL_TASK="close_jar open_drawer sweep_to_dustpan_of_size meat_off_grill turn_tap slide_block_to_color_target put_item_in_drawer reach_and_drag push_buttons stack_blocks stack_cups put_groceries_in_cupboard insert_onto_square_peg place_wine_at_rack_location put_money_in_safe"

PARALLEL_TASKS=4  # number of tasks to run concurrently
# System: Intel Core Ultra 9 285K (24 cores), 122 GB RAM, RTX 6000 Ada
# Each task spawns --processes=4 CoppeliaSim workers (~3 GB RAM, 1 core each)
# 4 tasks x 4 workers = 16 cores, ~48 GB RAM used — safe headroom on this machine

tasks=($ALL_TASK)
pids=()

for task in "${tasks[@]}"; do
    echo "### Generating demonstrations for task: $task"
    bash scripts/gen_demonstrations.sh "$task" &
    pids+=($!)

    # Once we have PARALLEL_TASKS running, wait for all of them to finish
    # before starting the next batch
    if (( ${#pids[@]} >= PARALLEL_TASKS )); then
        for pid in "${pids[@]}"; do
            wait "$pid"
            status=$?
            if [ $status -ne 0 ]; then
                echo "WARNING: A task process (PID $pid) exited with status $status"
            fi
        done
        pids=()
    fi
done

# Wait for any remaining tasks in the last (possibly partial) batch
for pid in "${pids[@]}"; do
    wait "$pid"
    status=$?
    if [ $status -ne 0 ]; then
        echo "WARNING: A task process (PID $pid) exited with status $status"
    fi
done

echo "### All tasks completed."
