# this script generates demonstrations for multiple tasks using a rolling worker pool.
# As soon as one task slot frees, the next task starts immediately.
# example:
#       bash scripts/gen_demonstrations_all.sh
#
# System: Intel Core Ultra 9 285K (24 cores, no HT), 122 GB RAM, RTX 6000 Ada (48 GB VRAM)
#   1 task = 2 generators x --processes=6 = 12 CoppeliaSim instances, ~24 GB RAM
#   MAX_PARALLEL_TASKS=2 -> 24 cores, ~48 GB RAM (safe), raise to 3 if RAM allows

set -euo pipefail

# ALL_TASK="close_jar open_drawer sweep_to_dustpan_of_size meat_off_grill turn_tap slide_block_to_color_target put_item_in_drawer reach_and_drag stack_blocks"
ALL_TASK="light_bulb_in put_money_in_safe place_wine_at_rack_location put_groceries_in_cupboard place_shape_in_shape_sorter push_buttons insert_onto_square_peg stack_cups place_cups"

MAX_PARALLEL_TASKS=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tasks=($ALL_TASK)
pids=()
names=()
FAILED_TASKS=()

for task in "${tasks[@]}"; do
    # Wait until a slot is free in the pool.
    while (( ${#pids[@]} >= MAX_PARALLEL_TASKS )); do
        new_pids=()
        new_names=()
        for idx in "${!pids[@]}"; do
            pid="${pids[$idx]}"
            tname="${names[$idx]}"
            if kill -0 "$pid" 2>/dev/null; then
                # Still running — keep in pool.
                new_pids+=("$pid")
                new_names+=("$tname")
            else
                # Finished — harvest exit status.
                wait "$pid" || true
                status=$?
                if [ $status -ne 0 ]; then
                    echo "WARNING: Task '${tname}' (PID ${pid}) exited with status ${status}"
                    FAILED_TASKS+=("$tname")
                else
                    echo "### Completed: ${tname}"
                fi
            fi
        done
        pids=("${new_pids[@]+"${new_pids[@]}"}")
        names=("${new_names[@]+"${new_names[@]}"}")
        # If pool still full, back off briefly before polling again.
        (( ${#pids[@]} >= MAX_PARALLEL_TASKS )) && sleep 1
    done

    echo "### Starting: ${task}"
    bash "${SCRIPT_DIR}/gen_demonstrations.sh" "$task" &
    pids+=($!)
    names+=("$task")
done

# Drain remaining jobs.
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    tname="${names[$idx]}"
    wait "$pid" || true
    status=$?
    if [ $status -ne 0 ]; then
        echo "WARNING: Task '${tname}' (PID ${pid}) exited with status ${status}"
        FAILED_TASKS+=("$tname")
    else
        echo "### Completed: ${tname}"
    fi
done

echo ""
echo "### All tasks finished."
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "FAILED tasks (${#FAILED_TASKS[@]}):"
    for t in "${FAILED_TASKS[@]}"; do echo "  - $t"; done
    exit 1
else
    echo "All ${#tasks[@]} tasks completed successfully."
fi
