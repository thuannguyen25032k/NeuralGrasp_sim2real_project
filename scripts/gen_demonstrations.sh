# this script generates demonstrations for a given task, for both training and evaluation.
# The nerf (train) and standard (test) generators write to independent paths,
# so both are launched in parallel and the script waits for both to finish.
# example:
#       bash scripts/gen_demonstrations.sh open_drawer

set -euo pipefail

task=${1:?Usage: gen_demonstrations.sh <task_name>}

# Resolve paths relative to this script so it can be called from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../third_party/RLBench/tools"

echo "[${task}] Starting nerf_dataset_generator (train data)..."
xvfb-run -a python nerf_dataset_generator.py \
    --tasks="${task}" \
    --save_path="../../../data/train_data" \
    --image_size=128,128 \
    --renderer=opengl \
    --episodes_per_task=20 \
    --processes=4 \
    --all_variations=True &
NERF_PID=$!

echo "[${task}] Starting dataset_generator (test data)..."
xvfb-run -a python dataset_generator.py \
    --tasks="${task}" \
    --save_path="../../../data/test_data" \
    --image_size=128,128 \
    --renderer=opengl \
    --episodes_per_task=25 \
    --processes=4 \
    --all_variations=True &
DATASET_PID=$!

# Wait for both generators; propagate any failure back to the caller.
FAILED=0
wait "${NERF_PID}"    || { echo "ERROR: nerf_dataset_generator failed for task: ${task}"; FAILED=1; }
wait "${DATASET_PID}" || { echo "ERROR: dataset_generator failed for task: ${task}";      FAILED=1; }

exit "${FAILED}"
