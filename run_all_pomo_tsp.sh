#!/usr/bin/env bash
set -euo pipefail

# Run a sweep of POMO trainings on TSP-20/50/100 with different epochs,
# scheduling experiments onto multiple GPUs so that:
#   - each experiment runs on exactly one GPU (single-device Lightning)
#   - when any experiment finishes, the freed GPU immediately starts the next one
#
# Epoch schedules:
#   - TSP-20:  100, 50, 25, 10, 5, 2, 1, 0
#   - TSP-50:  100, 50, 25, 10, 5, 2, 1, 0
#   - TSP-100:  50, 25, 10, 5, 2, 1, 0  (no 100)
#
# Usage:
#   # Use GPUs 0..7 as a pool
#   CUDA_GPUS=0,1,2,3,4,5,6,7 ./run_all_pomo_tsp.sh

cd "$(dirname "$0")"

# Parse GPU pool
CUDA_GPUS=${CUDA_GPUS:-0}
IFS=',' read -r -a GPU_POOL <<< "${CUDA_GPUS}"
if [[ "${#GPU_POOL[@]}" -eq 0 ]]; then
  echo "[RUN_ALL] No GPUs parsed from CUDA_GPUS='${CUDA_GPUS}'" >&2
  exit 1
fi

echo "[RUN_ALL] Using GPU pool: ${GPU_POOL[*]}"

EPOCHS_TSP20=(100 50 25 10 5 2 1 0)
EPOCHS_TSP50=(100 50 25 10 5 2 1 0)
EPOCHS_TSP100=(50 25 10 5 2 1 0)

# Build task list: each task is "script epochs"
TASKS=()
for e in "${EPOCHS_TSP20[@]}"; do
  TASKS+=("train_pomo_tsp20.sh ${e}")
done
for e in "${EPOCHS_TSP50[@]}"; do
  TASKS+=("train_pomo_tsp50.sh ${e}")
done
for e in "${EPOCHS_TSP100[@]}"; do
  TASKS+=("train_pomo_tsp100.sh ${e}")
done

FIFO="pomo_tasks_$$"
rm -f "${FIFO}"
mkfifo "${FIFO}"

# Worker function: each worker binds to one GPU and pulls tasks from FIFO
start_worker() {
  local gpu_id="$1"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    echo "[RUN_ALL][GPU ${gpu_id}] Worker started"
    while read -r script epochs; do
      [[ -z "${script}" ]] && continue
      echo "[RUN_ALL][GPU ${gpu_id}] Starting ${script} (EPOCHS=${epochs})"
      RUN_FOREGROUND=1 EPOCHS="${epochs}" "./${script}"
      echo "[RUN_ALL][GPU ${gpu_id}] Finished ${script} (EPOCHS=${epochs})"
    done < "${FIFO}"
    echo "[RUN_ALL][GPU ${gpu_id}] Worker exiting (no more tasks)"
  ) &
}

# Start one worker per GPU
for gpu in "${GPU_POOL[@]}"; do
  start_worker "${gpu}"
done

# Feed tasks into FIFO
for t in "${TASKS[@]}"; do
  echo "${t}" > "${FIFO}"
done

# Close writers so readers see EOF when done
exec 3>&-

wait
rm -f "${FIFO}"

echo "[RUN_ALL] All POMO TSP sweeps completed."
