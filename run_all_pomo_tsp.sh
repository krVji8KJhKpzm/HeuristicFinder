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

declare -A GPU_BUSY

num_tasks=${#TASKS[@]}
task_idx=0

echo "[RUN_ALL] Total tasks: ${num_tasks}"

while (( task_idx < num_tasks )); do
  assigned=0
  for gpu in "${GPU_POOL[@]}"; do
    # Check if this GPU is free (no running PID or PID finished)
    pid="${GPU_BUSY[$gpu]:-}"
    if [[ -n "${pid}" ]]; then
      if ! kill -0 "${pid}" 2>/dev/null; then
        GPU_BUSY[$gpu]=""
      fi
    fi

    if [[ -z "${GPU_BUSY[$gpu]:-}" ]]; then
      # Assign next task to this GPU
      task="${TASKS[$task_idx]}"
      ((task_idx++))
      script=${task%% *}
      epochs=${task##* }
      echo "[RUN_ALL][GPU ${gpu}] Starting ${script} (EPOCHS=${epochs})"
      CUDA_VISIBLE_DEVICES="${gpu}" RUN_FOREGROUND=1 EPOCHS="${epochs}" "./${script}" &
      GPU_BUSY[$gpu]=$!
      assigned=1
      # Break to let outer loop re-check from first GPU with updated state
      break
    fi
  done

  # If no GPU was free in this pass, wait for any child to finish
  if (( !assigned )); then
    # Wait for at least one child; ignore exit status
    wait -n || true
  fi
done

# Wait for all remaining trainings to finish
wait || true

echo "[RUN_ALL] All POMO TSP sweeps completed."
