#!/usr/bin/env bash
set -euo pipefail

# Run a sweep of POMO trainings on TSP-20/50/100 with different epochs, sequentially.
#
# Epoch schedules:
#   - TSP-20:  100, 50, 25, 10, 5, 2, 1, 0
#   - TSP-50:  100, 50, 25, 10, 5, 2, 1, 0
#   - TSP-100:  50, 25, 10, 5, 2, 1, 0  (no 100)
#
# Usage:
#   # Run all on GPU 0
#   CUDA_GPUS=0 ./run_all_pomo_tsp.sh
#   # Or specify a different GPU set
#   CUDA_GPUS=1 ./run_all_pomo_tsp.sh

cd "$(dirname "$0")"

CUDA_GPUS=${CUDA_GPUS:-0}
export CUDA_GPUS

EPOCHS_TSP20=(100 50 25 10 5 2 1 0)
EPOCHS_TSP50=(100 50 25 10 5 2 1 0)
EPOCHS_TSP100=(50 25 10 5 2 1 0)

run_one() {
  local script="$1"
  local epochs="$2"
  echo "[RUN_ALL] Starting ${script} with EPOCHS=${epochs} on CUDA_GPUS=${CUDA_GPUS}"
  RUN_FOREGROUND=1 EPOCHS="${epochs}" "./${script}"
  echo "[RUN_ALL] Finished ${script} with EPOCHS=${epochs}"
}

echo "[RUN_ALL] ===== TSP-20 sweep ====="
for e in "${EPOCHS_TSP20[@]}"; do
  run_one "train_pomo_tsp20.sh" "${e}"
done

echo "[RUN_ALL] ===== TSP-50 sweep ====="
for e in "${EPOCHS_TSP50[@]}"; do
  run_one "train_pomo_tsp50.sh" "${e}"
done

echo "[RUN_ALL] ===== TSP-100 sweep ====="
for e in "${EPOCHS_TSP100[@]}"; do
  run_one "train_pomo_tsp100.sh" "${e}"
done

echo "[RUN_ALL] All POMO TSP sweeps completed."

