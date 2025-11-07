#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Paths
BEST_PHI_FILE="phi_best.py"

if [[ ! -f "$BEST_PHI_FILE" ]]; then
  echo "[ERROR] $BEST_PHI_FILE not found. Run the search step first or place your def phi(state): ... in $BEST_PHI_FILE" >&2
  exit 1
fi

# GPU selection
# Set CUDA_GPUS to the GPU ids you want to use, e.g., "0" or "0,1".
# You can override at runtime: CUDA_GPUS=1 ./train_full_tsp20.sh
CUDA_GPUS=${CUDA_GPUS:-0}
export CUDA_VISIBLE_DEVICES="$CUDA_GPUS"
# Derive number of visible devices for Lightning
IFS=',' read -r -a GPU_ARR <<< "$CUDA_GPUS"
NUM_DEVICES=${#GPU_ARR[@]}

# Common overrides
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-512}
TRAIN_SIZE=${TRAIN_SIZE:-200000}
VAL_SIZE=${VAL_SIZE:-10000}
TEST_SIZE=${TEST_SIZE:-10000}
LR=${LR:-1e-4}
SEED=${SEED:-1234}

# 1) PBRS full training using best phi
# Enable Phi(s) logging to train_pbrs.log
export PBRS_LOG_PHI=${PBRS_LOG_PHI:-1}
export PBRS_LOG_PHI_MODE=${PBRS_LOG_PHI_MODE:-first}
export PBRS_LOG_PHI_EVERY=${PBRS_LOG_PHI_EVERY:-1}
echo "[INFO] Starting PBRS (best phi) full training... (train_pbrs.log)"
nohup python run.py \
  experiment=routing/pomopbrs-tsp20.yaml \
  callbacks=print_val_objective.yaml \
  model.potential="file:${BEST_PHI_FILE}" \
  +trainer.enable_progress_bar=false \
  trainer.accelerator=gpu \
  trainer.devices=${NUM_DEVICES} \
  trainer.max_epochs=${EPOCHS} \
  model.batch_size=${BATCH} \
  model.train_data_size=${TRAIN_SIZE} \
  model.val_data_size=${VAL_SIZE} \
  model.test_data_size=${TEST_SIZE} \
  model.optimizer_kwargs.lr=${LR} \
  model.optimizer_kwargs.weight_decay=1e-6 \
  model.lr_scheduler="MultiStepLR" \
  model.lr_scheduler_kwargs.milestones=[80,95] \
  model.lr_scheduler_kwargs.gamma=0.1 \
  seed=${SEED} \
  logger=csv logger.csv.name=pbrs-tsp20 \
  > train_pbrs.log 2>&1 &

# 2) Baseline POMO full training with original reward (same budgets)
echo "[INFO] Starting POMO baseline full training... (train_baseline.log)"
nohup python run.py \
  experiment=routing/pomo.yaml \
  callbacks=print_val_objective.yaml \
  env.generator_params.num_loc=20 \
  +trainer.enable_progress_bar=false \
  trainer.accelerator=gpu \
  trainer.devices=${NUM_DEVICES} \
  trainer.max_epochs=${EPOCHS} \
  model.batch_size=${BATCH} \
  model.train_data_size=${TRAIN_SIZE} \
  model.val_data_size=${VAL_SIZE} \
  model.test_data_size=${TEST_SIZE} \
  model.optimizer_kwargs.lr=${LR} \
  model.optimizer_kwargs.weight_decay=1e-6 \
  model.lr_scheduler="MultiStepLR" \
  model.lr_scheduler_kwargs.milestones=[80,95] \
  model.lr_scheduler_kwargs.gamma=0.1 \
  model.num_starts=null \
  seed=${SEED} \
  logger=csv logger.csv.name=pomo-tsp20 \
  > train_baseline.log 2>&1 &

echo "[INFO] Launched. Tail logs:\n  tail -f train_pbrs.log\n  tail -f train_baseline.log"
