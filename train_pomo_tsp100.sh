#!/usr/bin/env bash
set -euo pipefail

# Train POMO on TSP-100 for a configurable number of epochs.
# Recommended epoch settings: 50, 25, 10, 5, 2, 1, 0
# (epoch=100 is intentionally not used for TSP-100 as requested)
# Usage examples:
#   CUDA_GPUS=0 EPOCHS=50 ./train_pomo_tsp100.sh
#   CUDA_GPUS=1 EPOCHS=25 ./train_pomo_tsp100.sh

cd "$(dirname "$0")"

CUDA_GPUS=${CUDA_GPUS:-0}
export CUDA_VISIBLE_DEVICES="$CUDA_GPUS"
IFS=',' read -r -a GPU_ARR <<< "$CUDA_GPUS"
NUM_DEVICES=${#GPU_ARR[@]}

EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-64}
TRAIN_SIZE=${TRAIN_SIZE:-100000}
VAL_SIZE=${VAL_SIZE:-10000}
TEST_SIZE=${TEST_SIZE:-10000}
LR=${LR:-1e-4}
SEED=${SEED:-1234}

NUM_LOC=100
EXP_NAME="pomo-tsp${NUM_LOC}-ep${EPOCHS}"

echo "[INFO] Starting POMO training: TSP-${NUM_LOC}, epochs=${EPOCHS} (log: train_${EXP_NAME}.log)"

if [[ "${RUN_FOREGROUND:-0}" == "1" ]]; then
  python run.py \
    experiment=routing/pomo.yaml \
    callbacks=print_val_objective.yaml \
    env.generator_params.num_loc=${NUM_LOC} \
    +trainer.enable_progress_bar=false \
    trainer.accelerator=gpu \
    +trainer.devices=${NUM_DEVICES} \
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
    +model.num_starts=null \
    seed=${SEED} \
    logger=csv logger.csv.name=${EXP_NAME} \
    | tee "train_${EXP_NAME}.log"
else
  nohup python run.py \
    experiment=routing/pomo.yaml \
    callbacks=print_val_objective.yaml \
    env.generator_params.num_loc=${NUM_LOC} \
    +trainer.enable_progress_bar=false \
    trainer.accelerator=gpu \
    +trainer.devices=${NUM_DEVICES} \
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
    +model.num_starts=null \
    seed=${SEED} \
    logger=csv logger.csv.name=${EXP_NAME} \
    > "train_${EXP_NAME}.log" 2>&1 &

  echo "[INFO] Launched. Tail logs with: tail -f train_${EXP_NAME}.log"
fi
