#!/usr/bin/env bash
set -euo pipefail

# cd 到脚本所在目录（假设就是 repo 根目录）
cd "$(dirname "$0")"

# 可通过环境变量覆盖的参数
OUT_PATH="${OUT_PATH:-data/tsp100_offline_trajs_100000.pt}"
NUM_EPISODES="${NUM_EPISODES:-100000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SEED="${SEED:-1234}"
NUM_LOC="${NUM_LOC:-100}"
CKPT="${CKPT:-checkpoints/pomo_tsp100.ckpt}"
DEVICE="${DEVICE:-cuda}"   # 没 GPU 就改成 cpu

mkdir -p "$(dirname "$OUT_PATH")"

echo "[INFO] Collecting TSP${NUM_LOC} offline trajectories to ${OUT_PATH}"
python -m rl4co.heuristic_finder.offline_data_tsp20 \
  --out-path "${OUT_PATH}" \
  --num-episodes "${NUM_EPISODES}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --num-loc "${NUM_LOC}" \
  --ckpt "${CKPT}" \
  --device "${DEVICE}"
