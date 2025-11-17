#!/usr/bin/env bash
set -euo pipefail

# Change to the repo root (directory of this script)
cd "$(dirname "$0")"

# Parameters (can be overridden via environment variables)
OUT_PATH="${OUT_PATH:-data/tsp100_offline_trajs_100000.pt}"
NUM_EPISODES="${NUM_EPISODES:-100000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SEED="${SEED:-1234}"
NUM_LOC="${NUM_LOC:-100}"
DEVICE="${DEVICE:-cuda}"          # use 'cuda' for GPU, 'cpu' for CPU
SOLVER="${SOLVER:-concorde}"      # 'concorde' or 'lkh' (exact solvers), or 'pomo'
CONCORDE_WORKERS="${CONCORDE_WORKERS:-0}"  # 0 or <=0 means auto-detect
LKH_EXE="${LKH_EXE:-LKH}"         # LKH executable name or path (when SOLVER=lkh)

mkdir -p "$(dirname "$OUT_PATH")"

echo "[INFO] Collecting TSP${NUM_LOC} offline trajectories to ${OUT_PATH} (solver=${SOLVER})"
python -m rl4co.heuristic_finder.offline_data_tsp20 \
  --out-path "${OUT_PATH}" \
  --num-episodes "${NUM_EPISODES}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --num-loc "${NUM_LOC}" \
  --device "${DEVICE}" \
  --solver "${SOLVER}" \
  --concorde-workers "${CONCORDE_WORKERS}" \
  --lkh-exe "${LKH_EXE}"
