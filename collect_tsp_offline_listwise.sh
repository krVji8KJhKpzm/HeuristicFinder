#!/usr/bin/env bash
set -euo pipefail

# Generic TSP listwise offline data collection using a trained POMO solver.
# Usage:
#   ./collect_tsp_offline_listwise.sh NUM_LOC [TOTAL_INSTANCES] [N_SEEDS]
#
# Example (TSP-20, 10k instances over 5 seeds):
#   ./collect_tsp_offline_listwise.sh 20 10000 5
#
# This script assumes checkpoints are stored as:
#   checkpoints/pomo_tsp${NUM_LOC}.ckpt

# Change to repo root (directory of this script)
cd "$(dirname "$0")"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 NUM_LOC [TOTAL_INSTANCES] [N_SEEDS]" >&2
  exit 1
fi

NUM_LOC="$1"
TOTAL_INSTANCES="${2:-10000}"
N_SEEDS="${3:-5}"

if ! [[ "$NUM_LOC" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] NUM_LOC must be an integer, got '${NUM_LOC}'" >&2
  exit 1
fi

if ! [[ "$TOTAL_INSTANCES" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] TOTAL_INSTANCES must be an integer, got '${TOTAL_INSTANCES}'" >&2
  exit 1
fi

if ! [[ "$N_SEEDS" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] N_SEEDS must be an integer, got '${N_SEEDS}'" >&2
  exit 1
fi

if [[ "$N_SEEDS" -le 0 ]]; then
  echo "[ERROR] N_SEEDS must be >= 1" >&2
  exit 1
fi

INSTANCES_PER_SEED=$((TOTAL_INSTANCES / N_SEEDS))

CKPT="${CKPT:-checkpoints/pomo_tsp${NUM_LOC}.ckpt}"
OUT_DIR="${OUT_DIR:-data/tsp_listwise}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"
DECODE_MODE="${DECODE_MODE:-greedy}"   # 'greedy' or 'sampling'
STEP_STRIDE="${STEP_STRIDE:-1}"
MAX_STATES="${MAX_STATES:-}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Expected POMO checkpoint not found: ${CKPT}" >&2
  echo "        Override via CKPT=/path/to/ckpt.ckpt $0 ${NUM_LOC} ..." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[INFO] Collecting listwise TSP${NUM_LOC} offline data"
echo "[INFO] TOTAL_INSTANCES=${TOTAL_INSTANCES}, N_SEEDS=${N_SEEDS}, INSTANCES_PER_SEED=${INSTANCES_PER_SEED}"
echo "[INFO] CKPT=${CKPT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"

for SEED in $(seq 0 $((N_SEEDS - 1))); do
  echo "[INFO] Seed ${SEED}: collecting ${INSTANCES_PER_SEED} instances..."
  if [[ -n "${MAX_STATES}" ]]; then
    python -m rl4co.heuristic_finder.offline_data_tsp_listwise \
      --num-loc "${NUM_LOC}" \
      --n-instances "${INSTANCES_PER_SEED}" \
      --batch-size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --checkpoint "${CKPT}" \
      --out-dir "${OUT_DIR}" \
      --decode-mode "${DECODE_MODE}" \
      --step-stride "${STEP_STRIDE}" \
      --max-states "${MAX_STATES}" \
      --shard-id "${SEED}"
  else
    python -m rl4co.heuristic_finder.offline_data_tsp_listwise \
      --num-loc "${NUM_LOC}" \
      --n-instances "${INSTANCES_PER_SEED}" \
      --batch-size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --checkpoint "${CKPT}" \
      --out-dir "${OUT_DIR}" \
      --decode-mode "${DECODE_MODE}" \
      --step-stride "${STEP_STRIDE}" \
      --shard-id "${SEED}"
  fi
done

echo "[INFO] Done."

