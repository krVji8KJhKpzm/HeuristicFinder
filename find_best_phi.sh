#!/usr/bin/env bash
set -euo pipefail

# cd to repo root (directory of this script)
cd "$(dirname "$0")"

echo "[INFO] Launching EoH-style multi-pop search via DeepSeek API (logs: find_best_phi.log)" | tee -a setup.log

# ===== DeepSeek API configuration =====
# Required: export DEEPSEEK_API_KEY in your environment (already set as you mentioned).
# Optional overrides (defaults below):
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
export DEEPSEEK_MAX_TOKENS="${DEEPSEEK_MAX_TOKENS:-32768}"
export DEEPSEEK_TEMPERATURE="${DEEPSEEK_TEMPERATURE:-0.0}"

# ===== TSP nodes (short-training fitness) =====
# Controls the TSP size used during short-training fitness (train_fitness_phi_on_tsp20).
# Default to 20 to match TSP-20; change to 50, 100, etc. as needed.
export TSP_NUM_LOC="${TSP_NUM_LOC:-20}"

# ===== Evolution settings (EoH-style) =====
N_POPS=${N_POPS:-4}
POP_SIZE=${POP_SIZE:-8}
GENERATIONS=${GENERATIONS:-10}
OPERATORS=${OPERATORS:-e1,e2,m1,m2}
OP_WEIGHTS=${OP_WEIGHTS:-1,1,1,1}
TOURNAMENT_K=${TOURNAMENT_K:-2}

# Short-training fitness budgets
EPOCHS_PER_EVAL=${EPOCHS_PER_EVAL:-1}
BATCH_SIZE=${BATCH_SIZE:-64}
TRAIN_SIZE=${TRAIN_SIZE:-1000}
VAL_SIZE=${VAL_SIZE:-256}
NUM_STARTS=${NUM_STARTS:-8}

# PBRS shaping toggles
GAMMA_CHOICES=${GAMMA_CHOICES:-"1.0,-0.1,0.1"}
REWARD_SCALE=${REWARD_SCALE:-scale} # None|scale|norm
CENTER_DPHI=${CENTER_DPHI:-1}
NORM_DPHI=${NORM_DPHI:-1}

# Diversity / archive / memetic
MEMETIC_REPAIR_PROB=${MEMETIC_REPAIR_PROB:-0.25}
ARCHIVE_TOP_K=${ARCHIVE_TOP_K:-32}
ELITE_PARENT_K=${ELITE_PARENT_K:-2}
ELITE_REPLACE_WORST=${ELITE_REPLACE_WORST:-0}

# Output / misc
DUMP_DIR=${DUMP_DIR:-runs/eoh}
SAVE_PATH=${SAVE_PATH:-phi_best.py}
TOPK=${TOPK:-5}
SEED=${SEED:-1234}
GPU_IDS=${GPU_IDS:-}        # e.g., "0,1,2,3" for parallel short-training; leave empty for CPU

# Build command (no Ollama flags; we use DeepSeek API via env)
cmd=(python examples/auto_find_phi_tsp20.py
  --n-pops "$N_POPS"
  --pop-size "$POP_SIZE"
  --generations "$GENERATIONS"
  --operators "$OPERATORS"
  --operator-weights "$OP_WEIGHTS"
  --tournament-k "$TOURNAMENT_K"
  --epochs-per-eval "$EPOCHS_PER_EVAL"
  --batch-size "$BATCH_SIZE"
  --train-size "$TRAIN_SIZE"
  --val-size "$VAL_SIZE"
  --num-starts "$NUM_STARTS"
  --gamma-choices "$GAMMA_CHOICES"
  --reward-scale "$REWARD_SCALE"
  --dump-dir "$DUMP_DIR"
  --save-path "$SAVE_PATH"
  --topk "$TOPK"
  --seed "$SEED"
)

if [[ "${CENTER_DPHI}" == "1" ]]; then cmd+=(--center-dphi); fi
if [[ "${NORM_DPHI}" == "1" ]]; then cmd+=(--norm-dphi); fi

if [[ -n "$GPU_IDS" ]]; then
  cmd+=(--gpu-ids "$GPU_IDS")
fi

echo "Running: ${cmd[*]} $*" | tee -a setup.log
nohup "${cmd[@]}" "$@" > find_best_phi.log 2>&1 &

echo "[INFO] Started. Tail logs with: tail -f find_best_phi.log" | tee -a setup.log
