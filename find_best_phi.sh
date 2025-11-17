#!/usr/bin/env bash
set -euo pipefail

# cd to repo root (directory of this script)
cd "$(dirname "$0")"

echo "[INFO] Launching EoH-style multi-pop search via DeepSeek/Kimi API (logs: find_best_phi.log)" | tee -a setup.log

# ===== Remote LLM API configuration =====
# Required: export DEEPSEEK_API_KEY (default) or KIMI_API_KEY plus set LLM_API_PROVIDER=kimi.
export LLM_API_PROVIDER="${LLM_API_PROVIDER:-kimi}"
# DeepSeek overrides
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-reasoner}"
export DEEPSEEK_MAX_TOKENS="${DEEPSEEK_MAX_TOKENS:-32768}"
export DEEPSEEK_TEMPERATURE="${DEEPSEEK_TEMPERATURE:-0.0}"
export DEEPSEEK_STREAM="${DEEPSEEK_STREAM:-false}"
# Kimi overrides (used when provider is kimi)
export KIMI_API_BASE="${KIMI_API_BASE:-https://api.moonshot.cn/v1}"
export KIMI_MODEL="${KIMI_MODEL:-kimi-k2-turbo-preview}"
# export KIMI_API_BASE="${KIMI_API_BASE:-https://api.bltcy.ai/v1}"
# export KIMI_MODEL="${KIMI_MODEL:-gpt-4o-mini}"
export KIMI_MAX_TOKENS="${KIMI_MAX_TOKENS:-8192}"
export KIMI_TEMPERATURE="${KIMI_TEMPERATURE:-0.6}"
export KIMI_STREAM="${KIMI_STREAM:-false}"

# ===== TSP nodes (offline value dataset) =====
# Controls the TSP size used when generating offline trajectories for V(s) at different scales.
# This env var is still used by some online diagnostics; offline generators below use explicit --num-loc.
export TSP_NUM_LOC="${TSP_NUM_LOC:-100}"
export LLM_DEBUG=0
export LLM_DUMP_DIR=""

# ===== Evolution settings (EoH-style) =====
N_POPS=${N_POPS:-1}
POP_SIZE=${POP_SIZE:-10}
GENERATIONS=${GENERATIONS:-20}
OPERATORS=${OPERATORS:-e1,e2,m1,m2,m3}
OP_WEIGHTS=${OP_WEIGHTS:-1,1,1,1,0.3}
TOURNAMENT_K=${TOURNAMENT_K:-4}

# ===== Offline trajectories for Level-1 cheap eval =====
# Main fitness dataset (typically largest scale, e.g., TSP-100)
OFFLINE_TRAJ_PATH_100=${OFFLINE_TRAJ_PATH_100:-data/tsp100_offline_trajs_100000.pt}
# Additional datasets for multi-scale diagnostics / reflection
OFFLINE_TRAJ_PATH_20=${OFFLINE_TRAJ_PATH_20:-data/tsp20_offline_trajs_100000.pt}
OFFLINE_TRAJ_PATH_50=${OFFLINE_TRAJ_PATH_50:-data/tsp50_offline_trajs_100000.pt}
OFFLINE_NUM_EPISODES=${OFFLINE_NUM_EPISODES:-100000}
OFFLINE_BATCH_SIZE=${OFFLINE_BATCH_SIZE:-512}
BASELINE_CKPT=${BASELINE_CKPT:-baseline.ckpt}
CHEAP_EVAL_DEVICE=${CHEAP_EVAL_DEVICE:-cuda}
CHEAP_EVAL_BATCH_STATES=${CHEAP_EVAL_BATCH_STATES:-2048}

# Generate offline data for each requested scale if missing
gen_offline() {
  local path="$1"
  local num_loc="$2"
  if [[ -f "$path" ]]; then
    return
  fi
  echo "[INFO] Offline trajectories not found at $path; generating for TSP-${num_loc}..." | tee -a setup.log
  mkdir -p "$(dirname "$path")"
  gen_cmd=(python -m rl4co.heuristic_finder.offline_data_tsp20 \
    --out-path "$path" \
    --num-episodes "$OFFLINE_NUM_EPISODES" \
    --batch-size "$OFFLINE_BATCH_SIZE" \
    --seed "${SEED:-1234}" \
    --num-loc "$num_loc")
  if [[ -n "$BASELINE_CKPT" ]]; then
    gen_cmd+=(--ckpt "$BASELINE_CKPT")
  fi
  echo "Generating offline data: ${gen_cmd[*]}" | tee -a setup.log
  "${gen_cmd[@]}"
}

gen_offline "$OFFLINE_TRAJ_PATH_20" 20
gen_offline "$OFFLINE_TRAJ_PATH_50" 50
gen_offline "$OFFLINE_TRAJ_PATH_100" 100

# Output / misc
SEED_DUMP_DIR=${SEED_DUMP_DIR:-}
DUMP_DIR=${DUMP_DIR:-runs/eoh}
SAVE_PATH=${SAVE_PATH:-phi_best.py}
TOPK=${TOPK:-5}
SEED=${SEED:-1234}

# Build command (no Ollama flags; we use remote API via env)
cmd=(python examples/auto_find_phi_tsp20.py
  --n-pops "$N_POPS"
  --pop-size "$POP_SIZE"
  --generations "$GENERATIONS"
  --operators "$OPERATORS"
  --operator-weights "$OP_WEIGHTS"
  --tournament-k "$TOURNAMENT_K"
  --dump-dir "$DUMP_DIR"
  --save-path "$SAVE_PATH"
  --topk "$TOPK"
  --seed "$SEED"
  --offline-traj-path "$OFFLINE_TRAJ_PATH_100"
  --offline-traj-paths-multi "${OFFLINE_TRAJ_PATH_20},${OFFLINE_TRAJ_PATH_50},${OFFLINE_TRAJ_PATH_100}"
  --cheap-eval-device "${CHEAP_EVAL_DEVICE}"
  --cheap-eval-batch-states "${CHEAP_EVAL_BATCH_STATES}"
)

if [[ -n "$SEED_DUMP_DIR" ]]; then
  cmd+=(--seed-dump-dir "$SEED_DUMP_DIR")
fi

echo "Running: ${cmd[*]} $*" | tee -a setup.log
nohup "${cmd[@]}" "$@" > find_best_phi.log 2>&1 &

echo "[INFO] Started. Tail logs with: tail -f find_best_phi.log" | tee -a setup.log

# baseline: 4.534433
