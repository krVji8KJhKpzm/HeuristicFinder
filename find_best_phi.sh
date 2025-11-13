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
export KIMI_MAX_TOKENS="${KIMI_MAX_TOKENS:-8192}"
export KIMI_TEMPERATURE="${KIMI_TEMPERATURE:-0.6}"
export KIMI_STREAM="${KIMI_STREAM:-false}"

# ===== TSP nodes (short-training fitness) =====
# Controls the TSP size used during short-training fitness (train_fitness_phi_on_tsp20).
# Default to 20 to match TSP-20; change to 50, 100, etc. as needed.
export TSP_NUM_LOC="${TSP_NUM_LOC:-20}"
export LLM_DEBUG=0
export LLM_DUMP_DIR=""

# ===== Evolution settings (EoH-style) =====
N_POPS=${N_POPS:-1}
POP_SIZE=${POP_SIZE:-20}
GENERATIONS=${GENERATIONS:-20}
OPERATORS=${OPERATORS:-e1,e2,m1,m2}
OP_WEIGHTS=${OP_WEIGHTS:-1,1,1,1}
TOURNAMENT_K=${TOURNAMENT_K:-4}

# Short-training fitness budgets
EPOCHS_PER_EVAL=${EPOCHS_PER_EVAL:-10}
BATCH_SIZE=${BATCH_SIZE:-512}
TRAIN_SIZE=${TRAIN_SIZE:-10000}
VAL_SIZE=${VAL_SIZE:-1000}
NUM_STARTS=${NUM_STARTS:-20}

# PBRS shaping toggles
GAMMA_CHOICES=${GAMMA_CHOICES:-"0.2,-0.1,0.1,0.5"}
REWARD_SCALE=${REWARD_SCALE:-scale} # None|scale|norm
CENTER_DPHI=${CENTER_DPHI:-1}
NORM_DPHI=${NORM_DPHI:-1}

# Diversity / archive / memetic
MEMETIC_REPAIR_PROB=${MEMETIC_REPAIR_PROB:-0.25}
ARCHIVE_TOP_K=${ARCHIVE_TOP_K:-32}
ELITE_PARENT_K=${ELITE_PARENT_K:-4}
ELITE_REPLACE_WORST=${ELITE_REPLACE_WORST:-2}

# ===== Offline trajectories for Level-1 cheap eval =====
OFFLINE_TRAJ_PATH=${OFFLINE_TRAJ_PATH:-data/tsp20_offline_trajs.pt}
OFFLINE_NUM_EPISODES=${OFFLINE_NUM_EPISODES:-100000}
OFFLINE_BATCH_SIZE=${OFFLINE_BATCH_SIZE:-512}
BASELINE_CKPT=${BASELINE_CKPT:-baseline.ckpt}

# Generate offline data if missing
if [[ ! -f "$OFFLINE_TRAJ_PATH" ]]; then
  echo "[INFO] Offline trajectories not found at $OFFLINE_TRAJ_PATH; generating..." | tee -a setup.log
  mkdir -p "$(dirname "$OFFLINE_TRAJ_PATH")"
  gen_cmd=(python -m rl4co.heuristic_finder.offline_data_tsp20 \
    --out-path "$OFFLINE_TRAJ_PATH" \
    --num-episodes "$OFFLINE_NUM_EPISODES" \
    --batch-size "$OFFLINE_BATCH_SIZE" \
    --seed "${SEED:-1234}")
  if [[ -n "$BASELINE_CKPT" ]]; then
    gen_cmd+=(--ckpt "$BASELINE_CKPT")
  fi
  echo "Generating offline data: ${gen_cmd[*]}" | tee -a setup.log
  "${gen_cmd[@]}"
fi

# Output / misc
SEED_DUMP_DIR=${SEED_DUMP_DIR:-}
DUMP_DIR=${DUMP_DIR:-runs/eoh}
SAVE_PATH=${SAVE_PATH:-phi_best.py}
TOPK=${TOPK:-5}
SEED=${SEED:-1234}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5}        # e.g., "0,1,2,3" for parallel short-training; leave empty for CPU

# Build command (no Ollama flags; we use remote API via env)
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
  --offline-traj-path "$OFFLINE_TRAJ_PATH"
  --cheap-level-weight "${CHEAP_LEVEL_WEIGHT:-0.1}"
  --cheap-filter-threshold "${CHEAP_FILTER_THRESHOLD:--1e9}"
  --cheap-topk-ratio "${CHEAP_TOPK_RATIO:-0.3}"
  --max-candidates-rl-eval "${MAX_CANDIDATES_RL_EVAL:-8}"
  --max-step-shaping-ratio "${MAX_STEP_SHAPING_RATIO:-10.0}"
  --max-episode-shaping-ratio "${MAX_EPISODE_SHAPING_RATIO:-10.0}"
  --max-var-ratio-shaped-vs-base "${MAX_VAR_RATIO_SHAPED_VS_BASE:-10.0}"
  --min-abs-dphi-q95 "${MIN_ABS_DPHI_Q95:-1e-4}"
  --complexity-penalty-alpha "${COMPLEXITY_PENALTY_ALPHA:-0.001}"
  --refine-top-k "${REFINE_TOP_K:-5}"
  --refine-epochs "${REFINE_EPOCHS:-10}"
)

if [[ "${CENTER_DPHI}" == "1" ]]; then cmd+=(--center-dphi); fi
if [[ "${NORM_DPHI}" == "1" ]]; then cmd+=(--norm-dphi); fi

if [[ -n "$GPU_IDS" ]]; then
  cmd+=(--gpu-ids "$GPU_IDS")
fi

if [[ -n "$SEED_DUMP_DIR" ]]; then
  cmd+=(--seed-dump-dir "$SEED_DUMP_DIR")
fi

echo "Running: ${cmd[*]} $*" | tee -a setup.log
nohup "${cmd[@]}" "$@" > find_best_phi.log 2>&1 &

echo "[INFO] Started. Tail logs with: tail -f find_best_phi.log" | tee -a setup.log

# baseline: 4.534433
