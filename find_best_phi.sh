#!/usr/bin/env bash
set -euo pipefail

# cd to repo root (directory of this script)
cd "$(dirname "$0")"

# 1) Start Ollama bound to GPU #7 only (service sees only GPU 7)
#    This does NOT affect subsequent commands since CUDA_VISIBLE_DEVICES is set only for this process.
echo "[INFO] Starting Ollama on GPU 7 (logs: ollama.log)" | tee -a setup.log
CUDA_VISIBLE_DEVICES=7 nohup ollama serve > ollama.log 2>&1 &

# Optional: wait a moment for service to be ready
sleep 3 || true

# 2) Run evolutionary search with short-training fitness on GPUs 0,1,2,3
#    The search process will see all GPUs; internally it will assign each candidate to one GPU.
echo "[INFO] Launching auto_find_phi_tsp20 (logs: find_best_phi.log)" | tee -a setup.log
# PBRS search tuning (override via env):
#   GAMMA_CHOICES: comma list of gamma values to try per candidate
#   REWARD_SCALE:  None|scale|norm (advantage scaling)
#   CENTER_DPHI / NORM_DPHI are enabled by default here
GAMMA_CHOICES=${GAMMA_CHOICES:-"1.0,-0.1,0.1"}
REWARD_SCALE=${REWARD_SCALE:-scale}
nohup python examples/auto_find_phi_tsp20.py \
  --gpu-ids 0,1,2,3,4,5 \
  --population-size 6 \
  --iterations 10 \
  --ollama-model qwen3:32b \
  --num-starts 20 \
  --device gpu \
  --seed 1234 \
  --dump-dir phi_generations \
  --train-size 100000 \
  --val-size 1000 \
  --epochs-per-eval 2 \
  --gamma-choices "${GAMMA_CHOICES}" \
  --reward-scale "${REWARD_SCALE}" \
  --center-dphi \
  --norm-dphi \
  > find_best_phi.log 2>&1 &

echo "[INFO] Started. Tail logs with: tail -f find_best_phi.log ollama.log" | tee -a setup.log
