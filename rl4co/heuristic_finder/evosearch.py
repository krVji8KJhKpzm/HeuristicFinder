from __future__ import annotations

import glob
import json
import os
import random
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set, Any

import torch

from rl4co.heuristic_finder.potential_eval import (
    cheap_score_phi,
    compute_phi_stats,
    mse_phi_vs_value,
    robust_phi_objectives,
    listwise_preference_objectives,
)
from rl4co.heuristic_finder.offline_data_tsp20 import load_offline_trajectories
from rl4co.heuristic_finder.llm import (
    eoh_llm_i1,
    eoh_llm_e1,
    eoh_llm_e2,
    eoh_llm_m1,
    eoh_llm_m2,
    eoh_llm_m3,
    eoh_llm_repair,
)
from rl4co.heuristic_finder.potential import PotentialSpec, compile_potential
# diagnostics removed in EoH-faithful loop (no reflection)


# Simple cache for offline trajectory datasets keyed by path
_OFFLINE_TRAJ_CACHE: Dict[str, Any] = {}


@dataclass
class Candidate:
    """Evolutionary individual: potential function + its gamma."""

    spec: PotentialSpec
    gamma: float
    algorithm: Optional[str] = None
    code_hash: Optional[str] = None
    # Diagnostics / reflections
    stats: Optional[Dict[str, float]] = None
    stats_text: Optional[str] = None


@dataclass
class EvoConfig:
    # Multi-population controls (EoH style)
    n_pops: int = 1  # number of independent populations (EoH ec_n_pop)
    pop_size: int = 4  # individuals per population (EoH ec_pop_size)
    # Backward compatibility aliases:
    population_size: int = 4  # alias for pop_size
    generations: int = 2  # number of evolutionary generations
    iterations: int = 2  # alias for generations
    # Fitness evaluation
    epochs_per_eval: int = 1
    batch_size: int = 64
    train_size: int = 1000
    val_size: int = 256
    num_starts: int = 8
    device: str = "cpu"
    # LLM via Ollama only
    ollama_model: Optional[str] = None  # e.g., 'qwen3:32b'
    # EoH-style operator schedule
    operators: Optional[List[str]] = None  # e.g., ['e1','e2','m1','m2']
    operator_weights: Optional[List[float]] = None  # per-operator probability
    m_parents: int = 2  # number of parents for e1/e2
    tournament_k: int = 3  # tournament size
    # Optional: parallel short-training across multiple GPUs
    gpu_ids: Optional[List[int]] = None
    # Optional dir to dump all candidate codes per generation
    dump_dir: Optional[str] = None
    # Optional dir containing previous dump data for seeding
    seed_dump_dir: Optional[str] = None
    # Optional fixed seed for reproducible short-training
    seed: Optional[int] = None
    # PBRS controls
    pbrs_gamma_choices: Optional[List[float]] = None
    reward_scale: Optional[str] = None
    center_dphi: bool = False
    norm_dphi: bool = False
    # Penalize candidates whose estimated tour length exceeds this threshold
    objective_bad_threshold: float = 50.0
    # EoH-init: create this many x N seeds via i1
    initial_copies: int = 2
    # Keep gamma fixed by default to match EoH (no hyper evolution)
    mutate_gamma: bool = False
    # Diversity & dedup
    dedup_within_pop: bool = True
    dedup_global: bool = True
    # Thought & memetic
    enable_thought: bool = True
    memetic_repair_prob: float = 0.0  # chance to apply a light repair LLM pass
    # Elite archive
    archive_top_k: int = 8
    elite_parent_k: int = 2
    elite_replace_worst: int = 1
    # Toggle Level-1 cheap evaluation stage
    use_cheap_level: bool = True
    # Toggle Level-2 short RL evaluation stage
    use_level2_rl: bool = True
    # Level 1 / Level 2 multi-stage evaluation (Level-1 = offline credit-assignment diagnostics,
    # Level-2 = short RL training evaluation)
    offline_traj_path: str = "data/tsp20_offline_trajs.pt"
    # Optional: additional offline datasets (e.g., for TSP-20/50/100) used for diagnostics
    offline_traj_paths_multi: Optional[List[str]] = None
    cheap_level_weight: float = 0.1
    cheap_filter_threshold: float = -1e9
    cheap_topk_ratio: float = 0.3
    max_candidates_rl_eval: int = 8
    # Phi statistical hard filters
    max_step_shaping_ratio: float = 10.0
    max_episode_shaping_ratio: float = 10.0
    max_var_ratio_shaped_vs_base: float = 10.0
    min_abs_dphi_q95: float = 1e-4
    # Complexity regularization
    complexity_penalty_alpha: float = 0.001
    # Optional refine pass for elites per merge step
    refine_top_k: int = 0
    refine_epochs: int = 0
    # Device for cheap Level-1 eval
    cheap_eval_device: str = "cpu"
    cheap_eval_batch_states: Optional[int] = 4096
    # Diagnostics collection for reflection
    collect_stats: bool = True
    stats_batch_size: int = 64
    # Logging / diagnostics
    log_evo_details: bool = False
    # Listwise preference-based fitness (optional)
    listwise_data_path: Optional[str] = None  # path to listwise offline dataset (.pt)
    listwise_max_lists: Optional[int] = None  # cap number of lists per evaluation
    listwise_pair_weight: float = 0.5  # weight for pairwise accuracy in fitness


def compile_candidates(codes: List[str]) -> List[PotentialSpec]:
    out: List[PotentialSpec] = []
    for i, code in enumerate(codes):
        try:
            fn = compile_potential(code)
        except Exception:
            continue
        out.append(PotentialSpec(name=f"llm_{i}", code=code, fn=fn))
    return out


def _load_seed_candidates(seed_dir: str, max_count: int, cfg: EvoConfig) -> List[Candidate]:
    if not seed_dir or max_count <= 0 or not os.path.isdir(seed_dir):
        return []
    manifest_paths = sorted(glob.glob(os.path.join(seed_dir, "gen*.jsonl")))
    if not manifest_paths:
        return []
    records: List[Dict[str, object]] = []
    for manifest in manifest_paths:
        try:
            with open(manifest, encoding="utf-8") as mf:
                for line in mf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue
    if not records:
        return []
    records.sort(key=lambda r: float(r.get("score", float("-inf"))), reverse=True)
    seeds: List[Candidate] = []
    seen_hashes: Set[str] = set()
    for idx, rec in enumerate(records):
        if len(seeds) >= max_count:
            break
        file_name = rec.get("file")
        if not file_name:
            continue
        path = os.path.join(seed_dir, file_name)
        try:
            code = Path(path).read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            fn = compile_potential(code)
        except Exception:
            continue
        code_hash = rec.get("code_hash")
        if not code_hash:
            try:
                code_hash = compute_code_hash(code)
            except Exception:
                code_hash = None
        if code_hash and code_hash in seen_hashes:
            continue
        if code_hash:
            seen_hashes.add(code_hash)
        gamma_val = None
        if rec.get("gamma") is not None:
            try:
                gamma_val = float(rec["gamma"])
            except Exception:
                gamma_val = None
        if gamma_val is None:
            gamma_val = _init_gamma(cfg)
        alg = rec.get("algorithm")
        if not alg and getattr(cfg, "enable_thought", True):
            try:
                alg = extract_algorithm(code)
            except Exception:
                alg = None
        cand = Candidate(
            spec=PotentialSpec(name=f"seed_{idx}", code=code, fn=fn),
            gamma=gamma_val,
            algorithm=alg,
            code_hash=code_hash,
            stats=rec.get("stats"),
            stats_text=rec.get("stats_text"),
        )
        seeds.append(cand)
    return seeds


def _mutate_gamma(parent_gamma: float, cfg: EvoConfig) -> float:
    """Mutate gamma: sample from provided choices if any, otherwise jitter around parent.
    Clamps to [-2, 2] for safety.
    """
    choices = cfg.pbrs_gamma_choices
    if choices and len(choices) > 0:
        import random as _r

        return _r.choice(choices)
    import random as _r

    g = parent_gamma + _r.gauss(0.0, 0.2)
    return max(min(g, 2.0), -2.0)


def _init_gamma(cfg: EvoConfig) -> float:
    if cfg.pbrs_gamma_choices and len(cfg.pbrs_gamma_choices) > 0:
        return random.choice(cfg.pbrs_gamma_choices)
    return 1.0


def _tournament_select(scored: List[Tuple[Candidate, float]], m: int, k: int) -> List[Candidate]:
    """Tournament selection (maximize reward) mirroring EoH's tournament (min objective)."""
    parents: List[Candidate] = []
    if not scored:
        return parents
    while len(parents) < m:
        tour = random.sample(scored, k=min(k, len(scored)))
        winner = max(tour, key=lambda x: x[1])[0]
        parents.append(winner)
    return parents


def _parents_pack(pars: List[Candidate]) -> List[dict]:
    pack = []
    for p in pars:
        alg = p.algorithm
        if not alg:
            try:
                alg = extract_algorithm(p.spec.code)
            except Exception:
                alg = None
        if not alg:
            alg = "(no description)"
        entry = {"algorithm": alg, "code": p.spec.code}
        if p.stats_text:
            entry["diagnostics"] = p.stats_text
        pack.append(entry)
    return pack


def _propose_offspring_for_operator(
    scored: List[Tuple[Candidate, float]],
    cfg: EvoConfig,
    operator: str,
) -> List[Candidate]:
    """Faithful EoH-style offspring generation for a single operator.

    Generates exactly N (=population_size) new candidates using tournament selection
    from the current population and the specified operator.
    """
    # individuals per operator call equals the per-population size
    ps = cfg.pop_size if cfg.pop_size is not None else cfg.population_size
    N = int(ps)
    out: List[Candidate] = []
    debug = getattr(cfg, "log_evo_details", False)

    # Determine streaming setting from environment variables
    stream = False
    provider = os.environ.get("LLM_API_PROVIDER", "deepseek").lower()
    if provider == "kimi":
        stream = os.environ.get("KIMI_STREAM", "false").lower() in ("true", "1", "yes")
    elif provider == "deepseek":
        stream = os.environ.get("DEEPSEEK_STREAM", "false").lower() in ("true", "1", "yes")
    elif provider == "glm":
        stream = os.environ.get("GLM_STREAM", "false").lower() in ("true", "1", "yes")

    for _ in range(N):
        try:
            codes: List[str] = []
            if operator in ("e1", "e2"):
                pars = _tournament_select(scored, m=cfg.m_parents, k=cfg.tournament_k)
                pack = _parents_pack(pars)
                if operator == "e1":
                    codes = eoh_llm_e1(cfg.ollama_model, pack, n=1, env_name="tsp", debug=False, stream=stream)
                else:
                    codes = eoh_llm_e2(cfg.ollama_model, pack, n=1, env_name="tsp", debug=False, stream=stream)
            elif operator == "m1":
                par = _tournament_select(scored, m=1, k=cfg.tournament_k)[0]
                codes = eoh_llm_m1(cfg.ollama_model, par.spec.code, n=1, env_name="tsp", debug=False, stream=stream)
            elif operator == "m2":
                par = _tournament_select(scored, m=1, k=cfg.tournament_k)[0]
                codes = eoh_llm_m2(cfg.ollama_model, par.spec.code, n=1, env_name="tsp", debug=False, stream=stream)
            elif operator == "m3":
                par = _tournament_select(scored, m=1, k=cfg.tournament_k)[0]
                codes = eoh_llm_m3(cfg.ollama_model, par.spec.code, n=1, env_name="tsp", debug=False, stream=stream)
            else:
                # unknown operator: skip
                continue

            if debug and not codes:
                print(f"[Evo] operator={operator}: LLM returned 0 code samples for an offspring.", flush=True)

            # Optional memetic light repair on raw generation
            if cfg.memetic_repair_prob > 0.0 and random.random() < float(cfg.memetic_repair_prob):
                try:
                    repaired: List[str] = []
                    for c in codes:
                        rc = eoh_llm_repair(cfg.ollama_model, c, env_name="tsp", n=1, debug=False, stream=stream)
                        repaired.append(rc[0] if rc else c)
                    codes = repaired
                except Exception:
                    if debug:
                        print(f"[Evo] operator={operator}: memetic repair failed; keeping original codes.", flush=True)
                    pass

            specs = compile_candidates(codes)
            if debug and codes and not specs:
                print(
                    f"[Evo] operator={operator}: compile_candidates dropped all {len(codes)} code sample(s).",
                    flush=True,
                )
            for sp in specs:
                g = _mutate_gamma(_init_gamma(cfg), cfg) if cfg.mutate_gamma else _init_gamma(cfg)
                alg = extract_algorithm(sp.code) if cfg.enable_thought else None
                ch = compute_code_hash(sp.code)
                out.append(Candidate(spec=sp, gamma=g, algorithm=alg, code_hash=ch))
        except Exception as exc:
            if debug:
                print(f"[Evo] operator={operator}: exception during offspring generation: {exc}", flush=True)
            continue

    return out


class EliteArchive:
    def __init__(self, top_k: int = 8, dump_dir: Optional[str] = None):
        self.top_k = int(top_k)
        self.entries: List[Tuple[Candidate, float]] = []
        self.hashes: Set[str] = set()
        self.dump_path = None
        if dump_dir:
            try:
                os.makedirs(dump_dir, exist_ok=True)
                self.dump_path = os.path.join(dump_dir, "archive.jsonl")
            except Exception:
                self.dump_path = None

    def update(self, scored: List[Tuple[Candidate, float]]):
        changed = False
        for cand, score in scored:
            h = cand.code_hash or compute_code_hash(cand.spec.code)
            if h in self.hashes:
                continue
            self.entries.append((cand, score))
            self.hashes.add(h)
            changed = True
        if changed:
            # keep top-k by score
            self.entries.sort(key=lambda x: x[1], reverse=True)
            self.entries = self.entries[: self.top_k]
            self.hashes = set((c.code_hash or compute_code_hash(c.spec.code)) for c, _ in self.entries)
            if self.dump_path:
                try:
                    with open(self.dump_path, "a", encoding="utf-8") as f:
                        for c, s in scored:
                            # Ensure algorithm is present; fall back to extracting from code
                            try:
                                _algorithm = c.algorithm if c.algorithm else extract_algorithm(c.spec.code)
                            except Exception:
                                _algorithm = c.algorithm
                            _stats = c.stats if c.stats is not None else None
                            _stats_text = c.stats_text if c.stats_text is not None else None
                            rec = {
                                "score": float(s),
                                "gamma": float(c.gamma),
                                "algorithm": _algorithm,
                                "code_hash": c.code_hash or compute_code_hash(c.spec.code),
                                "code": c.spec.code,
                                "stats": _stats,
                                "stats_text": _stats_text,
                            }
                            import json as _json

                            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass

    def top_parents_pack(self, k: int) -> List[Dict[str, str]]:
        if k <= 0 or len(self.entries) == 0:
            return []
        sel = self.entries[: min(k, len(self.entries))]
        pack = []
        for c, _ in sel:
            alg = c.algorithm
            if not alg:
                try:
                    alg = extract_algorithm(c.spec.code)
                except Exception:
                    alg = None
            if not alg:
                alg = "(no description)"
            pack.append({"algorithm": alg, "code": c.spec.code})
        return pack


# ---------------- Diversity & Algorithm helpers -----------------
import ast
import hashlib
import math


def _ast_signature(code: str) -> str:
    try:
        # Extract the phi function region for hashing robustness
        m = re.search(r"def\s+phi\s*\(.*?\):[\s\S]*", code)
        span = m.group(0) if m else code
        tree = ast.parse(span)
        return ast.dump(tree, annotate_fields=False, include_attributes=False)
    except Exception:
        # fallback to stripped text
        return re.sub(r"\s+", " ", code).strip()


def compute_code_hash(code: str) -> str:
    sig = _ast_signature(code)
    h = hashlib.sha256(sig.encode("utf-8")).hexdigest()
    return h


def extract_algorithm(text: str) -> Optional[str]:
    """Extract one-sentence algorithm description using EoH-style rules.

    Primary rule: take the first {...} block content.
    Fallback: take the prefix before 'python' / 'import' / 'def'.
    """
    try:
        # 1) Prefer the first brace-wrapped segment
        m = re.search(r"\{(.*)\}", text, flags=re.DOTALL)
        if m:
            alg = m.group(1).strip()
            if alg:
                return " ".join(alg.split())

        # 2) Fallback: prefix before keywords, as in original EoH _get_alg
        prefix_candidates: List[str] = []
        if "python" in text:
            prefix_candidates = re.findall(r"^.*?(?=python)", text, flags=re.DOTALL)
        elif "import" in text:
            prefix_candidates = re.findall(r"^.*?(?=import)", text, flags=re.DOTALL)
        else:
            prefix_candidates = re.findall(r"^.*?(?=def)", text, flags=re.DOTALL)

        if prefix_candidates:
            alg = prefix_candidates[0].strip()
            if alg:
                return " ".join(alg.split())
    except Exception:
        return None
    return None


def _pearsonr(x, y) -> Optional[float]:
    try:
        import torch
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        if x.numel() < 2 or y.numel() < 2:
            return None
        x = x.flatten()
        y = y.flatten()
        xm = x - x.mean()
        ym = y - y.mean()
        xs = xm.std(unbiased=False)
        ys = ym.std(unbiased=False)
        if xs.item() == 0 or ys.item() == 0:
            return None
        r = (xm * ym).mean() / (xs * ys)
        return float(r.item())
    except Exception:
        return None


def _spearmanr(x, y) -> Optional[float]:
    try:
        import torch
        x = torch.as_tensor(x, dtype=torch.float32).flatten()
        y = torch.as_tensor(y, dtype=torch.float32).flatten()
        if x.numel() < 2 or y.numel() < 2:
            return None
        # ranks via argsort twice
        def _ranks(v):
            idx = torch.argsort(v, stable=True)
            ranks = torch.empty_like(idx, dtype=torch.float32)
            ranks[idx] = torch.arange(1, v.numel() + 1, dtype=torch.float32, device=v.device)
            return ranks
        rx = _ranks(x)
        ry = _ranks(y)
        return _pearsonr(rx, ry)
    except Exception:
        return None


def _summarize_stats(stats: Dict[str, float]) -> str:
    keys = [
        "corr_phi0_final_reward",
        "spr_phi0_final_reward",
        "corr_sum_dphi_final_reward",
        "spr_sum_dphi_final_reward",
        "gamma_T_phiT_minus_phi0_mean",
        "gamma_T_phiT_minus_phi0_std",
        "step_shaping_ratio",
        "episode_shaping_ratio",
        "var_ratio_shaped_vs_base",
        "abs_dphi_q95",
        "terminal_phi_var",
    ]
    parts = []
    for k in keys:
        if k in stats and stats[k] is not None:
            try:
                parts.append(f"{k}={float(stats[k]):.4g}")
            except Exception:
                parts.append(f"{k}={stats[k]}")
    return "; ".join(parts)


def _compute_phi_stats(spec: PotentialSpec, gamma: float, batch_size: int = 64, device: str = "cpu") -> Dict[str, float]:
    """Collect quick diagnostics for Phi on a sampled batch.

    Returns a flat dict with scalar metrics suitable for JSON.
    """
    import torch
    from rl4co.envs.routing.tsp.env import TSPEnv, TSPGenerator
    from rl4co.envs.routing.tsp.pbrs_env import DensePBRSTSPEnv
    from rl4co.models.zoo.pomo import POMO
    import os as _os

    try:
        num_loc = int(_os.environ.get("TSP_NUM_LOC", "50"))
    except Exception:
        num_loc = 50
    gen = TSPGenerator(num_loc=num_loc)
    base_env = TSPEnv(generator=gen)
    td0 = base_env.reset(base_env.generator(batch_size=[batch_size]).to(device))
    model = POMO(env=base_env)
    model.eval()
    with torch.no_grad():
        out = model.policy(td0, base_env, phase="val", return_actions=True, decode_type="greedy")
        actions = out["actions"]  # [B, T]
        base_final = base_env.get_reward(td0, actions)  # [B]

    pbrs_env = DensePBRSTSPEnv(potential_fn=spec.fn, generator=gen, gamma=gamma)
    td = pbrs_env.reset(td0.clone())
    B, T = actions.shape
    phis_b = []  # [B]
    dphis = []  # [B]
    shaped_steps = []  # [B]
    base_steps = []  # [B]
    phi_after_last = None
    phi_init = None

    for t in range(T):
        sv_before = pbrs_env._build_state_view(td)
        phi_before = pbrs_env._safe_phi(sv_before).squeeze(-1)
        if phi_init is None:
            phi_init = phi_before
        td.set("action", actions[:, t])
        td_next = pbrs_env.step(td)["next"]
        sv_after = pbrs_env._build_state_view(td_next)
        phi_after = pbrs_env._safe_phi(sv_after).squeeze(-1)
        shaped = td_next["reward"].squeeze(-1)
        base = shaped - float(gamma) * (phi_after - phi_before)
        phis_b.append(phi_before)
        dphis.append(phi_after - phi_before)
        shaped_steps.append(shaped)
        base_steps.append(base)
        td = td_next
        phi_after_last = phi_after
        if td["done"].all():
            break

    # Stack
    phis_b = torch.stack(phis_b, dim=1)  # [B, t]
    dphis = torch.stack(dphis, dim=1)  # [B, t]
    shaped_steps = torch.stack(shaped_steps, dim=1)
    base_steps = torch.stack(base_steps, dim=1)

    # Episode-level stats
    sum_dphi = dphis.sum(dim=1)  # [B]
    # Correlations
    corr_phi0 = _pearsonr(phis_b[:, 0], base_final)
    spr_phi0 = _spearmanr(phis_b[:, 0], base_final)
    corr_sum_dphi = _pearsonr(sum_dphi, base_final)
    spr_sum_dphi = _spearmanr(sum_dphi, base_final)

    # gamma^T Phi(s_T) - Phi(s_0)
    T_eff = dphis.shape[1]
    try:
        gT = float(gamma) ** float(T_eff)
    except Exception:
        gT = 0.0
    term_minus_start = gT * phi_after_last - (phi_init if phi_init is not None else 0.0)
    tms_mean = float(term_minus_start.mean().item()) if isinstance(term_minus_start, torch.Tensor) else float(term_minus_start)
    tms_std = float(term_minus_start.std(unbiased=False).item()) if isinstance(term_minus_start, torch.Tensor) else 0.0

    # Variance ratios
    base_var = float(base_steps.flatten().var(unbiased=False).item()) if base_steps.numel() > 1 else 0.0
    shaped_var = float(shaped_steps.flatten().var(unbiased=False).item()) if shaped_steps.numel() > 1 else 0.0
    var_ratio = shaped_var / base_var if base_var > 0 else math.inf

    # Magnitudes
    mean_abs_base = float(base_steps.abs().mean().item()) if base_steps.numel() > 0 else 0.0
    mean_abs_dphi = float(dphis.abs().mean().item()) if dphis.numel() > 0 else 0.0
    step_shaping_ratio = (abs(float(gamma)) * mean_abs_dphi / mean_abs_base) if mean_abs_base > 0 else math.inf
    # Episode ratio vs final reward magnitude
    mean_abs_sum_dphi = float(sum_dphi.abs().mean().item())
    mean_abs_final = float(base_final.abs().mean().item()) if base_final.numel() > 0 else 0.0
    episode_shaping_ratio = (abs(float(gamma)) * mean_abs_sum_dphi / mean_abs_final) if mean_abs_final > 0 else math.inf

    # Bounds and smoothness
    abs_dphi = dphis.abs()
    q95 = float(torch.quantile(abs_dphi.flatten(), torch.tensor(0.95, device=abs_dphi.device)).item()) if abs_dphi.numel() > 0 else 0.0
    term_phi_var = float(phi_after_last.var(unbiased=False).item()) if isinstance(phi_after_last, torch.Tensor) and phi_after_last.numel() > 1 else 0.0

    return {
        "corr_phi0_final_reward": corr_phi0,
        "spr_phi0_final_reward": spr_phi0,
        "corr_sum_dphi_final_reward": corr_sum_dphi,
        "spr_sum_dphi_final_reward": spr_sum_dphi,
        "gamma_T_phiT_minus_phi0_mean": tms_mean,
        "gamma_T_phiT_minus_phi0_std": tms_std,
        "var_ratio_shaped_vs_base": var_ratio,
        "step_shaping_ratio": step_shaping_ratio,
        "episode_shaping_ratio": episode_shaping_ratio,
        "abs_dphi_q95": q95,
        "terminal_phi_var": term_phi_var,
    }


def _ensure_stats_for(results: List[Tuple[Candidate, float]], cfg: EvoConfig) -> None:
    # In the symbolic regression setting, stats are populated directly during
    # MSE-based fitness evaluation in `_evaluate_population`, so this helper
    # becomes a no-op. It is kept for API compatibility.
    return


def evolution_search(cfg: EvoConfig) -> List[Tuple[Candidate, float]]:
    """EoH-style multi-population evolution loop with diversity, repair, and elite archive.

    - Independent populations with per-pop selection and variation
    - Explicit deduplication via AST-hash within and across populations
    - Optional memetic light-repair operator applied probabilistically
    - Elite archive maintained across generations with optional parent injection
    """
    # Resolve default operators/weights if not provided
    debug = getattr(cfg, "log_evo_details", False)
    ops = cfg.operators if cfg.operators is not None else ["e1", "e2", "m1", "m2", "m3"]
    op_weights = cfg.operator_weights if cfg.operator_weights is not None else [1.0 for _ in ops]
    if len(op_weights) != len(ops):
        op_weights = [1.0 for _ in ops]

    # Effective sizes
    pop_size = int(cfg.pop_size if cfg.pop_size is not None else cfg.population_size)
    n_pops = int(max(1, cfg.n_pops))
    n_gen = int(cfg.generations if cfg.generations is not None else cfg.iterations)

    # Global dedup state and elite archive
    seen_global: Set[str] = set()
    archive = EliteArchive(top_k=cfg.archive_top_k, dump_dir=cfg.dump_dir)
    seed_pool: List[Candidate] = []
    if cfg.seed_dump_dir:
        seed_pool = _load_seed_candidates(cfg.seed_dump_dir, n_pops * pop_size, cfg)
        if seed_pool:
            print(f"[INFO] Loaded {len(seed_pool)} seed candidates from {cfg.seed_dump_dir}")

    # Initialize populations independently
    populations: List[List[Tuple[Candidate, float]]] = []
    for pop_idx in range(n_pops):
        init_cands: List[Candidate] = []
        seen_pop: Set[str] = set()
        seeds_used = 0
        while seed_pool and len(init_cands) < pop_size:
            cand = seed_pool.pop(0)
            h = cand.code_hash or compute_code_hash(cand.spec.code)
            if cfg.dedup_within_pop and h in seen_pop:
                continue
            if cfg.dedup_global and h in seen_global:
                continue
            if h:
                seen_pop.add(h)
                seen_global.add(h)
            init_cands.append(cand)
            seeds_used += 1
        if len(init_cands) < pop_size:
            need_init = pop_size - len(init_cands)
            total_init = need_init * max(1, int(cfg.initial_copies))
            
            # Determine streaming setting from environment variables
            stream = False
            provider = os.environ.get("LLM_API_PROVIDER", "deepseek").lower()
            if provider == "kimi":
                stream = os.environ.get("KIMI_STREAM", "false").lower() in ("true", "1", "yes")
            elif provider == "deepseek":
                stream = os.environ.get("DEEPSEEK_STREAM", "false").lower() in ("true", "1", "yes")
            elif provider == "glm":
                stream = os.environ.get("GLM_STREAM", "false").lower() in ("true", "1", "yes")
            
            init_codes = eoh_llm_i1(cfg.ollama_model, n=total_init, env_name="tsp", debug=False, stream=stream)
            specs = compile_candidates(init_codes)
            if not specs:
                print(init_codes)
                raise RuntimeError("LLM produced no valid initial candidates. Check provider, API key, and model.")
            new_llm_specs = 0
            for s in specs:
                alg = extract_algorithm(s.code) if cfg.enable_thought else None
                h = compute_code_hash(s.code)
                if cfg.dedup_within_pop and h in seen_pop:
                    continue
                if cfg.dedup_global and h in seen_global:
                    continue
                seen_pop.add(h)
                seen_global.add(h)
                init_cands.append(
                    Candidate(spec=s, gamma=_init_gamma(cfg), algorithm=alg, code_hash=h)
                )
                new_llm_specs += 1
            if debug:
                print(
                    f"[Evo] Init pop {pop_idx}: seeds_used={seeds_used}, new_from_LLM={new_llm_specs}, "
                    f"total_init_cands={len(init_cands)}",
                    flush=True,
                )
        # Evaluate and keep top-N.
        # If a listwise dataset is configured, use preference-only fitness; otherwise fall back to MSE-based fitness.
        if getattr(cfg, "listwise_data_path", None):
            scored_init = _evaluate_population_listwise(init_cands, cfg)
        else:
            scored_init = _evaluate_population(init_cands, cfg)
        scored_init.sort(key=lambda x: x[1], reverse=True)
        scored_init = scored_init[:pop_size]
        populations.append(scored_init)
        archive.update(scored_init)
        if cfg.dump_dir:
            _dump_candidates(cfg.dump_dir, scored_init, gen_idx=0)

    # Evolution by generations, population-wise independent selection
    for gen_idx in range(n_gen):
        print(f"Evoving generation {gen_idx}/{n_gen}...", flush=True)
        for pop_idx in range(n_pops):
            scored = populations[pop_idx]
            # Optional: mix elite archive into parent pool by appending pseudo-scored elites
            if cfg.elite_parent_k > 0 and len(archive.entries) > 0:
                # Reuse elite candidates and their actual fitness scores
                k_elite = min(cfg.elite_parent_k, len(archive.entries))
                temp_cands: List[Tuple[Candidate, float]] = []
                for i in range(k_elite):
                    cand_e, score_e = archive.entries[i]
                    temp_cands.append((cand_e, float(score_e)))
                parent_pool = scored + temp_cands
            else:
                parent_pool = scored

            for op, pw in zip(ops, op_weights):
                r = random.random()
                if r > float(pw):
                    if debug:
                        print(
                            f"[Evo] Gen {gen_idx}, pop {pop_idx}, op={op}: skipped (r={r:.3f} > p={float(pw):.3f}).",
                            flush=True,
                        )
                    continue
                # Generate N offspring for this operator using current population parent_pool
                offspring = _propose_offspring_for_operator(parent_pool, cfg, op)
                if not offspring:
                    if debug:
                        print(
                            f"[Evo] Gen {gen_idx}, pop {pop_idx}, op={op}: no offspring generated.",
                            flush=True,
                        )
                    continue
                # Dedup offspring
                filtered: List[Candidate] = []
                seen_pop_hashes = set(c.code_hash for c, _ in scored if c.code_hash)
                for c in offspring:
                    h = c.code_hash or compute_code_hash(c.spec.code)
                    if cfg.dedup_within_pop and h in seen_pop_hashes:
                        continue
                    if cfg.dedup_global and h in seen_global:
                        continue
                    seen_pop_hashes.add(h)
                    seen_global.add(h)
                    filtered.append(c)
                if not filtered:
                    if debug:
                        print(
                            f"[Evo] Gen {gen_idx}, pop {pop_idx}, op={op}: "
                            f"all {len(offspring)} offspring removed by dedup.",
                            flush=True,
                        )
                    continue

                if getattr(cfg, "listwise_data_path", None):
                    off_results = _evaluate_population_listwise(filtered, cfg)
                else:
                    off_results = _evaluate_population(filtered, cfg)
                if debug:
                    print(
                        f"[Evo] Gen {gen_idx}, pop {pop_idx}, op={op}: "
                        f"offspring={len(offspring)}, kept_after_dedup={len(filtered)}, "
                        f"evaluated={len(off_results)}",
                        flush=True,
                    )
                # Merge and keep top-N
                scored.extend(off_results)
                scored.sort(key=lambda x: x[1], reverse=True)
                scored = scored[:pop_size]
                # Optional elite replacement of worst
                if cfg.elite_replace_worst > 0 and len(archive.entries) > 0:
                    k = min(cfg.elite_replace_worst, len(scored), len(archive.entries))
                    # replace last k individuals with elites and their true scores
                    for i in range(k):
                        scored[-(i + 1)] = (
                            archive.entries[i][0],
                            float(archive.entries[i][1]),
                        )
                populations[pop_idx] = scored
                archive.update(scored)
                if cfg.dump_dir:
                    _dump_candidates(cfg.dump_dir, scored, gen_idx=gen_idx + 1)
        if debug:
            # Simple per-generation summary of best fitness per population
            best_scores = [s[0][1] if s else float("-inf") for s in populations]
            print(
                f"[Evo] End gen {gen_idx}: best fitness per pop = {best_scores}",
                flush=True,
            )

    # Collect and return global top
    all_scored: List[Tuple[Candidate, float]] = []
    for s in populations:
        all_scored.extend(s)
    all_scored.sort(key=lambda x: x[1], reverse=True)
    return all_scored


def _evaluate_population(specs: List[Candidate], cfg: EvoConfig) -> List[Tuple[Candidate, float]]:
    """Evaluate a population of candidates using robust offline objectives as fitness.

    Fitness combines:
      - robust point-wise regression (Huber on Phi vs Monte Carlo value)
      - temporal smoothness / consistency of Delta-Phi vs step reward
      - variance of Delta-Phi
      - per-episode rank correlation between -Phi and -V(s)
    """
    if not specs:
        return []

    # Load (and cache) offline trajectories containing states and Monte Carlo values.
    # Support a main path for fitness and optional multiple paths for diagnostics.
    main_path = getattr(cfg, "offline_traj_path", None)
    multi_paths = getattr(cfg, "offline_traj_paths_multi", None) or []
    paths: List[str] = []
    if main_path:
        paths.append(main_path)
    for pth in multi_paths:
        if pth and pth not in paths:
            paths.append(pth)

    offline_sets: Dict[str, Dict[str, object]] = {}
    for pth in paths:
        if not pth:
            continue
        data = None
        if pth in _OFFLINE_TRAJ_CACHE:
            data = _OFFLINE_TRAJ_CACHE[pth]
        elif os.path.exists(pth):
            try:
                data = load_offline_trajectories(pth)
                _OFFLINE_TRAJ_CACHE[pth] = data
            except Exception:
                data = None
        if data is not None:
            offline_sets[pth] = data

    assert offline_sets, f"can't find offline data in any of: {paths if paths else '[none specified]'}"
    # Choose dataset used for fitness (default: cfg.offline_traj_path or first available)
    fitness_path = main_path if (main_path and main_path in offline_sets) else next(iter(offline_sets.keys()))
    fitness_trajs = offline_sets[fitness_path]

    # Weights for combining robust objectives (kept local for now)
    lambda_point = 1.0
    lambda_smooth = 0.1
    lambda_var = 0.01
    w_rank = 0.1

    results: List[Tuple[Candidate, float]] = []
    for c in specs:
        try:
            mse = mse_phi_vs_value(
                c.spec.fn,
                fitness_trajs,
                device=getattr(cfg, "cheap_eval_device", "cpu"),
                batch_states=getattr(cfg, "cheap_eval_batch_states", None),
                target="future_cost",
            )
        except Exception:
            mse = float("inf")

        if not math.isfinite(mse) or mse <= 0.0:
            fitness = 0.0
        else:
            # MSE 的倒数作为适应度
            fitness = 1.0 / float(mse)

        # Attach simple stats for logging / dumps
        stats_dict: Dict[str, float] = {}
        # Main dataset metrics
        if math.isfinite(mse) and mse >= 0.0:
            stats_dict["mse"] = float(mse)
            try:
                stats_dict["rmse"] = float(math.sqrt(mse))
            except Exception:
                stats_dict["rmse"] = float("nan")

        # Multi-scale diagnostics: per-dataset MSE/RMSE (e.g., tsp20/tsp50/tsp100).
        for pth, trajs in offline_sets.items():
            try:
                mse_k = mse_phi_vs_value(
                    c.spec.fn,
                    trajs,
                    device=getattr(cfg, "cheap_eval_device", "cpu"),
                    batch_states=getattr(cfg, "cheap_eval_batch_states", None),
                    target="future_cost",
                )
            except Exception:
                mse_k = float("inf")
            if not math.isfinite(mse_k) or mse_k < 0.0:
                continue
            # Derive a short label from meta.num_loc if available, else from filename.
            label = None
            try:
                meta = trajs.get("meta", {}) if isinstance(trajs, dict) else {}
                num_loc = meta.get("num_loc", None)
                if isinstance(num_loc, int):
                    label = f"tsp{num_loc}"
            except Exception:
                label = None
            if not label:
                base = os.path.basename(pth)
                label = os.path.splitext(base)[0]
            key_mse = f"mse_{label}"
            key_rmse = f"rmse_{label}"
            stats_dict[key_mse] = float(mse_k)
            try:
                stats_dict[key_rmse] = float(math.sqrt(mse_k))
            except Exception:
                stats_dict[key_rmse] = float("nan")

        # Override fitness: use worst-case (largest) MSE across all scales.
        mse_worst = None
        for key, val in stats_dict.items():
            if not key.startswith("mse"):
                continue
            if key in ("mse_worst",):
                continue
            try:
                v = float(val)
            except Exception:
                continue
            if not math.isfinite(v) or v < 0.0:
                continue
            if mse_worst is None or v > mse_worst:
                mse_worst = v
        if mse_worst is not None:
            fitness = 1.0 / float(mse_worst)
            stats_dict["mse_worst"] = float(mse_worst)
            try:
                stats_dict["rmse_worst"] = float(math.sqrt(mse_worst))
            except Exception:
                stats_dict["rmse_worst"] = float("nan")
        else:
            fitness = 0.0

        c.stats = stats_dict if stats_dict else None
        if stats_dict:
            parts = []
            # Always show global mse/rmse first if present
            if "mse" in stats_dict:
                parts.append(f"mse={stats_dict['mse']:.6g}")
            if "rmse" in stats_dict and math.isfinite(stats_dict["rmse"]):
                parts.append(f"rmse={stats_dict['rmse']:.6g}")
            # Then show per-dataset metrics in sorted order
            for k in sorted(stats_dict.keys()):
                if k in ("mse", "rmse"):
                    continue
                v = stats_dict[k]
                try:
                    if math.isfinite(v):
                        parts.append(f"{k}={float(v):.6g}")
                except Exception:
                    parts.append(f"{k}={v}")
            c.stats_text = "; ".join(parts)
        else:
            c.stats_text = None

        # Refine fitness with robust objectives (Huber / smoothness / variance / rank)
        try:
            robj = robust_phi_objectives(
                c.spec.fn,
                fitness_trajs,
                device=getattr(cfg, "cheap_eval_device", "cpu"),
                batch_states=getattr(cfg, "cheap_eval_batch_states", None),
                target="future_cost",
                huber_delta=1.0,
            )
            point_huber = float(robj.get("point_huber", float("inf")))
            smooth_mse = float(robj.get("smooth_mse", float("inf")))
            dphi_var = float(robj.get("dphi_var", float("inf")))
            mean_rank = float(robj.get("mean_spearman", 0.0))

            if math.isfinite(point_huber):
                fitness = (
                    -lambda_point * point_huber
                    - lambda_smooth * smooth_mse
                    - lambda_var * dphi_var
                    + w_rank * mean_rank
                )

            # Augment stats with robust objectives for logging
            if math.isfinite(point_huber):
                c.stats = c.stats or {}
                c.stats["point_huber"] = point_huber
            if math.isfinite(smooth_mse):
                c.stats = c.stats or {}
                c.stats["smooth_mse"] = smooth_mse
            if math.isfinite(dphi_var):
                c.stats = c.stats or {}
                c.stats["dphi_var"] = dphi_var
            if math.isfinite(mean_rank):
                c.stats = c.stats or {}
                c.stats["rank_spearman"] = mean_rank
        except Exception:
            # Fall back to MSE-based fitness if robust evaluation fails
            pass

        results.append((c, float(fitness)))

    return results


def _evaluate_population_listwise(specs: List[Candidate], cfg: EvoConfig) -> List[Tuple[Candidate, float]]:
    """Preference-only fitness based on listwise offline data.

    Uses only within-list ranking labels and does not regress numeric cost-to-go.
    """
    if not specs:
        return []

    path = getattr(cfg, "listwise_data_path", None)
    if not path:
        # Fallback to original MSE-based evaluation if no listwise dataset is configured
        return _evaluate_population(specs, cfg)

    if path in _OFFLINE_TRAJ_CACHE:
        obj = _OFFLINE_TRAJ_CACHE[path]
    elif os.path.exists(path):
        try:
            obj = torch.load(path, map_location="cpu")
            _OFFLINE_TRAJ_CACHE[path] = obj
        except Exception:
            obj = None
    else:
        obj = None

    if not (isinstance(obj, dict) and "coords" in obj):
        # No valid listwise dataset; fall back to MSE-based fitness
        return _evaluate_population(specs, cfg)

    data = obj
    lambda_pair = float(getattr(cfg, "listwise_pair_weight", 0.5))
    max_lists = getattr(cfg, "listwise_max_lists", None)
    dev = getattr(cfg, "cheap_eval_device", "cpu")
    bs = getattr(cfg, "cheap_eval_batch_states", None)

    results: List[Tuple[Candidate, float]] = []
    for c in specs:
        stats_dict: Dict[str, float] = {}
        try:
            obj_metrics = listwise_preference_objectives(
                c.spec.fn,
                data,
                device=dev,
                batch_states=bs,
                max_lists=max_lists,
            )
            top1 = float(obj_metrics.get("top1_acc", 0.0))
            pair_acc = float(obj_metrics.get("pair_acc", 0.0))
            err_rate = float(obj_metrics.get("error_rate", 0.0))

            # Fitness: higher is better; penalty for runtime errors
            penalty = 10.0 * max(0.0, err_rate)
            fitness = top1 + lambda_pair * pair_acc - penalty

            stats_dict["lw_top1"] = top1
            stats_dict["lw_pair"] = pair_acc
            stats_dict["lw_error_rate"] = err_rate
            stats_dict["lw_n_lists"] = float(obj_metrics.get("n_lists", 0.0))
            stats_dict["lw_n_lists_valid"] = float(obj_metrics.get("n_lists_valid", 0.0))
            stats_dict["lw_n_pairs"] = float(obj_metrics.get("n_pairs", 0.0))
        except Exception:
            fitness = -1e6
            stats_dict["lw_error_rate"] = 1.0

        c.stats = stats_dict if stats_dict else None
        if stats_dict:
            parts = []
            if "lw_top1" in stats_dict:
                parts.append(f"lw_top1={stats_dict['lw_top1']:.4f}")
            if "lw_pair" in stats_dict:
                parts.append(f"lw_pair={stats_dict['lw_pair']:.4f}")
            if "lw_error_rate" in stats_dict:
                parts.append(f"lw_err={stats_dict['lw_error_rate']:.4f}")
            c.stats_text = "; ".join(parts)
        else:
            c.stats_text = None

        results.append((c, float(fitness)))

    return results


def _sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)[:120]


def _dump_candidates(dump_dir: str, results: List[Tuple[Candidate, float]], gen_idx: int):
    os.makedirs(dump_dir, exist_ok=True)
    manifest_path = os.path.join(dump_dir, f"gen{gen_idx:02d}.jsonl")
    for i, (cand, score) in enumerate(results):
        base = f"gen{gen_idx:02d}_cand{i:03d}_{_sanitize_filename(cand.spec.name)}_{score:.4f}.py"
        path = os.path.join(dump_dir, base)
        try:
            algorithm = cand.algorithm
            if not algorithm:
                try:
                    algorithm = extract_algorithm(cand.spec.code)
                except Exception:
                    algorithm = None
            code_hash = cand.code_hash if cand.code_hash else compute_code_hash(cand.spec.code)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# score={float(score):.6f}\n")
                if cand.gamma is not None:
                    try:
                        f.write(f"# gamma={float(cand.gamma):.6f}\n")
                    except Exception:
                        pass
                if code_hash:
                    f.write(f"# code_hash={code_hash}\n")
                if cand.stats_text:
                    f.write(f"# stats: {cand.stats_text}\n")
                if algorithm:
                    # keep the one-sentence idea; ensure braces for consistency
                    if not (algorithm.startswith("{") and algorithm.endswith("}")):
                        aline = "# ALGORITHM: {" + algorithm + "}"
                    else:
                        aline = "# ALGORITHM: " + algorithm
                    f.write(aline + "\n")
                f.write(cand.spec.code)
            # append manifest entry
            try:
                import json as _json

                rec = {
                    "gen": int(gen_idx),
                    "index": int(i),
                    "file": base,
                    "score": float(score),
                    "gamma": float(cand.gamma),
                    "code_hash": code_hash,
                    "algorithm": algorithm,
                    "stats": cand.stats if cand.stats is not None else None,
                    "stats_text": cand.stats_text if cand.stats_text is not None else None,
                }
                with open(manifest_path, "a", encoding="utf-8") as mf:
                    mf.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass
        except Exception:
            pass

