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
from typing import List, Optional, Tuple, Dict, Set

from rl4co.heuristic_finder.evaluate import train_fitness_phi_on_tsp20
from rl4co.heuristic_finder.potential_eval import cheap_score_phi, compute_phi_stats
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


@dataclass
class Candidate:
    """Evolutionary individual: potential function + its gamma."""

    spec: PotentialSpec
    gamma: float
    algorithm: Optional[str] = None
    thought: Optional[str] = None
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
    # Level 1 / Level 2 multi-stage evaluation
    offline_traj_path: str = "data/tsp20_offline_trajs.pt"
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
        cand = Candidate(
            spec=PotentialSpec(name=f"seed_{idx}", code=code, fn=fn),
            gamma=gamma_val,
            algorithm=rec.get("algorithm"),
            thought=rec.get("thought"),
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
        alg = None
        if p.algorithm is not None and len(p.algorithm) > 0:
            alg = p.algorithm
        elif p.thought is not None and len(p.thought) > 0:
            alg = p.thought
        else:
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

            # Optional memetic light repair on raw generation
            if cfg.memetic_repair_prob > 0.0 and random.random() < float(cfg.memetic_repair_prob):
                try:
                    repaired: List[str] = []
                    for c in codes:
                        rc = eoh_llm_repair(cfg.ollama_model, c, env_name="tsp", n=1, debug=False, stream=stream)
                        repaired.append(rc[0] if rc else c)
                    codes = repaired
                except Exception:
                    pass

            specs = compile_candidates(codes)
            for sp in specs:
                g = _mutate_gamma(_init_gamma(cfg), cfg) if cfg.mutate_gamma else _init_gamma(cfg)
                th = extract_thought(sp.code) if cfg.enable_thought else None
                alg = th  # align with EoH: keep a separate 'algorithm' field
                ch = compute_code_hash(sp.code)
                out.append(Candidate(spec=sp, gamma=g, algorithm=alg, thought=th, code_hash=ch))
        except Exception:
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
                            # Ensure thought is populated; fall back to extracting from code
                            try:
                                _thought = c.thought if c.thought else extract_thought(c.spec.code)
                            except Exception:
                                _thought = c.thought
                            # Ensure algorithm is present (prefer explicit field, then thought, then extraction)
                            _algorithm = None
                            if c.algorithm is not None and len(c.algorithm) > 0:
                                _algorithm = c.algorithm
                            else:
                                _algorithm = _thought if _thought else extract_thought(c.spec.code)
                            _stats = c.stats if c.stats is not None else None
                            _stats_text = c.stats_text if c.stats_text is not None else None
                            rec = {
                                "score": float(s),
                                "gamma": float(c.gamma),
                                "algorithm": _algorithm,
                                "thought": _thought,
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
            if c.algorithm is not None and len(c.algorithm) > 0:
                alg = c.algorithm
            elif c.thought is not None and len(c.thought) > 0:
                alg = c.thought
            else:
                alg = "(no description)"
            pack.append({"algorithm": alg, "code": c.spec.code})
        return pack


# ---------------- Diversity & Thought helpers -----------------
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


def extract_thought(code: str) -> Optional[str]:
    try:
        # Prefer brace-wrapped form: # THOUGHT: { ... }
        m = re.search(r"^\s*#\s*THOUGHT:\s*\{([^}]*)\}\s*$", code, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
        # Fallback: accept a plain line without braces: # THOUGHT: ...
        m2 = re.search(r"^\s*#\s*THOUGHT:\s*(.+?)\s*$", code, flags=re.IGNORECASE | re.MULTILINE)
        if m2:
            txt = m2.group(1).strip()
            # Strip surrounding braces if present
            if txt.startswith("{") and txt.endswith("}"):
                txt = txt[1:-1].strip()
            return txt
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
    if not cfg.collect_stats:
        return
    for cand, _ in results:
        if cand.stats is not None:
            continue
        try:
            stats = _compute_phi_stats(cand.spec, cand.gamma, batch_size=cfg.stats_batch_size, device=cfg.device)
            cand.stats = stats
            cand.stats_text = _summarize_stats(stats)
        except Exception:
            cand.stats = None
            cand.stats_text = None


def evolution_search(cfg: EvoConfig) -> List[Tuple[Candidate, float]]:
    """EoH-style multi-population evolution loop with diversity, repair, and elite archive.

    - Independent populations with per-pop selection and variation
    - Explicit deduplication via AST-hash within and across populations
    - Optional memetic light-repair operator applied probabilistically
    - Elite archive maintained across generations with optional parent injection
    """
    # Resolve default operators/weights if not provided
    ops = cfg.operators if cfg.operators is not None else ["e1", "e2", "m1", "m2"]
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
            for s in specs:
                th = extract_thought(s.code) if cfg.enable_thought else None
                h = compute_code_hash(s.code)
                if cfg.dedup_within_pop and h in seen_pop:
                    continue
                if cfg.dedup_global and h in seen_global:
                    continue
                seen_pop.add(h)
                seen_global.add(h)
                init_cands.append(
                    Candidate(spec=s, gamma=_init_gamma(cfg), algorithm=th, thought=th, code_hash=h)
                )
        # Evaluate and keep top-N
        scored_init = _evaluate_population(init_cands, cfg)
        scored_init.sort(key=lambda x: x[1], reverse=True)
        scored_init = scored_init[:pop_size]
        # Optional refinement for top-k
        if cfg.refine_top_k and cfg.refine_epochs and cfg.refine_top_k > 0 and cfg.refine_epochs > 0:
            topk = min(int(cfg.refine_top_k), len(scored_init))
            refined: List[Tuple[Candidate, float]] = []
            for i in range(topk):
                cand = scored_init[i][0]
                try:
                    sc_ref = train_fitness_phi_on_tsp20(
                        cand.spec,
                        epochs=int(cfg.refine_epochs),
                        batch_size=cfg.batch_size,
                        train_data_size=cfg.train_size,
                        val_data_size=cfg.val_size,
                        num_starts=cfg.num_starts,
                        device=cfg.device,
                        accelerator="cpu",
                        devices=1,
                        seed=cfg.seed,
                        pbrs_gamma=cand.gamma,
                        reward_scale=cfg.reward_scale,
                        center_dphi=cfg.center_dphi,
                        norm_dphi=cfg.norm_dphi,
                    )
                    # Combine with cheap score if available
                    total = sc_ref
                    if cand.stats is not None:
                        # just keep consistency by reusing cheap weight as 0.0 contribution here
                        pass
                    refined.append((cand, float(total)))
                except Exception:
                    refined.append(scored_init[i])
            # replace top segment
            for i in range(len(refined)):
                scored_init[i] = refined[i]
        _ensure_stats_for(scored_init, cfg)
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
            if cfg.elite_parent_k > 0:
                elite_pack = archive.top_parents_pack(cfg.elite_parent_k)
                # convert elite pack to temporary Candidates for parent pool only
                temp_cands: List[Tuple[Candidate, float]] = []
                for e in elite_pack:
                    try:
                        fn = compile_potential(e["code"])  # reuse compile here directly
                        th = e.get("algorithm", None)
                        ch = compute_code_hash(e["code"])
                        temp_cands.append(
                            (
                                Candidate(
                                    spec=PotentialSpec(name="elite", code=e["code"], fn=fn),
                                    gamma=_init_gamma(cfg),
                                    algorithm=th,
                                    thought=th,
                                    code_hash=ch,
                                ),
                                float("inf"),
                            )
                        )
                    except Exception:
                        continue
                parent_pool = scored + temp_cands
            else:
                parent_pool = scored

            for op, pw in zip(ops, op_weights):
                if random.random() > float(pw):
                    continue
                # Generate N offspring for this operator using current population parent_pool
                offspring = _propose_offspring_for_operator(parent_pool, cfg, op)
                if not offspring:
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
                    continue

                off_results = _evaluate_population(filtered, cfg)
                # Merge and keep top-N
                scored.extend(off_results)
                scored.sort(key=lambda x: x[1], reverse=True)
                scored = scored[:pop_size]
                _ensure_stats_for(scored, cfg)
                # Optional refinement for top-k after merge
                if cfg.refine_top_k and cfg.refine_epochs and cfg.refine_top_k > 0 and cfg.refine_epochs > 0:
                    topk = min(int(cfg.refine_top_k), len(scored))
                    refined: List[Tuple[Candidate, float]] = []
                    for i in range(topk):
                        cand = scored[i][0]
                        try:
                            sc_ref = train_fitness_phi_on_tsp20(
                                cand.spec,
                                epochs=int(cfg.refine_epochs),
                                batch_size=cfg.batch_size,
                                train_data_size=cfg.train_size,
                                val_data_size=cfg.val_size,
                                num_starts=cfg.num_starts,
                                device=cfg.device,
                                accelerator="cpu",
                                devices=1,
                                seed=cfg.seed,
                                pbrs_gamma=cand.gamma,
                                reward_scale=cfg.reward_scale,
                                center_dphi=cfg.center_dphi,
                                norm_dphi=cfg.norm_dphi,
                            )
                            total = sc_ref
                            refined.append((cand, float(total)))
                        except Exception:
                            refined.append(scored[i])
                    for i in range(len(refined)):
                        scored[i] = refined[i]
                # Optional elite replacement of worst
                if cfg.elite_replace_worst > 0 and len(archive.entries) > 0:
                    k = min(cfg.elite_replace_worst, len(scored), len(archive.entries))
                    elites_to_inject = [archive.entries[i][0] for i in range(k)]
                    # replace last k individuals
                    for i in range(k):
                        scored[-(i + 1)] = (elites_to_inject[i], float("inf"))
                populations[pop_idx] = scored
                archive.update(scored)
                if cfg.dump_dir:
                    _dump_candidates(cfg.dump_dir, scored, gen_idx=gen_idx + 1)

    # Collect and return global top
    all_scored: List[Tuple[Candidate, float]] = []
    for s in populations:
        all_scored.extend(s)
    all_scored.sort(key=lambda x: x[1], reverse=True)
    return all_scored


def _worker_eval(args: Tuple[Tuple[str, float], dict, Optional[int]]):
    """Subprocess worker: evaluate one candidate on an assigned GPU (or CPU).
    Returns ((code,gamma), score).
    """
    (code, gamma), cfgd, gpu_id = args
    accelerator = "cpu"
    devices = 1
    device_str = cfgd.get("device", "cpu")
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        accelerator = "gpu"
        devices = 1
        device_str = "cuda"

    try:
        fn = compile_potential(code)
        spec = PotentialSpec(name="worker", code=code, fn=fn)
        sc = train_fitness_phi_on_tsp20(
            spec,
            epochs=cfgd["epochs_per_eval"],
            batch_size=cfgd["batch_size"],
            train_data_size=cfgd["train_size"],
            val_data_size=cfgd["val_size"],
            num_starts=cfgd["num_starts"],
            device=device_str,
            accelerator=accelerator,
            devices=devices,
            seed=cfgd.get("seed", None),
            pbrs_gamma=gamma,
            reward_scale=cfgd.get("reward_scale", None),
            center_dphi=bool(cfgd.get("center_dphi", False)),
            norm_dphi=bool(cfgd.get("norm_dphi", False)),
        )
        tour_len = -float(sc)
        bad_thr = float(cfgd.get("objective_bad_threshold", 4.0))
        if tour_len > bad_thr:
            sc = float("-inf")
        return (code, gamma), sc
    except Exception:
        return (code, gamma), float("-inf")


def _evaluate_population(specs: List[Candidate], cfg: EvoConfig) -> List[Tuple[Candidate, float]]:
    if not specs:
        return []

    # ---------- Level 1: Cheap offline evaluation for all candidates ----------
    offline_trajs = None
    if cfg.offline_traj_path and os.path.exists(cfg.offline_traj_path):
        try:
            offline_trajs = load_offline_trajectories(cfg.offline_traj_path)
        except Exception:
            offline_trajs = None

    meta: Dict[Tuple[str, float], Dict[str, object]] = {}
    cheap_rank_list: List[Tuple[Candidate, float]] = []
    for c in specs:
        cheap_score = 0.0
        stats_dict: Optional[Dict[str, float]] = None
        complexity = 0.0
        if offline_trajs is not None:
            try:
                s, stats, comp = cheap_score_phi(
                    c.spec.fn,
                    offline_trajs,
                    c.gamma,
                    config={
                        "max_step_shaping_ratio": cfg.max_step_shaping_ratio,
                        "max_episode_shaping_ratio": cfg.max_episode_shaping_ratio,
                        "max_var_ratio_shaped_vs_base": cfg.max_var_ratio_shaped_vs_base,
                        "min_abs_dphi_q95": cfg.min_abs_dphi_q95,
                        "complexity_penalty_alpha": cfg.complexity_penalty_alpha,
                        "cheap_eval_device": cfg.cheap_eval_device,
                        "cheap_eval_batch_states": cfg.cheap_eval_batch_states,
                    },
                    code=c.spec.code,
                )
                cheap_score = float(s)
                stats_dict = {
                    "mean_dphi": stats.mean_dphi,
                    "std_dphi": stats.std_dphi,
                    "abs_dphi_q95": stats.abs_dphi_q95,
                    "step_shaping_ratio": stats.step_shaping_ratio,
                    "episode_shaping_ratio": stats.episode_shaping_ratio,
                    "var_ratio_shaped_vs_base": stats.var_ratio_shaped_vs_base,
                    "corr_dphi_future_cost": stats.corr_dphi_future_cost,
                    "corr_phi0_final_reward": stats.corr_phi0_final_reward,
                    "corr_sum_dphi_final_reward": stats.corr_sum_dphi_final_reward,
                }
                complexity = float(comp)
            except Exception:
                cheap_score = float("-1e6")
                stats_dict = None
                complexity = 0.0
        else:
            # No offline data: neutral cheap score
            cheap_score = 0.0
            stats_dict = None
            complexity = 0.0

        c.stats = stats_dict
        c.stats_text = _summarize_stats(stats_dict) if stats_dict else None
        meta[(c.spec.code, c.gamma)] = {
            "cheap_score": cheap_score,
            "complexity": complexity,
            "stats": stats_dict,
        }
        cheap_rank_list.append((c, cheap_score))

    # Filter by cheap_score threshold, then keep top-K for Level 2 RL eval
    if offline_trajs is not None:
        cheap_rank_list = [(c, s) for (c, s) in cheap_rank_list if s >= float(cfg.cheap_filter_threshold)]
        cheap_rank_list.sort(key=lambda x: x[1], reverse=True)
        allow_n = max(1, int(math.ceil((cfg.pop_size if cfg.pop_size is not None else cfg.population_size) * float(cfg.cheap_topk_ratio))))
        allow_n = min(allow_n, int(cfg.max_candidates_rl_eval))
        level2_set: List[Candidate] = [c for c, _ in cheap_rank_list[:allow_n]]
    else:
        # No offline trajectories: evaluate all in Level 2 to preserve original behavior
        level2_set = list(specs)

    # ---------- Level 2: Short RL eval for top-K; others get default poor RL score ----------
    results: List[Tuple[Candidate, float]] = []
    rl_scores: Dict[Tuple[str, float], float] = {}

    def _finalize_score(cand: Candidate, rl_score: float) -> float:
        m = meta.get((cand.spec.code, cand.gamma), {})
        cheap = float(m.get("cheap_score", 0.0))
        # combine
        total = float(rl_score) + float(cfg.cheap_level_weight) * cheap
        # optional soft penalties based on stats
        st = m.get("stats") or {}
        try:
            vr = st.get("var_ratio_shaped_vs_base", None)
            if vr is not None and math.isfinite(vr):
                total -= 0.05 * max(0.0, float(vr) - 1.0)
        except Exception:
            pass
        return float(total)

    # If no GPUs provided, run sequentially
    gpu_ids = cfg.gpu_ids or []
    if not gpu_ids:
        for c in specs:
            if c in level2_set:
                score = train_fitness_phi_on_tsp20(
                    c.spec,
                    epochs=cfg.epochs_per_eval,
                    batch_size=cfg.batch_size,
                    train_data_size=cfg.train_size,
                    val_data_size=cfg.val_size,
                    num_starts=cfg.num_starts,
                    device=cfg.device,
                    accelerator="cpu",
                    devices=1,
                    seed=cfg.seed,
                    pbrs_gamma=c.gamma,
                    reward_scale=cfg.reward_scale,
                    center_dphi=cfg.center_dphi,
                    norm_dphi=cfg.norm_dphi,
                )
                tour_len = -float(score)
                if tour_len > float(cfg.objective_bad_threshold):
                    score = float("-inf")
            else:
                score = float(-1e6)  # default poor RL score for filtered-out candidates
            total = _finalize_score(c, score)
            results.append((c, total))
        return results

    # parallel across provided GPU ids for level2_set only
    cfgd = {
        "epochs_per_eval": cfg.epochs_per_eval,
        "batch_size": cfg.batch_size,
        "train_size": cfg.train_size,
        "val_size": cfg.val_size,
        "num_starts": cfg.num_starts,
        "device": cfg.device,
        "seed": cfg.seed,
        "reward_scale": cfg.reward_scale,
        "center_dphi": cfg.center_dphi,
        "norm_dphi": cfg.norm_dphi,
        "objective_bad_threshold": cfg.objective_bad_threshold,
    }
    key2cand = {(c.spec.code, c.gamma): c for c in level2_set}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx) as ex:
        futs = []
        for i, c in enumerate(level2_set):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            futs.append(ex.submit(_worker_eval, ((c.spec.code, c.gamma), cfgd, gpu_id)))
        for f in as_completed(futs):
            key, score = f.result()
            rl_scores[key] = score

    # Combine all candidates with their RL or default score
    for c in specs:
        key = (c.spec.code, c.gamma)
        rl_sc = rl_scores.get(key, float(-1e6))
        total = _finalize_score(c, rl_sc)
        results.append((c, total))
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
            thought = cand.thought if cand.thought else extract_thought(cand.spec.code)
            algorithm = cand.algorithm if cand.algorithm else thought
            if not thought and algorithm:
                thought = algorithm
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
                if thought:
                    # keep the one-sentence idea; ensure braces for consistency
                    if not (thought.startswith("{") and thought.endswith("}")):
                        tline = "# THOUGHT: {" + thought + "}"
                    else:
                        tline = "# THOUGHT: " + thought
                    f.write(tline + "\n")
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
                    "thought": thought,
                    "stats": cand.stats if cand.stats is not None else None,
                    "stats_text": cand.stats_text if cand.stats_text is not None else None,
                }
                with open(manifest_path, "a", encoding="utf-8") as mf:
                    mf.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass
        except Exception:
            pass
