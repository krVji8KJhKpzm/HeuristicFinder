from __future__ import annotations

import os
import random
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set

from rl4co.heuristic_finder.evaluate import train_fitness_phi_on_tsp20
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
    thought: Optional[str] = None
    code_hash: Optional[str] = None


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
    tournament_k: int = 2  # tournament size
    # Optional: parallel short-training across multiple GPUs
    gpu_ids: Optional[List[int]] = None
    # Optional dir to dump all candidate codes per generation
    dump_dir: Optional[str] = None
    # Optional fixed seed for reproducible short-training
    seed: Optional[int] = None
    # PBRS controls
    pbrs_gamma_choices: Optional[List[float]] = None
    reward_scale: Optional[str] = None
    center_dphi: bool = False
    norm_dphi: bool = False
    # Penalize candidates whose estimated tour length exceeds this threshold
    objective_bad_threshold: float = 5.0
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
    elite_parent_k: int = 0
    elite_replace_worst: int = 0


def compile_candidates(codes: List[str]) -> List[PotentialSpec]:
    out: List[PotentialSpec] = []
    for i, code in enumerate(codes):
        try:
            fn = compile_potential(code)
        except Exception:
            continue
        out.append(PotentialSpec(name=f"llm_{i}", code=code, fn=fn))
    return out


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
        alg = p.thought if (p.thought is not None and len(p.thought) > 0) else "(no description)"
        pack.append({"algorithm": alg, "code": p.spec.code})
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

    for _ in range(N):
        try:
            if operator in ("e1", "e2"):
                pars = _tournament_select(scored, m=cfg.m_parents, k=cfg.tournament_k)
                pack = _parents_pack(pars)
                if operator == "e1":
                    codes = eoh_llm_e1(cfg.ollama_model, pack, n=1, env_name="tsp", debug=False)
                else:
                    codes = eoh_llm_e2(cfg.ollama_model, pack, n=1, env_name="tsp", debug=False)
            elif operator == "m1":
                par = _tournament_select(scored, m=1, k=cfg.tournament_k)[0]
                codes = eoh_llm_m1(cfg.ollama_model, par.spec.code, n=1, env_name="tsp", debug=False)
            elif operator == "m2":
                par = _tournament_select(scored, m=1, k=cfg.tournament_k)[0]
                codes = eoh_llm_m2(cfg.ollama_model, par.spec.code, n=1, env_name="tsp", debug=False)
            elif operator == "m3":
                par = _tournament_select(scored, m=1, k=cfg.tournament_k)[0]
                codes = eoh_llm_m3(cfg.ollama_model, par.spec.code, n=1, env_name="tsp", debug=False)
            else:
                # unknown operator: skip
                continue

            # Optional memetic light repair on raw generation
            if cfg.memetic_repair_prob > 0.0 and random.random() < float(cfg.memetic_repair_prob):
                try:
                    repaired: List[str] = []
                    for c in codes:
                        rc = eoh_llm_repair(cfg.ollama_model, c, env_name="tsp", n=1, debug=False)
                        repaired.append(rc[0] if rc else c)
                    codes = repaired
                except Exception:
                    pass

            specs = compile_candidates(codes)
            for sp in specs:
                g = _mutate_gamma(_init_gamma(cfg), cfg) if cfg.mutate_gamma else _init_gamma(cfg)
                th = extract_thought(sp.code) if cfg.enable_thought else None
                ch = compute_code_hash(sp.code)
                out.append(Candidate(spec=sp, gamma=g, thought=th, code_hash=ch))
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
                            rec = {
                                "score": float(s),
                                "gamma": float(c.gamma),
                                "thought": c.thought,
                                "code_hash": c.code_hash or compute_code_hash(c.spec.code),
                                "code": c.spec.code,
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
            alg = c.thought if (c.thought is not None and len(c.thought) > 0) else "(no description)"
            pack.append({"algorithm": alg, "code": c.spec.code})
        return pack


# ---------------- Diversity & Thought helpers -----------------
import ast
import hashlib


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
        m = re.search(r"^\s*#\s*THOUGHT:\s*\{([^}]*)\}\s*$", code, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    except Exception:
        return None
    return None


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

    # Initialize populations independently
    populations: List[List[Tuple[Candidate, float]]] = []
    for pop_idx in range(n_pops):
        total_init = pop_size * max(1, int(cfg.initial_copies))
        init_codes = eoh_llm_i1(cfg.ollama_model, n=total_init, env_name="tsp", debug=False)
        specs = compile_candidates(init_codes)
        if not specs:
            raise RuntimeError("LLM produced no valid initial candidates. Check provider, API key, and model.")
        init_cands: List[Candidate] = []
        seen_pop: Set[str] = set()
        for s in specs:
            th = extract_thought(s.code) if cfg.enable_thought else None
            h = compute_code_hash(s.code)
            if cfg.dedup_within_pop and h in seen_pop:
                continue
            if cfg.dedup_global and h in seen_global:
                continue
            seen_pop.add(h)
            seen_global.add(h)
            init_cands.append(Candidate(spec=s, gamma=_init_gamma(cfg), thought=th, code_hash=h))
        # Evaluate and keep top-N
        scored_init = _evaluate_population(init_cands, cfg)
        scored_init.sort(key=lambda x: x[1], reverse=True)
        scored_init = scored_init[:pop_size]
        populations.append(scored_init)
        archive.update(scored_init)
        if cfg.dump_dir:
            _dump_candidates(cfg.dump_dir, scored_init, gen_idx=0)

    # Evolution by generations, population-wise independent selection
    for gen_idx in range(n_gen):
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

    key2cand = {(c.spec.code, c.gamma): c for c in specs}
    results: List[Tuple[Candidate, float]] = []

    gpu_ids = cfg.gpu_ids or []
    if not gpu_ids:
        # sequential
        for c in specs:
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
            results.append((c, score))
        return results

    # parallel across provided GPU ids
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
    # use 'spawn' to avoid CUDA + fork issues on Linux/Windows
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx) as ex:
        futs = []
        for i, c in enumerate(specs):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            futs.append(ex.submit(_worker_eval, ((c.spec.code, c.gamma), cfgd, gpu_id)))
        for f in as_completed(futs):
            key, score = f.result()
            cand = key2cand.get(key)
            if cand is not None:
                results.append((cand, score))
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
                    "thought": thought,
                }
                with open(manifest_path, "a", encoding="utf-8") as mf:
                    mf.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass
        except Exception:
            pass
