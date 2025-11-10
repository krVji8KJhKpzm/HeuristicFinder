from __future__ import annotations

import os
import random
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

from rl4co.heuristic_finder.evaluate import train_fitness_phi_on_tsp20
from rl4co.heuristic_finder.llm import (
    eoh_llm_i1,
    eoh_llm_e1,
    eoh_llm_e2,
    eoh_llm_m1,
    eoh_llm_m2,
    eoh_llm_m3,
    eoh_llm_mutate,
    eoh_llm_crossover,
)
from rl4co.heuristic_finder.potential import PotentialSpec, compile_potential
from rl4co.heuristic_finder.diagnostics import (
    profile_phi_on_tsp,
    summarize_profile_for_prompt,
    save_profile_json,
)


@dataclass
class Candidate:
    """Evolutionary individual: potential function + its gamma."""

    spec: PotentialSpec
    gamma: float


@dataclass
class EvoConfig:
    population_size: int = 4
    survivors: int = 2
    iterations: int = 2
    offspring_per_iter: int = 2
    epochs_per_eval: int = 1
    batch_size: int = 64
    train_size: int = 1000
    val_size: int = 256
    num_starts: int = 8
    device: str = "cpu"
    # LLM via Ollama only
    ollama_model: Optional[str] = None  # e.g., 'qwen3:32b'
    # EoH-style evolution controls (always enabled)
    frac_crossover: float = 0.5  # fraction of offspring via crossover
    tournament_k: int = 2  # tournament size for selecting parents
    novelty_weight: float = 0.0  # add small novelty pressure in selection
    # Optional: parallel short-training across multiple GPUs
    gpu_ids: Optional[List[int]] = None
    # Optional dir to dump all candidate codes per generation
    dump_dir: Optional[str] = None
    # Optional fixed seed for reproducible short-training
    seed: Optional[int] = None
    pbrs_gamma_choices: Optional[List[float]] = None
    reward_scale: Optional[str] = None
    center_dphi: bool = False
    norm_dphi: bool = False
    # Penalize candidates whose estimated tour length exceeds this threshold
    objective_bad_threshold: float = 5.0
    # Selection controls
    include_parents: bool = True  # (mu + lambda) selection; parents compete with offspring
    offspring_factor: int = 6  # keep top (offspring_factor * N) offspring before selection
    # Optional: run diagnostics and feed summary to LLM prompts
    diag_for_llm: bool = False
    diag_num_loc: Optional[List[int]] = None  # e.g., [20, 50]
    diag_batches_per_n: int = 2
    diag_batch_size: int = 128
    diag_gamma: float = 1.0
    diag_device: Optional[str] = None  # default to cfg.device


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


def propose_offspring(
    parents: List[Candidate],
    cfg: EvoConfig,
    eval_summaries: Optional[dict] = None,
) -> List[Candidate]:
    """Generate offspring using EoH ops on code, and mutate gamma as part of genome."""
    offspring: List[Candidate] = []

    if not cfg.ollama_model:
        print("[HeuristicFinder] No --ollama-model provided; using DeepSeek API fallback.", flush=True)

    # Helper to build a small parent pack with 'p' first for diversity
    def build_pack(p: Candidate, all_parents: List[Candidate], k_ctx: int = 3):
        pool = [q for q in all_parents if q is not p]
        ctx = random.sample(pool, k=min(max(k_ctx - 1, 0), len(pool))) if pool else []
        ordered = [p] + ctx
        return [{"algorithm": "(no description)", "code": sp.spec.code} for sp in ordered]

    for p in parents:
        # Retrieve optional eval summary for this parent
        p_summary = None
        if eval_summaries is not None:
            p_summary = eval_summaries.get(p.spec.code)
        try:
            # e1
            pack = build_pack(p, parents, k_ctx=3)
            codes = eoh_llm_e1(cfg.ollama_model, pack, n=1, env_name="tsp", debug=True)
            for sp in compile_candidates(codes):
                offspring.append(Candidate(spec=sp, gamma=_mutate_gamma(p.gamma, cfg)))
        except Exception:
            pass
        try:
            # e2
            pack = build_pack(p, parents, k_ctx=3)
            codes = eoh_llm_e2(cfg.ollama_model, pack, n=1, env_name="tsp", debug=True)
            for sp in compile_candidates(codes):
                offspring.append(Candidate(spec=sp, gamma=_mutate_gamma(p.gamma, cfg)))
        except Exception:
            pass
        try:
            # m1
            codes = eoh_llm_m1(cfg.ollama_model, p.spec.code, n=1, env_name="tsp", debug=True)
            for sp in compile_candidates(codes):
                offspring.append(Candidate(spec=sp, gamma=_mutate_gamma(p.gamma, cfg)))
        except Exception:
            pass
        try:
            # m2
            codes = eoh_llm_m2(cfg.ollama_model, p.spec.code, n=1, env_name="tsp", debug=True)
            for sp in compile_candidates(codes):
                offspring.append(Candidate(spec=sp, gamma=_mutate_gamma(p.gamma, cfg)))
        except Exception:
            pass
        try:
            # m3
            codes = eoh_llm_m3(cfg.ollama_model, p.spec.code, n=1, env_name="tsp", debug=True)
            for sp in compile_candidates(codes):
                offspring.append(Candidate(spec=sp, gamma=_mutate_gamma(p.gamma, cfg)))
        except Exception:
            pass

        # Reflection-guided mutate/crossover using eval summary (if provided)
        if p_summary:
            try:
                # mutate with eval summary
                codes = eoh_llm_mutate(
                    model=cfg.ollama_model or "",
                    parent_code=p.spec.code,
                    env_name="tsp",
                    guidance="Prefer low-variance, N-invariant features; normalize by graph_scale; avoid heavy branching.",
                    eval_summary=p_summary,
                    n=1,
                    debug=True,
                )
                for sp in compile_candidates(codes):
                    offspring.append(Candidate(spec=sp, gamma=_mutate_gamma(p.gamma, cfg)))
            except Exception:
                pass
            try:
                # crossover with a random other parent + merged summaries
                pool = [q for q in parents if q is not p]
                if pool:
                    q = random.choice(pool)
                    q_summary = eval_summaries.get(q.spec.code, "") if eval_summaries else ""
                    merged_summary = f"Parent A summary:\n{p_summary}\nParent B summary:\n{q_summary}"
                    codes = eoh_llm_crossover(
                        model=cfg.ollama_model or "",
                        parent_a=p.spec.code,
                        parent_b=q.spec.code,
                        env_name="tsp",
                        guidance="Blend complementary, stable cues; keep output scale moderate; ensure broadcastability [B,1].",
                        eval_summary=merged_summary,
                        n=1,
                        debug=True,
                    )
                    for sp in compile_candidates(codes):
                        offspring.append(Candidate(spec=sp, gamma=_mutate_gamma(p.gamma, cfg)))
            except Exception:
                pass

        # gamma-only mutation (keep code)
        try:
            offspring.append(Candidate(spec=p.spec, gamma=_mutate_gamma(p.gamma, cfg)))
        except Exception:
            pass

    return offspring


def evolution_search(cfg: EvoConfig) -> List[Tuple[Candidate, float]]:
    # init population via LLM i1 (EoH style) using Ollama if provided, otherwise DeepSeek API
    init_codes = eoh_llm_i1(cfg.ollama_model, n=cfg.population_size, env_name="tsp", debug=False)
    specs = compile_candidates(init_codes)
    if not specs:
        print(init_codes)
        raise RuntimeError("LLM produced no valid initial candidates. Check provider, API key, and model.")

    def _init_gamma() -> float:
        if cfg.pbrs_gamma_choices and len(cfg.pbrs_gamma_choices) > 0:
            return random.choice(cfg.pbrs_gamma_choices)
        return 1.0

    population: List[Candidate] = [Candidate(spec=s, gamma=_init_gamma()) for s in specs]
    scored: List[Tuple[Candidate, float]] = []

    # novelty archive (store fingerprints of best-so-far)
    archive: List[Tuple[str, set]] = []  # (code, fingerprint)

    # evaluate initial population (possibly parallel)
    init_results = _evaluate_population(population, cfg)
    scored.extend(init_results)
    if cfg.dump_dir:
        _dump_candidates(cfg.dump_dir, init_results, gen_idx=0)

    # iterate
    for gen_idx in range(cfg.iterations):
        # Use the entire current population as parents (size N)
        scored.sort(key=lambda x: x[1], reverse=True)
        parents = [s for s, _ in scored[: cfg.population_size]]

        # Optional: diagnostics for each parent and summary to feed LLM
        eval_summaries = None
        if cfg.diag_for_llm:
            eval_summaries = {}
            # prepare diagnostics params
            sizes = cfg.diag_num_loc if (cfg.diag_num_loc and len(cfg.diag_num_loc) > 0) else [20, 50]
            for pi, p in enumerate(parents):
                try:
                    prof = profile_phi_on_tsp(
                        p.spec,
                        num_loc_list=sizes,
                        batches_per_n=cfg.diag_batches_per_n,
                        batch_size=cfg.diag_batch_size,
                        device=cfg.diag_device or cfg.device,
                        seed=cfg.seed,
                        gamma=cfg.diag_gamma,
                    )
                    summary = summarize_profile_for_prompt(prof)
                    eval_summaries[p.spec.code] = summary
                    if cfg.dump_dir:
                        os.makedirs(cfg.dump_dir, exist_ok=True)
                        fname = f"gen{gen_idx:02d}_parent{pi:03d}_diag.json"
                        save_profile_json(prof, os.path.join(cfg.dump_dir, fname))
                except Exception:
                    continue

        # propose offspring: baseline ops + (optional) reflection-guided ops
        offspring = propose_offspring(parents, cfg, eval_summaries=eval_summaries)

        # evaluate offspring (possibly parallel)
        off_results = _evaluate_population(offspring, cfg)
        if cfg.dump_dir:
            _dump_candidates(cfg.dump_dir, off_results, gen_idx=gen_idx + 1)

        # Optionally keep only top-K offspring before combining with parents
        if off_results:
            if cfg.novelty_weight > 0:
                ranked_off = _rank_with_novelty(off_results, archive, cfg.novelty_weight)
            else:
                ranked_off = sorted(off_results, key=lambda x: x[1], reverse=True)
            k_off = max(cfg.population_size * max(1, int(cfg.offspring_factor)), cfg.population_size)
            ranked_off = ranked_off[: k_off]
        else:
            ranked_off = []

        # (mu + lambda) selection: combine parents and offspring, then select top-N
        combined = list(ranked_off)
        if cfg.include_parents and scored:
            combined.extend(scored)

        if cfg.novelty_weight > 0 and combined:
            next_scored = _rank_with_novelty(combined, archive, cfg.novelty_weight)[: cfg.population_size]
        else:
            next_scored = sorted(combined, key=lambda x: x[1], reverse=True)[: cfg.population_size]
        scored = next_scored

        # update novelty archive with current best of new generation
        if scored:
            for cand, _ in scored[: cfg.population_size]:
                fp = _code_fingerprint(cand.spec.code)
                archive.append((cand.spec.code, fp))
            if len(archive) > 200:
                archive = archive[-200:]

    # final ranking
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


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


# ----------------- Novelty helpers (lightweight) -----------------
def _code_fingerprint(code: str) -> set:
    """Very simple AST-based fingerprint: set of identifier and attribute names.

    Used to compute a rough Jaccard-based novelty score.
    """
    try:
        import ast

        tree = ast.parse(code)
        toks = set()
        for node in ast.walk(tree):
            if hasattr(node, "id") and isinstance(getattr(node, "id"), str):
                toks.add(getattr(node, "id"))
            if hasattr(node, "attr") and isinstance(getattr(node, "attr"), str):
                toks.add(getattr(node, "attr"))
        return toks
    except Exception:
        # fallback to tokenized words
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _novelty(code: str, archive: List[Tuple[str, set]]) -> float:
    fp = _code_fingerprint(code)
    if not archive:
        return 1.0
    sim = max((_jaccard(fp, bfp) for _, bfp in archive), default=0.0)
    return 1.0 - sim


def _rank_with_novelty(
    scored: List[Tuple[Candidate, float]],
    archive: List[Tuple[str, set]],
    novelty_weight: float,
) -> List[Tuple[Candidate, float]]:
    # Sort by (fitness + novelty_weight * novelty)
    decorated = []
    for cand, fit in scored:
        nv = _novelty(cand.spec.code, archive)
        decorated.append((cand, fit + novelty_weight * nv))
    decorated.sort(key=lambda x: x[1], reverse=True)
    return decorated


def _sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)[:120]


def _dump_candidates(dump_dir: str, results: List[Tuple[Candidate, float]], gen_idx: int):
    os.makedirs(dump_dir, exist_ok=True)
    for i, (cand, score) in enumerate(results):
        base = f"gen{gen_idx:02d}_cand{i:03d}_{_sanitize_filename(cand.spec.name)}_{score:.4f}.py"
        path = os.path.join(dump_dir, base)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# score={score:.6f}\n")
                f.write(cand.spec.code)
        except Exception:
            pass
