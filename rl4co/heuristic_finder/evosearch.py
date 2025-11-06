from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from rl4co.heuristic_finder.evaluate import train_fitness_phi_on_tsp20
from rl4co.heuristic_finder.llm import (
    eoh_llm_i1,
    eoh_llm_e1,
    eoh_llm_e2,
    eoh_llm_m1,
    eoh_llm_m2,
    eoh_llm_m3,
)
from rl4co.heuristic_finder.potential import PotentialSpec, compile_potential, seed_potentials


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
    frac_crossover: float = 0.5         # fraction of offspring via crossover
    tournament_k: int = 2               # tournament size for selecting parents
    novelty_weight: float = 0.0         # add small novelty pressure in selection
    # Optional: parallel short-training across multiple GPUs
    gpu_ids: Optional[List[int]] = None
    # Optional dir to dump all candidate codes per generation
    dump_dir: Optional[str] = None
    # Optional fixed seed for reproducible short-training
    seed: Optional[int] = None


    


def make_population_from_seeds(k: int) -> List[PotentialSpec]:
    seeds = list(seed_potentials().values())
    return seeds[:k] if k <= len(seeds) else seeds


def compile_candidates(codes: List[str]) -> List[PotentialSpec]:
    out: List[PotentialSpec] = []
    for i, code in enumerate(codes):
        try:
            fn = compile_potential(code)
        except Exception:
            continue
        out.append(PotentialSpec(name=f"llm_{i}", code=code, fn=fn))
    return out


def propose_offspring(
    parents: List[PotentialSpec], cfg: EvoConfig
) -> List[PotentialSpec]:
    offspring: List[PotentialSpec] = []

    if not cfg.ollama_model:
        print("[HeuristicFinder] Missing --ollama-model; cannot propose offspring.", flush=True)
        return offspring

    # 1) EoH-style LLM proposals via Ollama: e1, e2, m1, m2, m3
    if cfg.ollama_model:
        # tournament selection helper
        def pick_parent() -> PotentialSpec:
            if cfg.tournament_k <= 1 or len(parents) <= 1:
                return random.choice(parents)
            cand = random.sample(parents, k=min(cfg.tournament_k, len(parents)))
            return max(cand, key=lambda s: len(s.code))

        ops = ["e1", "e2", "m1", "m2", "m3"]
        i = 0
        while len(offspring) < cfg.offspring_per_iter:
            op = ops[i % len(ops)]
            i += 1
            try:
                if op in ("e1", "e2") and len(parents) >= 2:
                    # package parents as list of dicts expected by EoH prompts
                    k = min(3, len(parents))
                    ps = parents[:k]
                    pack = [{"algorithm": "(no description)", "code": p.code} for p in ps]
                    if op == "e1":
                        codes = eoh_llm_e1(cfg.ollama_model, pack, n=1, env_name="tsp", debug=True)
                    else:
                        codes = eoh_llm_e2(cfg.ollama_model, pack, n=1, env_name="tsp", debug=True)
                    offspring.extend(compile_candidates(codes))
                elif op == "m1":
                    pa = pick_parent()
                    codes = eoh_llm_m1(cfg.ollama_model, pa.code, n=1, env_name="tsp", debug=True)
                    offspring.extend(compile_candidates(codes))
                elif op == "m2":
                    pa = pick_parent()
                    codes = eoh_llm_m2(cfg.ollama_model, pa.code, n=1, env_name="tsp", debug=True)
                    offspring.extend(compile_candidates(codes))
                elif op == "m3":
                    pa = pick_parent()
                    codes = eoh_llm_m3(cfg.ollama_model, pa.code, n=1, env_name="tsp", debug=True)
                    offspring.extend(compile_candidates(codes))
            except Exception:
                # ignore failures for robustness
                continue

    return offspring


def evolution_search(cfg: EvoConfig) -> List[Tuple[PotentialSpec, float]]:
    # init population via LLM i1 (EoH style)
    if not cfg.ollama_model:
        raise RuntimeError("--ollama-model is required for LLM-based initialization.")
    init_codes = eoh_llm_i1(cfg.ollama_model, n=cfg.population_size, env_name="tsp", debug=True)
    population = compile_candidates(init_codes)
    if not population:
        print(init_codes)
        raise RuntimeError("LLM produced no valid initial candidates. Check Ollama and model.")
    scored: List[Tuple[PotentialSpec, float]] = []

    # novelty archive (store fingerprints of best-so-far)
    archive: List[Tuple[str, set]] = []  # (code, fingerprint)

    # evaluate initial population (possibly parallel)
    init_results = _evaluate_population(population, cfg)
    scored.extend(init_results)
    if cfg.dump_dir:
        _dump_candidates(cfg.dump_dir, init_results, gen_idx=0)

    # iterate
    for _ in range(cfg.iterations):
        # select survivors (higher reward plus optional novelty)
        if cfg.novelty_weight > 0 and scored:
            ranked = _rank_with_novelty(scored, archive, cfg.novelty_weight)
            survivors = [s for s, _ in ranked[: cfg.survivors]]
            scored = ranked[: cfg.population_size]
        else:
            scored.sort(key=lambda x: x[1], reverse=True)
            survivors = [s for s, _ in scored[: cfg.survivors]]

        # propose offspring
        offspring = propose_offspring(survivors, cfg)
        # evaluate offspring (possibly parallel)
        off_results = _evaluate_population(offspring, cfg)
        scored.extend(off_results)
        if cfg.dump_dir:
            _dump_candidates(cfg.dump_dir, off_results, gen_idx=_ + 1)

        # keep top-K as new population (raw fitness or novelty-adjusted)
        if cfg.novelty_weight > 0 and scored:
            scored = _rank_with_novelty(scored, archive, cfg.novelty_weight)[: cfg.population_size]
        else:
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[: cfg.population_size]

        # update novelty archive with current best
        if scored:
            for spec, _ in scored[: cfg.survivors]:
                fp = _code_fingerprint(spec.code)
                archive.append((spec.code, fp))
            # cap archive size
            if len(archive) > 200:
                archive = archive[-200:]

    # final ranking
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _worker_eval(args: Tuple[str, dict, Optional[int]]):
    """Subprocess worker: evaluate one candidate on an assigned GPU (or CPU). Returns (code, score)."""
    code, cfgd, gpu_id = args
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
        score = train_fitness_phi_on_tsp20(
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
        )
        return code, score
    except Exception:
        return code, float("-inf")


def _evaluate_population(specs: List[PotentialSpec], cfg: EvoConfig) -> List[Tuple[PotentialSpec, float]]:
    if not specs:
        return []

    code2spec = {s.code: s for s in specs}
    results: List[Tuple[PotentialSpec, float]] = []

    gpu_ids = cfg.gpu_ids or []
    if not gpu_ids:
        # sequential
        for s in specs:
            score = train_fitness_phi_on_tsp20(
                s,
                epochs=cfg.epochs_per_eval,
                batch_size=cfg.batch_size,
                train_data_size=cfg.train_size,
                val_data_size=cfg.val_size,
                num_starts=cfg.num_starts,
                device=cfg.device,
                accelerator="cpu",
                devices=1,
                seed=cfg.seed,
            )
            results.append((s, score))
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
    }
    # use 'spawn' to avoid CUDA + fork issues on Linux
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx) as ex:
        futs = []
        for i, s in enumerate(specs):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            futs.append(ex.submit(_worker_eval, (s.code, cfgd, gpu_id)))
        for f in as_completed(futs):
            code, score = f.result()
            spec = code2spec.get(code)
            if spec is not None:
                results.append((spec, score))
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
        import re

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
    scored: List[Tuple[PotentialSpec, float]],
    archive: List[Tuple[str, set]],
    novelty_weight: float,
) -> List[Tuple[PotentialSpec, float]]:
    # Sort by (fitness + novelty_weight * novelty)
    decorated = []
    for spec, fit in scored:
        nv = _novelty(spec.code, archive)
        decorated.append((spec, fit + novelty_weight * nv))
    decorated.sort(key=lambda x: x[1], reverse=True)
    return decorated


def _sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)[:120]


def _dump_candidates(dump_dir: str, results: List[Tuple[PotentialSpec, float]], gen_idx: int):
    os.makedirs(dump_dir, exist_ok=True)
    for i, (spec, score) in enumerate(results):
        base = f"gen{gen_idx:02d}_cand{i:03d}_{_sanitize_filename(spec.name)}_{score:.4f}.py"
        path = os.path.join(dump_dir, base)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# score={score:.6f}\n")
                f.write(spec.code)
        except Exception:
            pass
