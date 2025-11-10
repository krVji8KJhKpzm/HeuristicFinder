from __future__ import annotations

from typing import List, Optional, Dict
import os
import json
import time


def format_prompt(env_name: str = "tsp", guidance: str = "") -> str:
    return (
        "You are designing a potential function Phi(state) for PBRS in "
        f"combinatorial optimization env '{env_name}'.\n"
        "Output format (strict):\n"
        "- Return ONLY a single fenced code block starting with: ```python and ending with: ```\n"
        "- The code must define exactly one function: def phi(state): and nothing else.\n"
        "- Use only torch ops; no prints, no explanations, no comments outside code.\n"
        "- Ensure result is broadcastable to [B,1]; handle NaNs via torch.nan_to_num.\n"
        "Goal: robust, node-count-invariant outputs. Use reductions (mean/max/min/std/softmin) over N-dependent tensors.\n"
        "Avoid Python loops; prefer vectorized torch ops.\n"
        "Available state helpers (batch-friendly):\n"
        "Raw N-dependent (use with reductions): action_mask() -> [B,N] (True=unvisited); visited_mask() -> [B,N];\n"
        "  current_node_index() -> [B]; first_node_index() -> [B]; distances_from_current(normalize=True) -> [B,N];\n"
        "  distance_matrix(normalize=True) -> [B,N,N] (diag=0).\n"
        "Return a tensor broadcastable to [B,1]. Keep it simple and stable.\n"
        + guidance
    )

# "Progress scalars: remaining_ratio() -> [B,1]; visited_ratio() -> [B,1]; step_ratio() -> [B,1]\n"
# "Geometry (fixed-size): graph_scale() -> [B,1]; current_loc() -> [B,2]; start_loc() -> [B,2];\n"
# "  nearest_unvisited_distance(normalize=True) -> [B,1]; k_nearest_unvisited(k=3) -> [B,k]; k_farthest_unvisited(k=3) -> [B,k];\n"
# "  centroid_unvisited() -> [B,2]; distance_to_centroid(normalize=True) -> [B,1]; distance_to_start(normalize=True) -> [B,1];\n"
# "  mean_unvisited_distance(normalize=True) -> [B,1]; max_unvisited_distance(normalize=True) -> [B,1]; std_unvisited_distance(normalize=True) -> [B,1]\n"

def _strip_think_tags(text: str) -> str:
    """Remove '<think> ... </think>' blocks that some models emit."""
    try:
        import re
        return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    except Exception:
        return text


def _extract_code_block(text: str) -> str:
    """Extract code from Markdown-style fenced blocks if present; otherwise return raw text."""
    if not text:
        return text
    # look for ```python ... ``` or ``` ... ```
    start = None
    lang_markers = ("```python", "```py", "```")
    for lm in lang_markers:
        idx = text.find(lm)
        if idx != -1:
            start = idx + len(lm)
            break
    if start is None:
        return text
    end = text.find("```", start)
    if end == -1:
        return text
    return text[start:end].strip()


def _extract_phi_from_text(text: str) -> str:
    """Robust fallback: extract a `def phi(...):` function from mixed text.

    Strategy:
    1) Find the first occurrence of `def phi(` and return until the next triple backticks or end of text.
    2) If not found, fallback to a broad 'def ... return' span similar to EoH.
    """
    try:
        import re
        m = re.search(r"def\s+phi\s*\(.*?\):", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            tail = text[m.start():]
            # cut at next fenced block end if present
            fence = tail.find("```")
            if fence != -1:
                tail = tail[:fence]
            return tail.strip()
        # EoH-like coarse capture: from first 'def' to last 'return'
        ms = re.findall(r"def[\s\S]*?return[\s\S]*", text)
        if ms:
            return ms[0].strip()
    except Exception:
        pass
    return text


def _maybe_dump(kind: str, content: str, suffix: str = ".txt") -> None:
    """Optionally dump LLM I/O to disk for debugging.

    Set env var `LLM_DUMP_DIR` to a directory path to enable.
    Files are named as `<timestamp>_<kind><suffix>`.
    """
    dump_dir = os.environ.get("LLM_DUMP_DIR")
    if not dump_dir:
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(dump_dir, f"{ts}_{kind}{suffix}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else str(content))
    except Exception:
        # best-effort only
        pass


def _looks_like_phi(code: Optional[str]) -> bool:
    if not isinstance(code, str):
        return False
    s = code.strip()
    return ("def phi(" in s) and ("return" in s)


def _build_repair_instruction(raw: str) -> str:
    return (
        "Convert the following content into ONLY a Python fenced code block implementing a single function '\n"
        "def phi(state):\n' using torch ops. Requirements:\n"
        "- Return ONLY one fenced block: ```python ... ``` with no extra text.\n"
        "- Use provided state helpers conceptually referenced; distances with normalize=True.\n"
        "- Output must be broadcastable to [B,1]; handle NaNs via torch.nan_to_num.\n"
        "- Keep it stable and node-count-invariant.\n"
        "Content begins:\n" + raw + "\nContent ends."
    )


def _reasoner_context(env_name: str = "tsp") -> str:
    """Build a neutral, spec-only context for the reasoner stage without code-only directives."""
    parts = _phi_prompt_parts(env_name)
    goal = (
        "Design a potential function Phi(state) for PBRS in the given environment.\n"
        "Constraints: node-count-invariant, use normalized distances, output broadcastable to [B,1],\n"
        "handle NaNs with torch.nan_to_num, and prefer simple/stable formulations.\n"
    )
    ctx = (
        goal
        + "Available state helpers (batch-friendly):\n"
        + parts["inout_inf"]
    )
    return ctx


def generate_candidates_via_ollama(
    model: str, prompt: str, n: int = 1, debug: bool = False
) -> List[str]:
    """Generate code snippets via Ollama. Requires 'ollama' available in runtime.

    Returns a list of code strings (each must define def phi(state): ...).

    Note: This function is not executed in typical offline environments.
    """
    try:
        import ollama  # type: ignore
    except Exception as e:
        if debug:
            print("[HeuristicFinder] Ollama Python package not found or failed to import:", e, flush=True)
        return []

    out: List[str] = []
    for i in range(n):
        try:
            resp = ollama.generate(model=model, prompt=prompt, stream=False)
            # Robustly extract the text field from various client return types
            raw = None
            if isinstance(resp, dict):
                raw = resp.get("response", None)
            if raw is None:
                raw = getattr(resp, "response", None)
            if raw is None:
                s = str(resp)
                try:
                    import re
                    m = re.search(r"response\s*=\s*(['\"])(.*?)\1", s, flags=re.DOTALL)
                    raw = m.group(2) if m else s
                except Exception:
                    raw = s
            cleaned = _strip_think_tags(raw)
            _maybe_dump("ollama_raw", cleaned)
            code = _extract_code_block(cleaned)
            code = _extract_phi_from_text(code if code else cleaned)
            _maybe_dump("ollama_code_parsed", code, suffix=".py")

            # Second-pass repair if not a valid function
            if not _looks_like_phi(code):
                try:
                    repair_prompt = _build_repair_instruction(cleaned)
                    r2 = ollama.generate(model=model, prompt=repair_prompt, stream=False)
                    r2_raw = None
                    if isinstance(r2, dict):
                        r2_raw = r2.get("response", None)
                    if r2_raw is None:
                        r2_raw = getattr(r2, "response", None)
                    if r2_raw is None:
                        r2_raw = str(r2)
                    r2_clean = _strip_think_tags(r2_raw)
                    _maybe_dump("ollama_repair_raw", r2_clean)
                    r2_code = _extract_code_block(r2_clean)
                    r2_code = _extract_phi_from_text(r2_code if r2_code else r2_clean)
                    if _looks_like_phi(r2_code):
                        code = r2_code
                        _maybe_dump("ollama_repair_code", code, suffix=".py")
                except Exception as _:
                    pass
            if debug:
                print("=" * 80, flush=True)
                print(code, flush=True)
                print("=" * 80, flush=True)
            out.append(code)
        except Exception as e:
            if debug:
                print(f"[HeuristicFinder] Ollama generate failed at sample {i}: {e}", flush=True)
            continue
    return out


def generate_candidates_via_deepseek(
    prompt: str,
    n: int = 1,
    model: Optional[str] = None,
    debug: bool = False,
    system_prompt: Optional[str] = None,
    expect_code: bool = True,
) -> List[str]:
    """Generate code snippets via DeepSeek API (OpenAI-compatible).

    Reads API key from env var `DEEPSEEK_API_KEY`. Optional overrides:
    - `DEEPSEEK_MODEL` (default: 'deepseek-chat')
    - `DEEPSEEK_API_BASE` (default: 'https://api.deepseek.com/v1')
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        if debug:
            print("[HeuristicFinder] DEEPSEEK_API_KEY not set; skipping DeepSeek calls.", flush=True)
        return []

    base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    model_name = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # System prompt (overridable): default enforces code-only
    sys_prompt = system_prompt or (
        "You are a code generator. Return ONLY Python code for a single"
        " function 'def phi(state):' using torch ops, broadcastable to [B,1]."
        " Do not include explanations. Wrap the code in a fenced block:```python ...```"
    )

    # Allow env overrides for generation parameters
    try:
        temperature = float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.0"))
    except Exception:
        temperature = 0.0
    try:
        max_tokens = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "1024"))
    except Exception:
        max_tokens = 32768

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    url = base_url.rstrip("/") + "/chat/completions"

    out: List[str] = []
    for i in range(n):
        try:
            # Prefer requests, fallback to urllib
            try:
                import requests  # type: ignore

                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                import urllib.request
                import urllib.error

                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=60) as r:
                    b = r.read()
                    data = json.loads(b.decode("utf-8"))

            # DeepSeek (OpenAI-style) response
            raw = None
            try:
                choices = data.get("choices", []) if isinstance(data, dict) else []
                if choices:
                    msg = choices[0].get("message", {})
                    raw = msg.get("content")
                    # Some DeepSeek variants include 'reasoning_content'
                    if not raw and "reasoning_content" in msg:
                        raw = msg.get("reasoning_content")
            except Exception:
                raw = None

            if raw is None:
                raw = str(data)

            cleaned = _strip_think_tags(raw)
            _maybe_dump("deepseek_stage1_raw" if not expect_code else "deepseek_raw", cleaned)
            if not expect_code:
                # Reasoner/spec stage: return raw cleaned text (e.g., JSON)
                if debug:
                    print("=" * 80, flush=True)
                    print(cleaned, flush=True)
                    print("=" * 80, flush=True)
                out.append(cleaned)
                continue
            else:
                code = _extract_code_block(cleaned)
                code = _extract_phi_from_text(code if code else cleaned)
                _maybe_dump("deepseek_code_parsed", code, suffix=".py")

                # Second-pass repair if not a valid function
                if not _looks_like_phi(code):
                    try:
                        repair_payload = {
                            "model": model_name,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a code generator. Return ONLY a single fenced Python code block"
                                        " defining 'def phi(state):' using torch ops; no extra text."
                                    ),
                                },
                                {"role": "user", "content": _build_repair_instruction(cleaned)},
                            ],
                            "temperature": 0.2,
                            "max_tokens": 1024,
                            "stream": False,
                        }
                        try:
                            import requests  # type: ignore

                            r2 = requests.post(url, headers=headers, json=repair_payload, timeout=60)
                            r2.raise_for_status()
                            d2 = r2.json()
                        except Exception:
                            import urllib.request
                            import urllib.error

                            req2 = urllib.request.Request(
                                url, data=json.dumps(repair_payload).encode("utf-8"), headers=headers, method="POST"
                            )
                            with urllib.request.urlopen(req2, timeout=60) as rr:
                                bb = rr.read()
                                d2 = json.loads(bb.decode("utf-8"))

                        r2_raw = None
                        try:
                            ch = d2.get("choices", []) if isinstance(d2, dict) else []
                            if ch:
                                msg2 = ch[0].get("message", {})
                                r2_raw = msg2.get("content")
                                if not r2_raw and "reasoning_content" in msg2:
                                    r2_raw = msg2.get("reasoning_content")
                        except Exception:
                            r2_raw = None
                        if r2_raw is None:
                            r2_raw = str(d2)

                        r2_clean = _strip_think_tags(r2_raw)
                        _maybe_dump("deepseek_repair_raw", r2_clean)
                        r2_code = _extract_code_block(r2_clean)
                        r2_code = _extract_phi_from_text(r2_code if r2_code else r2_clean)
                        if _looks_like_phi(r2_code):
                            code = r2_code
                            _maybe_dump("deepseek_repair_code", code, suffix=".py")
                    except Exception as _:
                        pass
                if debug:
                    print("=" * 80, flush=True)
                    print(code, flush=True)
                    print("=" * 80, flush=True)
                out.append(code)
        except Exception as e:
            if debug:
                print(f"[HeuristicFinder] DeepSeek call failed at sample {i}: {e}", flush=True)
            continue
    return out


def generate_candidates(
    prompt: str,
    n: int = 1,
    debug: bool = False,
    ollama_model: Optional[str] = None,
    api_model: Optional[str] = None,
) -> List[str]:
    """Unified LLM generation: prefer Ollama if a model is provided, otherwise DeepSeek API.

    - If `ollama_model` is not None, uses local Ollama.
    - Else, uses DeepSeek API with env `DEEPSEEK_API_KEY`.
    """
    if ollama_model:
        try:
            return generate_candidates_via_ollama(ollama_model, prompt, n=n, debug=debug)
        except Exception:
            if debug:
                print("[HeuristicFinder] Ollama path failed; falling back to DeepSeek API.", flush=True)
            # fall through to DeepSeek
    return generate_candidates_via_deepseek(prompt, n=n, model=api_model, debug=debug)


def _reasoner_spec_prompt(context: str) -> str:
    return (
        "You are an expert in PBRS for TSP. Based on the following context, produce ONLY a concise JSON specification"
        " of a potential function 'phi(state)' without any code or explanations. JSON schema:\n"
        "{\n"
        "  \"summary\": string,\n"
        "  \"terms\": [\n"
        "    {\"name\": string, \"weight\": number, \"formula\": string}\n"
        "  ],\n"
        "  \"constraints\": [string],\n"
        "  \"edge_cases\": [string],\n"
        "  \"final_formula\": string\n"
        "}\n"
        "Return strictly valid JSON.\n\n"
        "Context begins:\n" + context + "\nContext ends."
    )


def _coder_prompt_from_spec(env_name: str, spec_text: str) -> str:
    base = format_prompt(env_name)
    inst = (
        "\nImplement 'phi(state)' that follows this specification.\n"
        "Return ONLY one fenced code block.\n"
        "Specification (verbatim):\n" + spec_text + "\n"
    )
    return base + "\n" + inst


def two_stage_generate_candidates(
    prompt: str,
    n: int = 1,
    env_name: str = "tsp",
    debug: bool = False,
    reasoner_model: Optional[str] = None,
    coder_ollama_model: Optional[str] = None,
) -> List[str]:
    """Two-stage generation: DeepSeek reasoner for spec, then Ollama (Qwen) for code.

    - Stage 1 (spec): uses DeepSeek API with model from env `DEEPSEEK_MODEL` or `deepseek-reasoner`.
    - Stage 2 (code): uses local Ollama model (e.g., qwen3:32b). If Ollama not available, falls back to DeepSeek chat.
    """
    # Stage 1: reasoner spec
    spec_ctx = _reasoner_context(env_name)
    spec_msgs_prompt = _reasoner_spec_prompt(spec_ctx)
    # Force reasoner model default
    rm = reasoner_model or os.environ.get("DEEPSEEK_REASONER_MODEL", None) or os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")
    json_sys_prompt = (
        "You are a PBRS/TSP expert. Return ONLY valid JSON (no code, no explanations) that matches the requested schema."
    )
    specs = generate_candidates_via_deepseek(
        spec_msgs_prompt, n=n, model=rm, debug=debug, system_prompt=json_sys_prompt, expect_code=False
    )
    if not specs:
        # fallback: use single-stage DeepSeek with stronger system prompt
        if debug:
            print("[HeuristicFinder] Reasoner produced no specs; falling back to single-stage generation.", flush=True)
        return generate_candidates(prompt, n=n, debug=debug, ollama_model=coder_ollama_model)

    out: List[str] = []
    for i, spec in enumerate(specs):
        try:
            coder_prompt = _coder_prompt_from_spec(env_name, spec)
            coder_model = coder_ollama_model or os.environ.get("TWO_STAGE_CODER_MODEL", None)
            if coder_model:
                codes = generate_candidates_via_ollama(coder_model, coder_prompt, n=1, debug=debug)
                if not codes or not _looks_like_phi(codes[0]):
                    # fall back to DeepSeek chat coder
                    codes = generate_candidates_via_deepseek(coder_prompt, n=1, model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"), debug=debug)
            else:
                # no Ollama model provided; try DeepSeek chat directly
                codes = generate_candidates_via_deepseek(coder_prompt, n=1, model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"), debug=debug)
            if codes:
                out.append(codes[0])
        except Exception as e:
            if debug:
                print(f"[HeuristicFinder] two-stage failed at sample {i}: {e}", flush=True)
            continue
    return out


# --- EoH-style prompts/operators (Ollama-only) ---
def _phi_prompt_parts(env_name: str = "tsp") -> Dict[str, object]:
    """Emulate EoH prompt parts for our PBRS phi(state) function."""
    task = (
        "Design a potential function for potential-based reward shaping (PBRS) in "
        f"the {env_name.upper()} environment. Implement a Python function named 'phi' that"
        " takes a single input 'state' (TSPStateView) and returns a scalar per batch"
        " as a torch tensor broadcastable to [B,1]."
    )
    func_name = "phi"
    func_inputs = ["state"]
    func_outputs = ["value"]
    inout_inf = (
        "Input 'state' offers helper methods (batch-friendly):\n"
        "Progress: remaining_ratio() -> [B,1]; visited_ratio() -> [B,1]; step_ratio() -> [B,1]\n"
        "Geometry: graph_scale() -> [B,1]; current_loc() -> [B,2]; start_loc() -> [B,2];\n"
        "  nearest_unvisited_distance(normalize=True) -> [B,1]; k_nearest_unvisited(k=3) -> [B,k]; k_farthest_unvisited(k=3) -> [B,k];\n"
        "  centroid_unvisited() -> [B,2]; distance_to_centroid(normalize=True) -> [B,1]; distance_to_start(normalize=True) -> [B,1];\n"
        "  mean_unvisited_distance(normalize=True) -> [B,1]; max_unvisited_distance(normalize=True) -> [B,1]; std_unvisited_distance(normalize=True) -> [B,1]\n"
        "Raw N-dependent (reduce over them): action_mask() -> [B,N] (True=unvisited); visited_mask() -> [B,N];\n"
        "  current_node_index() -> [B]; first_node_index() -> [B]; distances_from_current(normalize=True) -> [B,N];\n"
        "  distance_matrix(normalize=True) -> [B,N,N] (diag=0)."
    )
    other_inf = (
        "Constraints: Use only torch ops; do not import; ensure node-count-invariant outputs by reducing over N-dependent tensors;"
        " normalize distances by graph_scale(); avoid Python loops; ensure outputs are finite and reasonably scaled;"
        " return ONLY the Python function without any extra text."
    )
    return {
        "task": task,
        "func_name": func_name,
        "func_inputs": func_inputs,
        "func_outputs": func_outputs,
        "inout_inf": inout_inf,
        "other_inf": other_inf,
    }


def _join_list_for_prompt(items: List[str]) -> str:
    if len(items) > 1:
        return ", ".join("'" + s + "'" for s in items)
    return "'" + items[0] + "'"


def _prompt_i1(env_name: str = "tsp") -> str:
    p = _phi_prompt_parts(env_name)
    return (
        p["task"]
        + "\nReturn ONLY a valid Python function named 'phi' that accepts 'state' and returns a tensor broadcastable to [B,1].\n"
        + p["inout_inf"]
        + " "
        + p["other_inf"]
    )


def _prompt_e1(parents: List[Dict[str, str]], env_name: str = "tsp") -> str:
    p = _phi_prompt_parts(env_name)
    prompt_indiv = ""
    for i, ind in enumerate(parents):
        alg = ind.get("algorithm", "(no description)")
        code = ind.get("code", "")
        prompt_indiv += f"No.{i+1} algorithm and the corresponding code are: \n{alg}\n{code}\n"
    return (
        p["task"]
        + "\nI have "
        + str(len(parents))
        + " existing algorithms with their codes as follows: \n"
        + prompt_indiv
        + "Please help me create a new algorithm that has a totally different form from the given ones. \n"
        + "First, describe your new algorithm and main steps in one sentence. "
        + "The description must be inside a brace. Next, implement it in Python as a function named "
        + p["func_name"]
        + ". This function should accept "
        + str(len(p["func_inputs"]))
        + " input(s): "
        + _join_list_for_prompt(p["func_inputs"])  # type: ignore
        + ". The function should return "
        + str(len(p["func_outputs"]))
        + " output(s): "
        + _join_list_for_prompt(p["func_outputs"])  # type: ignore
        + ". "
        + p["inout_inf"]
        + " "
        + p["other_inf"]
        + "\nDo not give additional explanations."
    )


def _prompt_e2(parents: List[Dict[str, str]], env_name: str = "tsp") -> str:
    p = _phi_prompt_parts(env_name)
    prompt_indiv = ""
    for i, ind in enumerate(parents):
        alg = ind.get("algorithm", "(no description)")
        code = ind.get("code", "")
        prompt_indiv += f"No.{i+1} algorithm and the corresponding code are: \n{alg}\n{code}\n"
    return (
        p["task"]
        + "\nI have "
        + str(len(parents))
        + " existing algorithms with their codes as follows: \n"
        + prompt_indiv
        + "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"
        + "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. "
        + "The description must be inside a brace. Thirdly, implement it in Python as a function named "
        + p["func_name"]
        + ". This function should accept "
        + str(len(p["func_inputs"]))
        + " input(s): "
        + _join_list_for_prompt(p["func_inputs"])  # type: ignore
        + ". The function should return "
        + str(len(p["func_outputs"]))
        + " output(s): "
        + _join_list_for_prompt(p["func_outputs"])  # type: ignore
        + ". "
        + p["inout_inf"]
        + " "
        + p["other_inf"]
        + "\nDo not give additional explanations."
    )


def _prompt_m1(parent: Dict[str, str], env_name: str = "tsp") -> str:
    p = _phi_prompt_parts(env_name)
    alg = parent.get("algorithm", "(no description)")
    code = parent.get("code", "")
    return (
        p["task"]
        + "\nI have one algorithm with its code as follows. \n"
        + "Algorithm description: "
        + alg
        + "\nCode:\n\n"
        + code
        + "\nPlease assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"
        + "First, describe your new algorithm and main steps in one sentence. "
        + "The description must be inside a brace. Next, implement it in Python as a function named "
        + p["func_name"]
        + ". This function should accept "
        + str(len(p["func_inputs"]))
        + " input(s): "
        + _join_list_for_prompt(p["func_inputs"])  # type: ignore
        + ". The function should return "
        + str(len(p["func_outputs"]))
        + " output(s): "
        + _join_list_for_prompt(p["func_outputs"])  # type: ignore
        + ". "
        + p["inout_inf"]
        + " "
        + p["other_inf"]
        + "\nDo not give additional explanations."
    )


def _prompt_m2(parent: Dict[str, str], env_name: str = "tsp") -> str:
    p = _phi_prompt_parts(env_name)
    alg = parent.get("algorithm", "(no description)")
    code = parent.get("code", "")
    return (
        p["task"]
        + "\nI have one algorithm with its code as follows. \n"
        + "Algorithm description: "
        + alg
        + "\nCode:\n\n"
        + code
        + "\nPlease identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"
        + "First, describe your new algorithm and main steps in one sentence. "
        + "The description must be inside a brace. Next, implement it in Python as a function named "
        + p["func_name"]
        + ". This function should accept "
        + str(len(p["func_inputs"]))
        + " input(s): "
        + _join_list_for_prompt(p["func_inputs"])  # type: ignore
        + ". The function should return "
        + str(len(p["func_outputs"]))
        + " output(s): "
        + _join_list_for_prompt(p["func_outputs"])  # type: ignore
        + ". "
        + p["inout_inf"]
        + " "
        + p["other_inf"]
        + "\nDo not give additional explanations."
    )


def _prompt_m3(parent: Dict[str, str], env_name: str = "tsp") -> str:
    p = _phi_prompt_parts(env_name)
    code = parent.get("code", "")
    return (
        "First, you need to identify the main components in the function below. "
        "Next, analyze whether any of these components can be overfit to the in-distribution instances. "
        "Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. "
        "Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. \n"
        + code
        + "\n"
        + p["inout_inf"]
        + "\nDo not give additional explanations."
    )


def eoh_llm_e1(model: Optional[str], parents: List[Dict[str, str]], n: int = 1, env_name: str = "tsp", debug: bool = False) -> List[str]:
    prompt = _prompt_e1(parents, env_name)
    # allow env to force debug
    debug = debug or os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    if os.environ.get("TWO_STAGE_CODEGEN", "").lower() in ("1", "true", "yes"):  # reasoner -> coder(Qwen)
        return two_stage_generate_candidates(prompt, n=n, env_name=env_name, debug=debug, coder_ollama_model=model)
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)


def eoh_llm_e2(model: Optional[str], parents: List[Dict[str, str]], n: int = 1, env_name: str = "tsp", debug: bool = False) -> List[str]:
    prompt = _prompt_e2(parents, env_name)
    debug = debug or os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    if os.environ.get("TWO_STAGE_CODEGEN", "").lower() in ("1", "true", "yes"):
        return two_stage_generate_candidates(prompt, n=n, env_name=env_name, debug=debug, coder_ollama_model=model)
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)


def eoh_llm_i1(model: Optional[str], n: int = 1, env_name: str = "tsp", debug: bool = False) -> List[str]:
    prompt = _prompt_i1(env_name)
    debug = debug or os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    if os.environ.get("TWO_STAGE_CODEGEN", "").lower() in ("1", "true", "yes"):
        return two_stage_generate_candidates(prompt, n=n, env_name=env_name, debug=debug, coder_ollama_model=model)
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)


def eoh_llm_m1(model: Optional[str], parent_code: str, n: int = 1, env_name: str = "tsp", debug: bool = False) -> List[str]:
    parent = {"algorithm": "(no description)", "code": parent_code}
    prompt = _prompt_m1(parent, env_name)
    debug = debug or os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    if os.environ.get("TWO_STAGE_CODEGEN", "").lower() in ("1", "true", "yes"):
        return two_stage_generate_candidates(prompt, n=n, env_name=env_name, debug=debug, coder_ollama_model=model)
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)


def eoh_llm_m2(model: Optional[str], parent_code: str, n: int = 1, env_name: str = "tsp", debug: bool = False) -> List[str]:
    parent = {"algorithm": "(no description)", "code": parent_code}
    prompt = _prompt_m2(parent, env_name)
    debug = debug or os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    if os.environ.get("TWO_STAGE_CODEGEN", "").lower() in ("1", "true", "yes"):
        return two_stage_generate_candidates(prompt, n=n, env_name=env_name, debug=debug, coder_ollama_model=model)
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)


def eoh_llm_m3(model: Optional[str], parent_code: str, n: int = 1, env_name: str = "tsp", debug: bool = False) -> List[str]:
    parent = {"code": parent_code}
    prompt = _prompt_m3(parent, env_name)
    if os.environ.get("TWO_STAGE_CODEGEN", "").lower() in ("1", "true", "yes"):
        return two_stage_generate_candidates(prompt, n=n, env_name=env_name, debug=debug, coder_ollama_model=model)
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)
def build_eoh_prompt(
    operator: str,
    env_name: str,
    parent_a: str,
    parent_b: Optional[str] = None,
    guidance: str = "",
    eval_summary: str = "",
) -> str:
    """Construct an instruction for an LLM to mutate or crossover `phi` code.

    The LLM must output ONLY a valid Python function: `def phi(state): ...`.
    """
    header = format_prompt(env_name, guidance)
    if eval_summary:
        header += f"\nPerformance summary (for reference):\n{eval_summary}\n"

    if operator.lower() == "mutate":
        instr = (
            "Operator: MUTATE.\n"
            "Make small but meaningful changes to improve stability and validation reward.\n"
            "Preserve shapes and broadcasting; avoid dependence on exact N.\n"
            "Parent A (current):\n" + parent_a + "\n"
            "Return ONLY the new function in a fenced code block (```python ... ```), no extra text."
        )
    elif operator.lower() == "crossover" and parent_b is not None:
        instr = (
            "Operator: CROSSOVER.\n"
            "Merge good ideas from two parents into a single, concise function.\n"
            "Avoid duplicated work and keep constants well-scaled (use graph_scale).\n"
            "Parent A:\n" + parent_a + "\n\nParent B:\n" + parent_b + "\n"
            "Return ONLY the new function in a fenced code block (```python ... ```), no extra text."
        )
    else:
        instr = (
            "Rewrite to improve clarity and robustness without changing I/O.\n"
            + parent_a
            + "\nReturn ONLY the new function in a fenced code block (```python ... ```), no extra text."
        )

    return header + "\n" + instr


def eoh_llm_mutate(
    model: str,
    parent_code: str,
    env_name: str = "tsp",
    guidance: str = "",
    eval_summary: str = "",
    n: int = 1,
    debug: bool = False,
) -> List[str]:
    prompt = build_eoh_prompt(
        operator="mutate",
        env_name=env_name,
        parent_a=parent_code,
        guidance=guidance,
        eval_summary=eval_summary,
    )
    return generate_candidates_via_ollama(model, prompt, n=n, debug=debug)


def eoh_llm_crossover(
    model: str,
    parent_a: str,
    parent_b: str,
    env_name: str = "tsp",
    guidance: str = "",
    eval_summary: str = "",
    n: int = 1,
    debug: bool = False,
) -> List[str]:
    prompt = build_eoh_prompt(
        operator="crossover",
        env_name=env_name,
        parent_a=parent_a,
        parent_b=parent_b,
        guidance=guidance,
        eval_summary=eval_summary,
    )
    return generate_candidates_via_ollama(model, prompt, n=n, debug=debug)
