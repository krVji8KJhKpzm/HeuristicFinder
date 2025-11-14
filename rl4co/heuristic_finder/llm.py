from __future__ import annotations

from typing import List, Optional, Dict
import os
import json
import time


def _available_helpers_text() -> str:
    """Canonical list of state helpers exposed to the model.

    Keep names/signatures aligned with InvariantTSPStateView in
    rl4co/envs/routing/tsp/pbrs_env.py.
    """
    return (
        "Available state helpers (batch-friendly):\n"
        "Raw N-dependent ONLY (reduce to keep invariance): action_mask() -> [B,N] (True=unvisited); "
        "visited_mask() -> [B,N]; unvisited_mask() -> [B,N];\n"
        "  current_node_index() -> [B]; first_node_index() -> [B]; distance_matrix() -> [B,N,N] (diag=0).\n"
        "  all_node_coords() -> [B,N,2]; partial_path_indices() -> [B,N] (visit order, -1 padded where missing).\n"
        "Fixed-size (safe direct use): current_loc() -> [B,2]; start_loc() -> [B,2].\n"
    )


def format_prompt(env_name: str = "tsp", guidance: str = "") -> str:
    return (
        "You are designing a potential function Phi(state) for PBRS in "
        f"combinatorial optimization env '{env_name}'.\n"
        "Output format (strict):\n"
        "- Return ONLY a single fenced code block starting with: ```python and ending with: ```\n"
        "- Inside the code block, the FIRST line must be: # THOUGHT: {one-sentence idea}.\n"
        "- Then define exactly one function: def phi(state): and nothing else.\n"
        "- Use only torch ops; no prints, no explanations, no comments outside code.\n"
        "- Ensure result is broadcastable to [B,1]; handle NaNs via torch.nan_to_num.\n"
        "Goal: robust, node-count-invariant outputs. Use reductions (mean/max/min/std/softmin) over N-dependent tensors.\n"
        "Avoid Python loops; prefer vectorized torch ops.\n"
        + _available_helpers_text()
        + "Do NOT use any other state.* helpers (e.g., visited_ratio, remaining_ratio, nearest/centroid/start distances, graph_scale, distances_from_current).\n"
        + "Return a tensor broadcastable to [B,1]. Keep it simple and stable.\n"
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
    """Extract the phi function and preserve an optional preceding THOUGHT comment line.

    Strategy:
    - Find the first occurrence of `def phi(`. If found, slice from there to the end of the code block (or text end).
    - If a line like `# THOUGHT: { ... }` appears immediately before the function, preserve it above the function.
    - If not found, fallback to a coarse 'def ... return' capture.
    """
    try:
        import re
        m = re.search(r"def\s+phi\s*\(.*?\):", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            start = m.start()
            tail = text[start:]
            # cut at next fenced block end if present
            fence = tail.find("```")
            if fence != -1:
                tail = tail[:fence]
            # look back for THOUGHT or algorithm line before 'def phi'
            prefix = text[:start]
            # 1) exact comment form with braces
            tm = re.search(r"(?mi)^\s*#\s*THOUGHT:\s*\{([^}]*)\}\s*$", prefix)
            if tm:
                t = tm.group(1).strip()
                thought_line = f"# THOUGHT: {{{t}}}"
                return (thought_line + "\n" + tail.strip()).strip()
            # 2) plain THOUGHT: ... (with or without braces)
            tm2 = re.search(r"(?mi)^\s*THOUGHT:\s*(\{?)([^\n\r}]*)\}?\s*$", prefix)
            if tm2:
                t = tm2.group(2).strip()
                thought_line = f"# THOUGHT: {{{t}}}"
                return (thought_line + "\n" + tail.strip()).strip()
            # 3) fallback: last {...} block in prefix as algorithm
            br = re.findall(r"\{([^}]*)\}", prefix)
            if br:
                t = br[-1].strip()
                # avoid extremely long captures (likely JSON/specs)
                if 0 < len(t) <= 400:
                    thought_line = f"# THOUGHT: {{{t}}}"
                    return (thought_line + "\n" + tail.strip()).strip()
            return tail.strip()
        # Coarse fallback: from first 'def' to last 'return'
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


def _ensure_thought_line(code: Optional[str], default: str = "auto") -> str:
    """Ensure the first non-empty line is a THOUGHT comment.

    If missing, prepend "# THOUGHT: {default}" followed by a newline.
    """
    if not isinstance(code, str):
        return ""
    lines = code.splitlines()
    # find first non-empty line index
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx < len(lines):
        first = lines[idx].lstrip()
        if first.lower().startswith("# thought:"):
            return code
    prefix = f"# THOUGHT: {{{default}}}\n"
    return prefix + code.lstrip("\n")


def _build_repair_instruction(raw: str) -> str:
    return (
        "Convert the following content into ONLY a Python fenced code block implementing a single function '\n"
        "def phi(state):\n' using torch ops. Requirements:\n"
        "- Return ONLY one fenced block: ```python ... ``` with no extra text.\n"
        "- Use provided state helpers conceptually referenced.\n"
        "- Output must be broadcastable to [B,1]; handle NaNs via torch.nan_to_num.\n"
        "- Keep it stable and node-count-invariant.\n"
        "Content begins:\n" + raw + "\nContent ends."
    )


def _reasoner_context(env_name: str = "tsp") -> str:
    """Build a neutral, spec-only context for the reasoner stage without code-only directives."""
    parts = _phi_prompt_parts(env_name)
    goal = (
        "Design a potential function Phi(state) for PBRS in the given environment.\n"
        "Constraints: node-count-invariant, output broadcastable to [B,1], handle NaNs with torch.nan_to_num,\n"
        "and prefer simple/stable formulations. Use ONLY the listed raw helpers; do not invent extra methods.\n"
    )
    ctx = goal + parts["inout_inf"]
    return ctx


def generate_candidates_via_ollama(
    model: str, prompt: str, n: int = 1, debug: bool = False, stream: bool = False
) -> List[str]:
    """Generate code snippets via Ollama. Requires 'ollama' available in runtime.

    Returns a list of code strings (each must define def phi(state): ...).

    Note: This function is not executed in typical offline environments.
    """
    try:
        import ollama  # type: ignore
    except Exception as e:
        # if debug:
        print("[HeuristicFinder] Ollama Python package not found or failed to import:", e, flush=True)
        return []

    out: List[str] = []
    for i in range(n):
        try:
            resp = ollama.generate(model=model, prompt=prompt, stream=stream)
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
            # Extract code while preserving any preceding THOUGHT/algorithm from raw content
            code = _extract_phi_from_text(cleaned)
            code = _ensure_thought_line(code)
            _maybe_dump("ollama_code_parsed", code, suffix=".py")

            # Second-pass repair if not a valid function
            if not _looks_like_phi(code):
                try:
                    repair_prompt = _build_repair_instruction(cleaned)
                    r2 = ollama.generate(model=model, prompt=repair_prompt, stream=stream)
                    r2_raw = None
                    if isinstance(r2, dict):
                        r2_raw = r2.get("response", None)
                    if r2_raw is None:
                        r2_raw = getattr(r2, "response", None)
                    if r2_raw is None:
                        r2_raw = str(r2)
                    r2_clean = _strip_think_tags(r2_raw)
                    _maybe_dump("ollama_repair_raw", r2_clean)
                    r2_code = _extract_phi_from_text(r2_clean)
                    r2_code = _ensure_thought_line(r2_code)
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
            # if debug:
            print(f"[HeuristicFinder] Ollama generate failed at sample {i}: {e}", flush=True)
            continue
    return out


_DEFAULT_PHI_SYSTEM_PROMPT = (
    "You are a code generator. Return ONLY one fenced Python code block."
    " The FIRST line inside must be a single comment: '# THOUGHT: {one-sentence idea}'."
    " Then define exactly one function 'def phi(state):' using torch ops, broadcastable to [B,1]."
    " Do not include explanations outside the code block."
)


def _generate_candidates_via_openai_compatible_api(
    *,
    prompt: str,
    n: int,
    model: Optional[str],
    debug: bool,
    system_prompt: Optional[str],
    expect_code: bool,
    env_prefix: str,
    default_base: str,
    default_model: str,
    default_system_prompt: Optional[str],
    stream: bool = False,
) -> List[str]:
    prefix_upper = env_prefix.upper()
    api_key = os.environ.get(f"{prefix_upper}_API_KEY")
    if not api_key:
        if debug:
            print(
                f"[HeuristicFinder] {prefix_upper}_API_KEY not set; skipping {env_prefix} calls.",
                flush=True,
            )
        return []

    base_url = os.environ.get(f"{prefix_upper}_API_BASE", default_base) or default_base
    model_name = model or os.environ.get(f"{prefix_upper}_MODEL", default_model)

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "DMXAPI/1.0.0"
    }

    effective_system_prompt = system_prompt if system_prompt is not None else default_system_prompt

    try:
        temperature = float(os.environ.get(f"{prefix_upper}_TEMPERATURE", "0.0"))
    except Exception:
        temperature = 0.0
    try:
        max_tokens = int(os.environ.get(f"{prefix_upper}_MAX_TOKENS", "1024"))
    except Exception:
        max_tokens = 32768

    messages: List[Dict[str, str]] = []
    if effective_system_prompt:
        messages.append({"role": "system", "content": effective_system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    url = base_url.rstrip("/") + "/chat/completions"
    dump_tag = env_prefix.lower()

    out: List[str] = []
    for i in range(n):
        print("Requesting to LLM...")
        try:
            start_time = time.time()
            try:
                import requests  # type: ignore

                if stream:
                    headers = {
                        **headers,
                        "Accept": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }

                timeout = (10, 300)

                with requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                    stream=stream,
                ) as resp:
                    resp.raise_for_status()

                    if stream:
                        event_buf = []
                        raw_acc = ""
                        for raw_line in resp.iter_lines(decode_unicode=True):
                            if raw_line is None:
                                continue
                            line = raw_line.strip()
                            if line == "":
                                if not event_buf:
                                    continue
                                data_lines = [l[5:].lstrip() for l in event_buf if l.startswith("data:")]
                                event_buf.clear()
                                if not data_lines:
                                    continue
                                data_str = "\n".join(data_lines)
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue

                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {}) or {}
                                    piece = delta.get("content") or ""
                                    piece_rc = delta.get("reasoning_content") or ""
                                    if piece_rc:
                                        piece = piece_rc + piece

                                    if piece:
                                        raw_acc += piece
                                        print(piece, end="", flush=True)
                                continue
                            if line.startswith(":"):
                                continue

                            event_buf.append(line)

                        data = {"choices": [{"message": {"content": raw_acc}}]}
                    else:
                        data = resp.json()

                print(f"Request cost time: {time.time() - start_time} (s)", flush=True)
            except Exception as e:
                print(f"Request failed, error:{e}", flush=True)
                import urllib.request
                import urllib.error

                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10) as r:
                    b = r.read()
                    data = json.loads(b.decode("utf-8"))

            raw = None
            try:
                choices = data.get("choices", []) if isinstance(data, dict) else []
                if choices:
                    msg = choices[0].get("message", {})
                    raw = msg.get("content")
                    if not raw and "reasoning_content" in msg:
                        raw = msg.get("reasoning_content")
            except Exception:
                raw = None

            if raw is None:
                raw = str(data)
                print(f"LLM response: {raw}")

            cleaned = _strip_think_tags(raw)
            _maybe_dump(f"{dump_tag}_stage1_raw" if not expect_code else f"{dump_tag}_raw", cleaned)

            if not expect_code:
                if debug:
                    print("=" * 80, flush=True)
                    print(cleaned, flush=True)
                    print("=" * 80, flush=True)
                out.append(cleaned)
                continue

            code = _extract_phi_from_text(cleaned)
            code = _ensure_thought_line(code)
            _maybe_dump(f"{dump_tag}_code_parsed", code, suffix=".py")

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
                        "temperature": 0.0,
                        "max_tokens": 32768,
                        "stream": stream,
                    }
                    try:
                        import requests  # type: ignore

                        # r2 = requests.post(url, headers=headers, json=repair_payload, timeout=60)
                        # r2.raise_for_status()
                        
                        # if stream:
                        #     # Handle streaming response for repair
                        #     r2_raw = ""
                        #     for line in r2.iter_lines():
                        #         if line:
                        #             line_str = line.decode('utf-8')
                        #             if line_str.startswith('data: '):
                        #                 data_str = line_str[6:]
                        #                 if data_str == '[DONE]':
                        #                     break
                        #                 try:
                        #                     chunk_data = json.loads(data_str)
                        #                     choices = chunk_data.get("choices", [])
                        #                     if choices:
                        #                         delta = choices[0].get("delta", {})
                        #                         content = delta.get("content", "")
                        #                         if content:
                        #                             r2_raw += content
                        #                 except json.JSONDecodeError:
                        #                     continue
                        #     d2 = {"choices": [{"message": {"content": r2_raw}}]}
                        # else:
                        #     d2 = r2.json()
                        if stream:
                            headers = {
                                **headers,
                                "Accept": "text/event-stream",
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                            }

                        with requests.post(
                            url,
                            headers=headers,
                            json=payload,
                            timeout=timeout,
                            stream=stream,
                        ) as r2:
                            r2.raise_for_status()

                            if stream:
                                event_buf = []
                                raw_acc = ""
                                for raw_line in r2.iter_lines(decode_unicode=True):
                                    if raw_line is None:
                                        continue
                                    line = raw_line.strip()
                                    if line == "":
                                        if not event_buf:
                                            continue
                                        data_lines = [l[5:].lstrip() for l in event_buf if l.startswith("data:")]
                                        event_buf.clear()
                                        if not data_lines:
                                            continue
                                        data_str = "\n".join(data_lines)
                                        if data_str.strip() == "[DONE]":
                                            break
                                        try:
                                            chunk = json.loads(data_str)
                                        except json.JSONDecodeError:
                                            continue

                                        choices = chunk.get("choices", [])
                                        if choices:
                                            delta = choices[0].get("delta", {}) or {}
                                            piece = delta.get("content") or ""
                                            piece_rc = delta.get("reasoning_content") or ""
                                            if piece_rc:
                                                piece = piece_rc + piece

                                            if piece:
                                                raw_acc += piece
                                                print(piece, end="", flush=True)
                                        continue
                                    if line.startswith(":"):
                                        continue

                                    event_buf.append(line)

                                d2 = {"choices": [{"message": {"content": raw_acc}}]}
                            else:
                                d2 = resp.json()
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
                    _maybe_dump(f"{dump_tag}_repair_raw", r2_clean)
                    r2_code = _extract_phi_from_text(r2_clean)
                    r2_code = _ensure_thought_line(r2_code)
                    if _looks_like_phi(r2_code):
                        code = r2_code
                        _maybe_dump(f"{dump_tag}_repair_code", code, suffix=".py")
                except Exception:
                    pass

            if debug:
                print("=" * 80, flush=True)
                print(code, flush=True)
                print("=" * 80, flush=True)
            out.append(code)
        except Exception as exc:
            # if debug:
            print(f"[HeuristicFinder] {env_prefix} call failed at sample {i}: {exc}", flush=True)
            continue
    return out


def generate_candidates_via_deepseek(
    prompt: str,
    n: int = 1,
    model: Optional[str] = None,
    debug: bool = False,
    system_prompt: Optional[str] = None,
    expect_code: bool = True,
    stream: bool = False,
) -> List[str]:
    """Generate code snippets via DeepSeek API (OpenAI-compatible)."""
    return _generate_candidates_via_openai_compatible_api(
        prompt=prompt,
        n=n,
        model=model,
        debug=debug,
        system_prompt=system_prompt,
        expect_code=expect_code,
        env_prefix="deepseek",
        default_base="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        default_system_prompt=_DEFAULT_PHI_SYSTEM_PROMPT,
        stream=stream,
    )


def generate_candidates_via_kimi(
    prompt: str,
    n: int = 1,
    model: Optional[str] = None,
    debug: bool = False,
    system_prompt: Optional[str] = None,
    expect_code: bool = True,
    stream: bool = False,
) -> List[str]:
    """Generate code snippets via Kimi's OpenAI-compatible API (e.g., k2-thinking)."""
    return _generate_candidates_via_openai_compatible_api(
        prompt=prompt,
        n=n,
        model=model,
        debug=debug,
        system_prompt=system_prompt,
        expect_code=expect_code,
        env_prefix="kimi",
        default_base="https://api.moonshot.cn/v1",
        default_model="kimi-k2-turbo-preview",
        default_system_prompt=_DEFAULT_PHI_SYSTEM_PROMPT,
        stream=stream,
    )


def generate_candidates_via_glm(
    prompt: str,
    n: int = 1,
    model: Optional[str] = None,
    debug: bool = False,
    system_prompt: Optional[str] = None,
    expect_code: bool = True,
    stream: bool = False,
) -> List[str]:
    """Generate code snippets via GLM (General Language Model) API (OpenAI-compatible)."""
    return _generate_candidates_via_openai_compatible_api(
        prompt=prompt,
        n=n,
        model=model,
        debug=debug,
        system_prompt=system_prompt,
        expect_code=expect_code,
        env_prefix="glm",
        default_base="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4",
        default_system_prompt=_DEFAULT_PHI_SYSTEM_PROMPT,
        stream=stream,
    )


def _resolve_llm_api_provider(preferred: Optional[str] = None) -> str:
    candidates = {"deepseek", "kimi", "glm"}
    if preferred:
        candidate = preferred.strip().lower()
        if candidate in candidates:
            return candidate
    env_provider = os.environ.get("LLM_API_PROVIDER")
    if env_provider:
        candidate = env_provider.strip().lower()
        if candidate in candidates:
            return candidate
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("KIMI_API_KEY"):
        return "kimi"
    if os.environ.get("GLM_API_KEY"):
        return "glm"
    return "deepseek"


def _default_reasoner_model(provider: str) -> str:
    if provider == "kimi":
        return os.environ.get("KIMI_REASONER_MODEL") or os.environ.get("KIMI_MODEL") or "kimi-k2-thinking"
    if provider == "glm":
        return os.environ.get("GLM_REASONER_MODEL") or os.environ.get("GLM_MODEL") or "glm-4"
    return os.environ.get("DEEPSEEK_REASONER_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-reasoner"


def _default_coder_model(provider: str) -> str:
    if provider == "kimi":
        return os.environ.get("KIMI_MODEL") or "kimi-k2-thinking"
    if provider == "glm":
        return os.environ.get("GLM_MODEL") or "glm-4"
    return os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"


def _generate_candidates_for_provider(
    provider: str,
    prompt: str,
    n: int = 1,
    model: Optional[str] = None,
    debug: bool = False,
    system_prompt: Optional[str] = None,
    expect_code: bool = True,
    stream: bool = False,
) -> List[str]:
    if provider == "kimi":
        return generate_candidates_via_kimi(
            prompt=prompt,
            n=n,
            model=model,
            debug=debug,
            system_prompt=system_prompt,
            expect_code=expect_code,
            stream=stream,
        )
    if provider == "glm":
        return generate_candidates_via_glm(
            prompt=prompt,
            n=n,
            model=model,
            debug=debug,
            system_prompt=system_prompt,
            expect_code=expect_code,
            stream=stream,
        )
    return generate_candidates_via_deepseek(
        prompt=prompt,
        n=n,
        model=model,
        debug=debug,
        system_prompt=system_prompt,
        expect_code=expect_code,
        stream=stream,
    )


def generate_candidates(
    prompt: str,
    n: int = 1,
    debug: bool = False,
    ollama_model: Optional[str] = None,
    api_model: Optional[str] = None,
) -> List[str]:
    """Unified LLM generation: prefer Ollama if available, otherwise DeepSeek, Kimi, or GLM API.

    - If `ollama_model` is provided, uses local Ollama.
    - Else, uses the OpenAI-compatible provider selected by `LLM_API_PROVIDER` (DeepSeek, Kimi, or GLM).
    """
    if ollama_model:
        try:
            return generate_candidates_via_ollama(ollama_model, prompt, n=n, debug=debug)
        except Exception:
            # if debug:
            print("[HeuristicFinder] Ollama path failed; falling back to remote provider.", flush=True)
            # fall through to DeepSeek
    provider = _resolve_llm_api_provider()
    return _generate_candidates_for_provider(provider, prompt, n=n, model=api_model, debug=debug)


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
    api_provider: Optional[str] = None,
) -> List[str]:
    """Two-stage generation: provider reasoner/spec followed by Ollama (Qwen) or provider chat coder.

    - Stage 1 (spec): uses the selected provider (DeepSeek or Kimi) reasoner model.
    - Stage 2 (code): uses local Ollama (if available) otherwise the provider chat model.
    """
    # Stage 1: reasoner spec
    spec_ctx = _reasoner_context(env_name)
    spec_msgs_prompt = _reasoner_spec_prompt(spec_ctx)
    provider = _resolve_llm_api_provider(api_provider)
    rm = reasoner_model or _default_reasoner_model(provider)
    json_sys_prompt = (
        "You are a PBRS/TSP expert. Return ONLY valid JSON (no code, no explanations) that matches the requested schema."
    )
    specs = _generate_candidates_for_provider(
        provider,
        spec_msgs_prompt,
        n=n,
        model=rm,
        debug=debug,
        system_prompt=json_sys_prompt,
        expect_code=False,
    )
    if not specs:
        # fallback: use single-stage provider with a stronger system prompt
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
                    fallback_model = _default_coder_model(provider)
                    codes = _generate_candidates_for_provider(
                        provider,
                        coder_prompt,
                        n=1,
                        model=fallback_model,
                        debug=debug,
                    )
            else:
                fallback_model = _default_coder_model(provider)
                codes = _generate_candidates_for_provider(
                    provider,
                    coder_prompt,
                    n=1,
                    model=fallback_model,
                    debug=debug,
                )
            if codes:
                out.append(codes[0])
        except Exception as e:
            # if debug:
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
        "Input 'state' offers helper methods (batch-friendly):\n" + _available_helpers_text()
    )
    # "Forbidden: visited_ratio, remaining_ratio, step_ratio, current_loc, start_loc, nearest/centroid/start distances, k-nearest/farthest, graph_scale, distances_from_current.\n"
    other_inf = (
        "Constraints: Use only torch ops; do not import; ensure node-count-invariant outputs by reducing over N-dependent tensors;"
        " avoid Python loops; ensure outputs are finite and reasonably scaled."
        " Return ONLY one fenced Python code block: the FIRST line is '# THOUGHT: {one-sentence idea}',"
        " followed by exactly one function 'def phi(state):'. No extra text outside the code block."
        "\nQuality constraints (aim to satisfy):\n"
        "- Shaping strength should be moderate: target step_shaping_ratio and episode_shaping_ratio in [0.01, 0.20].\n"
        "- Terminal consistency: for complete trajectories, make gamma^T·Phi(s_T) - Phi(s_0) close to 0; keep Phi equal across terminal goal states.\n"
        "- No loop arbitrage: ensure the sum of gamma·ΔPhi over any loop is bounded and near 0 (no reward farming).\n"
        "- Positive progress correlation: Phi or ΣΔPhi should correlate positively with final reward/progress/success probability.\n"
        "- Smoothness: keep |ΔPhi| changes reasonable (e.g., controlled 95th percentile); avoid large jumps between adjacent states.\n"
        "- Permutation invariance over node indices for combinatorial instances (use reductions like mean/max/min/std/softmin).\n"
        "- Prefer not to increase reward variance: shaped reward variance should not drastically exceed base reward variance.\n"
        "If a 'Diagnostics summary' is provided for parents, use it to adjust the design towards these targets (increase correlation, keep ratios within range, reduce terminal Phi variance, and bound |ΔPhi|)."
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
        + "Before the function, add a single Python comment line starting with '# THOUGHT: {your one-sentence idea here}' describing the main idea in braces.\n"
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
        diag = ind.get("diagnostics", None)
        prompt_indiv += f"No.{i+1} algorithm and the corresponding code are: \n{alg}\n{code}\n"
        if diag:
            prompt_indiv += f"Diagnostics summary: \n{diag}\n"
    return (
        p["task"]
        + "\nI have "
        + str(len(parents))
        + " existing algorithms with their codes as follows: \n"
        + prompt_indiv
        + "Please help me create a new algorithm that has a totally different form from the given ones. \n"
        + "First, describe your new algorithm and main steps in one sentence, and put it inside a single brace.\n"
        + "Place this sentence as a Python comment on the first line: '# THOUGHT: { ... }'.\n"
        + "Next, implement it in Python as a function named "
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
        diag = ind.get("diagnostics", None)
        prompt_indiv += f"No.{i+1} algorithm and the corresponding code are: \n{alg}\n{code}\n"
        if diag:
            prompt_indiv += f"Diagnostics summary: \n{diag}\n"
    return (
        p["task"]
        + "\nI have "
        + str(len(parents))
        + " existing algorithms with their codes as follows: \n"
        + prompt_indiv
        + "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"
        + "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence, inside a single brace.\n"
        + "Place this sentence as a Python comment on the first line: '# THOUGHT: { ... }'.\n"
        + "Thirdly, implement it in Python as a function named "
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
    diag = parent.get("diagnostics", None)
    return (
        p["task"]
        + "\nI have one algorithm with its code as follows. \n"
        + "Algorithm description: "
        + alg
        + "\nCode:\n\n"
        + code
        + ("\nDiagnostics summary:\n" + diag + "\n" if diag else "\n")
        + "Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"
        + "First, describe your new algorithm and main steps in one sentence, inside a single brace, and place it as the first line Python comment '# THOUGHT: { ... }'.\n"
        + "Next, implement it in Python as a function named "
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
    diag = parent.get("diagnostics", None)
    return (
        p["task"]
        + "\nI have one algorithm with its code as follows. \n"
        + "Algorithm description: "
        + alg
        + "\nCode:\n\n"
        + code
        + ("\nDiagnostics summary:\n" + diag + "\n" if diag else "\n")
        + "Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"
        + "First, describe your new algorithm and main steps in one sentence, inside a single brace, and place it as the first line Python comment '# THOUGHT: { ... }'.\n"
        + "Next, implement it in Python as a function named "
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
    diag = parent.get("diagnostics", None)
    return (
        "First, you need to identify the main components in the function below. "
        "Next, analyze whether any of these components can be overfit to the in-distribution instances. "
        "Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. "
        "Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. On the first line, add a Python comment '# THOUGHT: {one-sentence change rationale}'. \n"
        + code
        + ("\nDiagnostics summary:\n" + diag + "\n" if diag else "\n")
        + p["inout_inf"]
        + "\nDo not give additional explanations."
    )


def eoh_llm_repair(model: Optional[str], parent_code: str, env_name: str = "tsp", n: int = 1, debug: bool = False, stream: bool = False) -> List[str]:
    """Memetic-style light repair/simplify pass that preserves I/O and improves stability.

    The model should return ONLY a fenced Python code block for `def phi(state): ...`.
    The first line should be a comment like '# THOUGHT: {short rationale}'.
    """
    guidance = (
        "Lightly revise the function to: enforce broadcasting to [B,1], replace NaNs via torch.nan_to_num,"
        " prefer reductions over N-dependent tensors, avoid hard-coded N, and clamp magnitudes."
    )
    p = _phi_prompt_parts(env_name)
    prompt = (
        p["task"]
        + "\nHere is the current function to analyze and revise:\n\n"
        + parent_code
        + "\n\n"
        + guidance
        + "\nAdd a first line comment '# THOUGHT: {one-sentence change rationale}'. Return ONLY the code."
    )
    debug = debug or os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)


def eoh_llm_e1(model: Optional[str], parents: List[Dict[str, str]], n: int = 1, env_name: str = "tsp", debug: bool = False, stream: bool = False) -> List[str]:
    prompt = _prompt_e1(parents, env_name)
    # allow env to force debug
    debug = debug or os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    if os.environ.get("TWO_STAGE_CODEGEN", "").lower() in ("1", "true", "yes"):  # reasoner -> coder(Qwen)
        return two_stage_generate_candidates(prompt, n=n, env_name=env_name, debug=debug, coder_ollama_model=model)
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)


def eoh_llm_e2(model: Optional[str], parents: List[Dict[str, str]], n: int = 1, env_name: str = "tsp", debug: bool = False, stream: bool = False) -> List[str]:
    prompt = _prompt_e2(parents, env_name)
    debug = debug or os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    if os.environ.get("TWO_STAGE_CODEGEN", "").lower() in ("1", "true", "yes"):
        return two_stage_generate_candidates(prompt, n=n, env_name=env_name, debug=debug, coder_ollama_model=model)
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)


def eoh_llm_i1(model: Optional[str], n: int = 1, env_name: str = "tsp", debug: bool = False, stream: bool = False) -> List[str]:
    prompt = _prompt_i1(env_name)
    debug = debug or os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    if os.environ.get("TWO_STAGE_CODEGEN", "").lower() in ("1", "true", "yes"):
        return two_stage_generate_candidates(prompt, n=n, env_name=env_name, debug=debug, coder_ollama_model=model)
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)


def eoh_llm_m1(model: Optional[str], parent_code: str, n: int = 1, env_name: str = "tsp", debug: bool = False, stream: bool = False) -> List[str]:
    parent = {"algorithm": "(no description)", "code": parent_code}
    prompt = _prompt_m1(parent, env_name)
    debug = debug or os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    if os.environ.get("TWO_STAGE_CODEGEN", "").lower() in ("1", "true", "yes"):
        return two_stage_generate_candidates(prompt, n=n, env_name=env_name, debug=debug, coder_ollama_model=model)
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)


def eoh_llm_m2(model: Optional[str], parent_code: str, n: int = 1, env_name: str = "tsp", debug: bool = False, stream: bool = False) -> List[str]:
    parent = {"algorithm": "(no description)", "code": parent_code}
    prompt = _prompt_m2(parent, env_name)
    debug = debug or os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    if os.environ.get("TWO_STAGE_CODEGEN", "").lower() in ("1", "true", "yes"):
        return two_stage_generate_candidates(prompt, n=n, env_name=env_name, debug=debug, coder_ollama_model=model)
    return generate_candidates(prompt, n=n, debug=debug, ollama_model=model)


def eoh_llm_m3(model: Optional[str], parent_code: str, n: int = 1, env_name: str = "tsp", debug: bool = False, stream: bool = False) -> List[str]:
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
            "Avoid duplicated work and keep constants well-scaled.\n"
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
