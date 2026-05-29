# E2_2_capiso_scale.py -- Capability-isolation: scale variant of Stage 0' (the-search#3).
#
# One variable changed vs Stage 0': core capability/scale.
#   Stage 0': gemma-4-E2B-it (2B, 384 thinking tokens) -> KILL (0/9 solves, 83% parse)
#   This:     gemma-4-26B-A4B-it (26B, 2048 thinking tokens) -> ???
#
# Substrate HELD FIXED: code-synthesis generate-and-test, N=5, error feedback.
# Tasks: same rng_sample seed (42) -> same 30-task pool; extend toward 30 since TIME_CAP raised.
#
# Pre-registered kill: CORE_ONLY > 0 -> capability was the wall -> Stage 1 (meta-layer).
# Pre-registered forward: CORE_ONLY = 0 -> structural wall -> EBM/H-JEPA next.
#
# Leo directive: #11575. Refs: the-search#3.

import json, re, time, heapq, numpy as np
import urllib.request, urllib.error
from pathlib import Path

# ── SUBSTRATE (for seed-baseline) ────────────────────────────────────────────
substrate_src = open('B:/M/the-search/experiments/components/SUBSTRATE.py').read()
run_idx = substrate_src.index('\n# RUN:')
exec(substrate_src[:run_idx])
BASIS_NAMES = list(BASIS.keys())

# ── Sandboxed code execution ──────────────────────────────────────────────────
SAFE_BUILTINS = {
    '__builtins__': {
        'range': range, 'len': len, 'zip': zip, 'enumerate': enumerate,
        'list': list, 'tuple': tuple, 'set': set, 'dict': dict,
        'min': min, 'max': max, 'sum': sum, 'abs': abs,
        'int': int, 'float': float, 'bool': bool, 'str': str,
        'isinstance': isinstance, 'type': type, 'sorted': sorted,
        'any': any, 'all': all, 'map': map, 'filter': filter,
        'reversed': reversed, 'print': print,
    },
    'np': np,
}

def sandboxed_exec(code_str, grid_in, timeout_s=3.0):
    ns = dict(SAFE_BUILTINS)
    try:
        exec(code_str, ns)
        solve_fn = ns.get('solve')
        if solve_fn is None:
            return None, "no 'solve' function defined"
        arr = np.array(grid_in, dtype=np.int64)
        result = solve_fn(arr)
        if result is None:
            return None, "solve() returned None"
        return np.array(result, dtype=np.int64), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"

def extract_python_code(text):
    if text is None:
        return None
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\s*(def solve.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'(def solve\s*\(.*?)(?:\n\n|\Z)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

# ── LLM client ────────────────────────────────────────────────────────────────
LLAMA_URL = "http://127.0.0.1:9876/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are an ARC (Abstraction and Reasoning Corpus) program synthesizer. "
    "Given input/output grid pairs (2D arrays of integers 0-9), write a Python function "
    "that transforms the input to the output.\n\n"
    "Rules:\n"
    "- Function signature: def solve(grid):\n"
    "- Input/output: numpy arrays of shape (H, W), dtype int64\n"
    "- You may use numpy (imported as np) and standard Python\n"
    "- Return the transformed grid as a numpy array\n"
    "- Output ONLY the function in a ```python ... ``` code block, no explanation"
)

def make_prompt(train_pairs, prev_attempt=None, error_msg=None):
    lines = []
    for i, pair in enumerate(train_pairs[:3]):
        lines.append(f"Example {i+1}:")
        lines.append(f"  input  = {pair['input']}")
        lines.append(f"  output = {pair['output']}")
    if prev_attempt and error_msg:
        lines.append(f"\nYour previous attempt failed:")
        lines.append(f"```python\n{prev_attempt}\n```")
        lines.append(f"Error: {error_msg}")
        lines.append("\nWrite a corrected solve(grid) function.")
    else:
        lines.append("\nWrite a solve(grid) function that performs this transformation.")
    return "\n".join(lines)

def llm_generate(prompt, temperature=0.4, timeout=60):
    payload = json.dumps({
        "model": "local",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": temperature,
        "stream": False
    }).encode()
    req = urllib.request.Request(
        LLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            return (msg.get("content") or "").strip() or None
    except Exception:
        return None

def verify_on_pairs(code_str, pairs):
    for pair in pairs:
        result, err = sandboxed_exec(code_str, pair['input'])
        if err or result is None:
            return False, err or "None result"
        expected = np.array(pair['output'], dtype=np.int64)
        if not np.array_equal(result, expected):
            return False, f"wrong output: got shape {result.shape}, expected {expected.shape}"
    return True, None

# ── Seed-enumeration baseline ─────────────────────────────────────────────────
def h_dist(a, b):
    return abs(a.shape[0] - b.shape[0]) + abs(a.shape[1] - b.shape[1])

def seed_solve(x, y, budget=500, max_depth=3):
    if np.array_equal(x, y):
        return True, 0
    ctr = [0]
    frontier = []
    heapq.heappush(frontier, (h_dist(x, y) * 2, 0, ctr[0], (), x))
    seen = set()
    exp = 0
    while frontier and exp < budget:
        pri, depth, _, seq, cur = heapq.heappop(frontier)
        exp += 1
        k = (cur.shape, cur.tobytes()[:24])
        if k in seen:
            continue
        seen.add(k)
        if depth >= max_depth:
            continue
        for nm in BASIS_NAMES:
            try:
                nxt = BASIS[nm](cur)
            except Exception:
                continue
            if np.array_equal(nxt, y):
                return True, exp
            h = h_dist(nxt, y)
            ctr[0] += 1
            heapq.heappush(frontier, (h*2+depth+1, depth+1, ctr[0], seq+(nm,), nxt))
    return False, exp

# ── Data loading ──────────────────────────────────────────────────────────────
D2 = json.load(open('incoming/arc-agi1-visa/03_R4_transfer_wall/D2_1_final_out.json'))
ARC_EVAL = Path('incoming/arc-agi1-visa/03_R4_transfer_wall/ARC-AGI/data/evaluation')

rng_sample = np.random.default_rng(42)   # same seed as Stage 0' for task comparability
N_TASKS = 30
N_CANDIDATES = 5
TIME_CAP = 600    # raised from 260s -- 26B is slower per call; decisive test warrants more compute

sample_ids = list(rng_sample.choice(D2['ids'], size=min(N_TASKS, len(D2['ids'])), replace=False))

# ── Wait for server ───────────────────────────────────────────────────────────
print("Waiting for gemma-4-26B at :9876 (may take 60-90s to load 26B weights)...")
t_wait = time.time()
while time.time() - t_wait < 180:
    try:
        req = urllib.request.Request("http://127.0.0.1:9876/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if json.loads(resp.read()).get("status") == "ok":
                print(f"  ready after {time.time()-t_wait:.1f}s")
                break
    except Exception:
        time.sleep(3)
else:
    print("ERROR: server not ready in 180s. Abort.")
    exit(1)

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\nCapability-isolation: {N_TASKS} tasks x N={N_CANDIDATES} attempts -- gemma-4-26B")
print(f"Variable changed: core scale (2B -> 26B). Substrate held fixed.")
print(f"Pre-registered: CORE_ONLY > 0 -> capability wall -> Stage 1")
print(f"Pre-registered: CORE_ONLY = 0 -> structural wall -> EBM/H-JEPA next")
print("=" * 70)

results = []
t0 = time.time()
n_llm_solves = 0
n_seed_solves = 0

for i, tid in enumerate(sample_ids):
    if time.time() - t0 > TIME_CAP:
        print(f"\n[TIME CAP {TIME_CAP}s] Reached after {i} tasks.")
        break

    p = ARC_EVAL / f"{tid}.json"
    if not p.exists():
        continue
    task = json.load(open(p))
    train = task['train']

    x0 = np.array(train[0]['input'],  dtype=np.int64)
    y0 = np.array(train[0]['output'], dtype=np.int64)
    seed_ok, seed_exp = seed_solve(x0, y0)

    solved_code = None
    prev_code = None
    prev_err = None
    attempts = []

    for attempt in range(N_CANDIDATES):
        if time.time() - t0 > TIME_CAP:
            break
        prompt = make_prompt(train, prev_code, prev_err)
        temp = 0.4 + attempt * 0.1   # same temperature schedule as Stage 0'
        raw = llm_generate(prompt, temperature=temp)
        code = extract_python_code(raw)
        if code is None:
            prev_err = "no parseable code block in response"
            attempts.append({"attempt": attempt, "code": None, "solved": False, "error": prev_err})
            continue
        ok, err = verify_on_pairs(code, train)
        attempts.append({"attempt": attempt, "code": code[:200], "solved": ok, "error": err})
        if ok:
            solved_code = code
            break
        prev_code = code
        prev_err = err

    llm_ok = solved_code is not None
    if llm_ok:
        n_llm_solves += 1
    if seed_ok:
        n_seed_solves += 1

    status = "CODE-SOLVE" if llm_ok else ("SEED-SOLVE" if seed_ok else "unsolved")
    elapsed = time.time() - t0
    n_attempts = len(attempts)
    parseable = sum(1 for a in attempts if a['code'] is not None)
    print(f"  [{i+1:2d}/{N_TASKS}] {tid[:8]}: {status}  attempts={n_attempts}  parse={parseable}/{n_attempts}  elapsed={elapsed:.1f}s")
    if llm_ok:
        print(f"    code: {solved_code[:100].strip()}")

    results.append({
        "task_id": tid,
        "seed_solved": seed_ok, "llm_solved": llm_ok,
        "n_attempts": n_attempts, "attempts": attempts
    })

# ── Verdict ───────────────────────────────────────────────────────────────────
n_tested = len(results)
total_time = time.time() - t0
parse_ok = sum(1 for r in results for a in r['attempts'] if a['code'] is not None)
total_attempts = sum(r['n_attempts'] for r in results)

print("\n" + "=" * 70)
print("CAPABILITY-ISOLATION RESULT (scale: gemma-4-26B vs gemma-4-2B)")
print("=" * 70)
print(f"Tasks tested:          {n_tested}/{N_TASKS}")
print(f"Code-synthesis solves: {n_llm_solves}/{n_tested} ({100*n_llm_solves/max(n_tested,1):.1f}%)")
print(f"Seed-basis solves:     {n_seed_solves}/{n_tested}")
print(f"Parse success rate:    {parse_ok}/{total_attempts} ({100*parse_ok/max(total_attempts,1):.1f}%)")
print(f"Stage 0' baseline:     0/9 solves, 35/42 parse (83%) -- 2B core")
print(f"Total time:            {total_time:.1f}s")
print()

if n_llm_solves > 0:
    verdict = "CAPABILITY_WALL"
    print(f"CAPABILITY WALL: 26B core solved {n_llm_solves} tasks. 2B failure was a capability gap.")
    print("Proceed to Stage 1 (meta-layer abstraction-accumulation -- the ACTUAL RSI test).")
elif n_llm_solves == 0:
    verdict = "STRUCTURAL_WALL"
    print("STRUCTURAL WALL: 26B core still 0 solves. Rule-inference from 2-3 I/O examples")
    print("  is fundamentally hard at this level -- not a capability gap.")
    print("Next: LeCun EBM / H-JEPA (energy-based inference-by-optimization) -- pre-registered.")
else:
    verdict = "MARGINAL"

result_data = {
    "experiment": "E2.2-capiso-scale",
    "issue_ref": "the-search#3",
    "date": "2026-05-29",
    "baseline_ref": "E2.2-stage0prime (2B, 384 thinking, 0/9 solves, 83% parse)",
    "design": {
        "core": "gemma-4-26B-A4B-it-Q4_K_L at :9876, reasoning-budget=512",
        "scale_vs_baseline": "26B vs 2B (13x parameter scale, same architecture)",
        "method": "code-synthesis + sandboxed exec + execution-feedback refinement",
        "substrate_held_fixed": True,
        "n_tasks_sampled": N_TASKS,
        "n_tasks_tested": n_tested,
        "n_candidates_per_task": N_CANDIDATES,
        "sample_seed": 42,
        "time_cap_s": TIME_CAP,
        "max_tokens_per_call": 1024,
        "timeout_per_call_s": 60,
    },
    "baselines": {
        "seed_enumeration_solves": n_seed_solves,
        "oracle_ceiling_pct": 0.7,
    },
    "llm_results": {
        "solves": n_llm_solves,
        "solve_rate_pct": round(100*n_llm_solves/max(n_tested,1), 2),
        "total_attempts": total_attempts,
        "parseable_code_attempts": parse_ok,
        "parse_rate_pct": round(100*parse_ok/max(total_attempts,1), 2),
    },
    "mechanism_split": {
        "stage0prime_parse_rate": 83.3,
        "stage0prime_solve_rate": 0.0,
        "capiso_parse_rate": round(100*parse_ok/max(total_attempts,1), 2),
        "capiso_solve_rate": round(100*n_llm_solves/max(n_tested,1), 2),
        "finding": "parse_rate shows format-compliance; solve_rate shows rule-inference"
    },
    "verdict": verdict,
    "total_runtime_s": round(total_time, 1),
    "task_results": results
}

with open('incoming/arc-agi1-visa/03_R4_transfer_wall/E2_2_capiso_scale_result.json', 'w') as f:
    json.dump(result_data, f, indent=2)
print("\nResult written to E2_2_capiso_scale_result.json")
