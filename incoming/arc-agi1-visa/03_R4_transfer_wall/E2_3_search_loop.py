# E2_3_search_loop.py -- Inference-by-optimization: search vs sample (the-search#3).
#
# One variable changed vs Stage 0'' (E2.2'): inference MODE.
#   Stage 0'': N=5 independent autoregressive samples -> STRUCTURAL_WALL (0/4 solves, 100% parse)
#   This:      iterative search over program space; budget 50-100 evals/task
#
# The test: does replacing sampling with search move the solve count?
# Energy = #train-examples-satisfied (partial credit).
# Each step conditions on best-so-far + its failure trace (concrete gradient signal).
#
# Pre-registered kill (Leo #11589):
#   search CORE_ONLY > 0 -> inference MODE was the wall -> search proposer is viable core
#   search CORE_ONLY = 0 -> BOTH autoregressive AND search exhausted locally ->
#     strategic escalation (not another variant)
#
# Report: search CORE_ONLY solve-count + eval-budget-vs-solve curve.
# Leo directive: #11589. Refs: the-search#3.

import json, re, time, heapq, numpy as np
import urllib.request, urllib.error
from pathlib import Path

# ---- SUBSTRATE (for seed-baseline) ------------------------------------------
substrate_src = open('B:/M/the-search/experiments/components/SUBSTRATE.py').read()
run_idx = substrate_src.index('\n# RUN:')
exec(substrate_src[:run_idx])
BASIS_NAMES = list(BASIS.keys())

# ---- Sandboxed code execution ------------------------------------------------
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

def sandboxed_exec(code_str, grid_in):
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

# ---- Energy function (partial credit) ----------------------------------------
def energy(code_str, train_pairs):
    """Returns (score 0.0-1.0, failure_info).
    score = matched_examples / total. failure_info = None if solved."""
    if code_str is None:
        return 0.0, {"example": 0, "error": "no code", "got": None}
    matched = 0
    first_fail = None
    for i, pair in enumerate(train_pairs):
        result, err = sandboxed_exec(code_str, pair['input'])
        expected = np.array(pair['output'], dtype=np.int64)
        if err is None and result is not None and np.array_equal(result, expected):
            matched += 1
        elif first_fail is None:
            got_repr = result.tolist() if result is not None else None
            first_fail = {
                "example": i,
                "input": pair['input'],
                "expected": pair['output'],
                "got": got_repr,
                "error": err,
            }
    score = matched / len(train_pairs)
    return score, first_fail

# ---- LLM client --------------------------------------------------------------
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

def make_initial_prompt(train_pairs):
    lines = []
    for i, pair in enumerate(train_pairs[:3]):
        lines.append(f"Example {i+1}:")
        lines.append(f"  input  = {pair['input']}")
        lines.append(f"  output = {pair['output']}")
    lines.append("\nWrite a solve(grid) function that performs this transformation.")
    return "\n".join(lines)

def make_search_prompt(train_pairs, best_code, best_score, fail_info):
    """Refinement prompt: best solution so far + concrete failure trace."""
    lines = []
    for i, pair in enumerate(train_pairs[:3]):
        lines.append(f"Example {i+1}:")
        lines.append(f"  input  = {pair['input']}")
        lines.append(f"  output = {pair['output']}")

    n = len(train_pairs)
    matched = int(round(best_score * n))
    lines.append(f"\nYour current best solution ({matched}/{n} training examples correct):")
    lines.append(f"```python\n{best_code}\n```")

    if fail_info:
        lines.append(f"\nFailure on Example {fail_info['example']+1}:")
        lines.append(f"  input    = {fail_info['input']}")
        lines.append(f"  expected = {fail_info['expected']}")
        if fail_info['got'] is not None:
            lines.append(f"  got      = {fail_info['got']}")
        if fail_info['error']:
            lines.append(f"  error    = {fail_info['error']}")

    lines.append("\nImprove the solution to fix this failure. Output ONLY a corrected ```python ... ``` block.")
    return "\n".join(lines)

def llm_generate(prompt, temperature=0.6, timeout=60):
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

# ---- Seed-enumeration baseline -----------------------------------------------
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

# ---- Data loading ------------------------------------------------------------
D2 = json.load(open('incoming/arc-agi1-visa/03_R4_transfer_wall/D2_1_final_out.json'))
ARC_EVAL = Path('incoming/arc-agi1-visa/03_R4_transfer_wall/ARC-AGI/data/evaluation')

rng_sample = np.random.default_rng(42)   # same seed as Stage 0' for task comparability
N_TASKS   = 20
N_INIT    = 5    # independent warm-start samples (same as Stage 0')
MAX_EVALS = 100  # total evals per task (N_INIT + up to 95 search steps)
CHECKPOINTS = [10, 25, 50, 100]
TIME_CAP  = 900  # 15 min -- more generous to get the budget-vs-solve curve

sample_ids = list(rng_sample.choice(D2['ids'], size=min(N_TASKS, len(D2['ids'])), replace=False))

# ---- Wait for server ---------------------------------------------------------
print("Waiting for 2B server at :9876...")
t_wait = time.time()
while time.time() - t_wait < 120:
    try:
        req = urllib.request.Request("http://127.0.0.1:9876/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if json.loads(resp.read()).get("status") == "ok":
                print(f"  ready after {time.time()-t_wait:.1f}s")
                break
    except Exception:
        time.sleep(3)
else:
    print("ERROR: server not ready in 120s. Abort.")
    exit(1)

# ---- Main loop ---------------------------------------------------------------
print(f"\nInference-by-optimization: {N_TASKS} tasks x {MAX_EVALS} max evals -- 2B search loop")
print(f"Phase 1: {N_INIT} independent samples (warm start)")
print(f"Phase 2: iterative refinement from best, conditioned on failure trace")
print(f"Checkpoints: {CHECKPOINTS}  TIME_CAP: {TIME_CAP}s")
print(f"Pre-registered: CORE_ONLY > 0 -> inference MODE was the wall")
print(f"Pre-registered: CORE_ONLY = 0 -> BOTH modes exhausted -> strategic escalation")
print("=" * 70)

all_results = []
t0 = time.time()
n_solves = 0
n_seed_solves = 0

# checkpoint solve counts
cp_solves = {cp: 0 for cp in CHECKPOINTS}

for i, tid in enumerate(sample_ids):
    if time.time() - t0 > TIME_CAP:
        print(f"\n[TIME CAP {TIME_CAP}s] Reached after {i} tasks.")
        break

    p = ARC_EVAL / f"{tid}.json"
    if not p.exists():
        continue
    task = json.load(open(p))
    train = task['train']

    x0 = np.array(train[0]['input'], dtype=np.int64)
    y0 = np.array(train[0]['output'], dtype=np.int64)
    seed_ok, seed_exp = seed_solve(x0, y0)
    if seed_ok:
        n_seed_solves += 1

    # ---- Phase 1: warm start ------------------------------------------------
    best_code   = None
    best_score  = -1.0
    best_fail   = None
    eval_count  = 0
    solved      = False
    energy_traj = []   # (eval_n, best_energy, new_energy, improved)
    cp_state    = {}   # checkpoint -> solved_bool

    temps_init = [0.4 + k * 0.1 for k in range(N_INIT)]   # 0.4..0.8
    for k in range(N_INIT):
        if time.time() - t0 > TIME_CAP:
            break
        prompt = make_initial_prompt(train)
        raw    = llm_generate(prompt, temperature=temps_init[k])
        code   = extract_python_code(raw)
        sc, fail = energy(code, train)
        eval_count += 1
        improved = sc > best_score
        if improved:
            best_code, best_score, best_fail = code, sc, fail
        energy_traj.append({"n": eval_count, "best": round(best_score, 3),
                             "new": round(sc, 3), "improved": improved})
        if best_score == 1.0:
            solved = True
            break

        # record checkpoints
        for cp in CHECKPOINTS:
            if eval_count == cp:
                cp_state[cp] = solved

    # ---- Phase 2: iterative search from best --------------------------------
    stall = 0
    while not solved and eval_count < MAX_EVALS:
        if time.time() - t0 > TIME_CAP:
            break

        # temperature: alternate between exploitation (0.5) and exploration (0.8)
        temp = 0.5 if stall < 5 else 0.85

        if best_code is not None:
            prompt = make_search_prompt(train, best_code, best_score, best_fail)
        else:
            prompt = make_initial_prompt(train)

        raw  = llm_generate(prompt, temperature=temp)
        code = extract_python_code(raw)
        sc, fail = energy(code, train)
        eval_count += 1

        improved = sc > best_score
        if improved:
            best_code, best_score, best_fail = code, sc, fail
            stall = 0
        else:
            stall += 1

        energy_traj.append({"n": eval_count, "best": round(best_score, 3),
                             "new": round(sc, 3), "improved": improved})

        if best_score == 1.0:
            solved = True

        for cp in CHECKPOINTS:
            if eval_count == cp:
                cp_state[cp] = solved

    # fill remaining checkpoints
    for cp in CHECKPOINTS:
        if cp not in cp_state:
            cp_state[cp] = solved

    if solved:
        n_solves += 1
        for cp in CHECKPOINTS:
            if cp_state.get(cp, False):
                cp_solves[cp] += 1

    status  = "SOLVED" if solved else f"best={best_score:.2f}"
    elapsed = time.time() - t0
    print(f"  [{i+1:2d}/{N_TASKS}] {tid[:8]}: {status}  evals={eval_count}  seed={'Y' if seed_ok else 'N'}  elapsed={elapsed:.1f}s")

    all_results.append({
        "task_id":      tid,
        "solved":       solved,
        "seed_solved":  seed_ok,
        "eval_count":   eval_count,
        "best_energy":  round(best_score, 4),
        "cp_state":     cp_state,
        "energy_traj":  energy_traj,
    })

# ---- Results -----------------------------------------------------------------
n_tested   = len(all_results)
total_time = time.time() - t0

print("\n" + "=" * 70)
print("INFERENCE-BY-OPTIMIZATION RESULT (search loop vs Stage 0'' sampling)")
print("=" * 70)
print(f"Tasks tested:             {n_tested}/{N_TASKS}")
print(f"Search CORE_ONLY solves:  {n_solves}/{n_tested} ({100*n_solves/max(n_tested,1):.1f}%)")
print(f"Seed-basis solves:        {n_seed_solves}/{n_tested}")
print(f"Stage 0'' baseline:       0/4 solves, N=5 independent samples, 100% parse -- 2B")
print()
print("Eval-budget-vs-solve curve:")
for cp in CHECKPOINTS:
    tasks_at_cp = sum(1 for r in all_results if r['eval_count'] >= cp or r['solved'])
    print(f"  at {cp:3d} evals: {cp_solves[cp]}/{tasks_at_cp} solved")
print()

# energy improvement stats
max_energies = [r['best_energy'] for r in all_results]
if max_energies:
    print(f"Best energy distribution across tasks:")
    print(f"  max={max(max_energies):.3f}  mean={sum(max_energies)/len(max_energies):.3f}")
    above_zero = sum(1 for e in max_energies if e > 0)
    print(f"  tasks with any partial credit: {above_zero}/{n_tested}")
print()

if n_solves > 0:
    verdict = "INFERENCE_MODE_WALL"
    print(f"SEARCH SOLVES: {n_solves} task(s) solved. Inference MODE was the wall.")
    print("Search-based proposer is a viable core. Proceed to proper CORE_ONLY measurement.")
else:
    verdict = "BOTH_MODES_EXHAUSTED"
    print("SEARCH CORE_ONLY = 0. BOTH autoregressive AND search exhausted at local scale.")
    print("This is a STRATEGIC FINDING: local-substrate RSI program may not be viable.")
    print("Next: strategic escalation conversation (not another variant).")

print(f"\nTotal time: {total_time:.1f}s")

result_data = {
    "experiment":     "E2.3-search-loop",
    "issue_ref":      "the-search#3",
    "date":           "2026-05-29",
    "baseline_ref":   "E2.2-capiso-scale (26B, 0/4 solves, 100% parse, N=5 independent)",
    "design": {
        "proposer":       "gemma-4-2B at :9876",
        "inference_mode": "iterative-search (hill-climb from best-so-far + failure trace)",
        "n_init_samples": N_INIT,
        "max_evals_per_task": MAX_EVALS,
        "checkpoints":    CHECKPOINTS,
        "energy_fn":      "matched_train_examples / total_train_examples",
        "n_tasks_sampled": N_TASKS,
        "n_tasks_tested":  n_tested,
        "sample_seed":     42,
        "time_cap_s":      TIME_CAP,
    },
    "results": {
        "search_core_only_solves": n_solves,
        "solve_rate_pct":          round(100*n_solves/max(n_tested,1), 2),
        "seed_basis_solves":       n_seed_solves,
        "cp_solves":               cp_solves,
        "max_energy_mean":         round(sum(max_energies)/max(len(max_energies),1), 4) if max_energies else 0,
        "max_energy_max":          round(max(max_energies), 4) if max_energies else 0,
        "tasks_above_zero_energy": sum(1 for e in max_energies if e > 0),
    },
    "verdict":         verdict,
    "total_runtime_s": round(total_time, 1),
    "task_results":    all_results,
}

out = 'incoming/arc-agi1-visa/03_R4_transfer_wall/E2_3_search_loop_result.json'
with open(out, 'w') as f:
    json.dump(result_data, f, indent=2)
print(f"Result written to {out}")
