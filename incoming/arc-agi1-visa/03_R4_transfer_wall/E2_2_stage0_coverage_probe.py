# E2_2_stage0_coverage_probe.py — E2.2' Stage-0 CORE_ONLY coverage probe (the-search#3).
#
# Architecture: frozen LLM at :9876 (Qwen3.6-27B-Q4_K_M) as generative program-proposer.
# Proposes K candidate programs conditioned on I/O examples. Parse + execute via SUBSTRATE.py.
#
# Pre-registered kill: LLM-alone does NOT expand coverage above seed/oracle on sampled subset
#   → coverage-expansion via this core is FALSIFIED → stop.
# Pre-registered forward: LLM solves ANY task seed-enumeration (budget=20000) cannot
#   → coverage-expansion confirmed → proceed to Stage 1.
#
# Seed-enumeration baseline: 0/395 solves (K1 probe, budget=500; D2 formal bound budget=20000).
# Oracle ceiling: 0.7% (Leg 3). Beating either = coverage expansion.
#
# Leo directive: #11531. Refs: the-search#3.

import json, re, time, heapq, numpy as np, urllib.request, urllib.error
from pathlib import Path
from collections import defaultdict

# ── SUBSTRATE ────────────────────────────────────────────────────────────────
substrate_src = open('B:/M/the-search/experiments/components/SUBSTRATE.py').read()
run_idx = substrate_src.index('\n# RUN:')
exec(substrate_src[:run_idx])

BASIS_NAMES = list(BASIS.keys())

# ── LLM client ────────────────────────────────────────────────────────────────
LLAMA_URL = "http://127.0.0.1:9876/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are an ARC program synthesizer. Given input/output grid pairs, propose a "
    "program as a Python list of operation names. Available operations:\n"
    "  id, fh (flip horizontal), fv (flip vertical), tr (transpose),\n"
    "  rot (rotate 90° CW), crop (crop to non-zero bounding box),\n"
    "  dup_h (duplicate horizontally), dup_v (duplicate vertically),\n"
    "  mir_h (mirror left+right), mir_v (mirror top+bottom),\n"
    "  up2 (2x upscale), down2 (2x downscale)\n"
    "Operations are applied left-to-right. Output ONLY a Python list, e.g. ['rot', 'fh']."
)

def grid_str(grid):
    return "[" + ",\n ".join(str(list(row)) for row in grid) + "]"

def build_user_prompt(train_pairs):
    """Format all train pairs as context, then ask for program."""
    lines = []
    for i, pair in enumerate(train_pairs):
        lines.append(f"Example {i+1}:")
        lines.append(f"  Input:  {grid_str(pair['input'])}")
        lines.append(f"  Output: {grid_str(pair['output'])}")
    lines.append("\nPropose a program that performs this transformation. Output ONLY a Python list.")
    return "\n".join(lines)

def llm_propose(train_pairs, temperature=0.7, timeout=30):
    """Call LLM, return raw response text."""
    user_msg = build_user_prompt(train_pairs)
    payload = json.dumps({
        "model": "local",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ],
        "max_tokens": 2048,
        "temperature": temperature,
        "stream": False
    }).encode()
    req = urllib.request.Request(
        LLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return None

def parse_program(text):
    """Extract list of op names from LLM response. Returns list or None."""
    if text is None:
        return None
    # Strip thinking blocks — Qwen3 emits <think>...</think> before the answer
    think_stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    search_text = think_stripped if think_stripped else text
    # Match a Python list literal
    m = re.search(r'\[([^\[\]]*)\]', search_text)
    if not m:
        return None
    inner = m.group(1)
    # Extract quoted strings
    ops = re.findall(r"'([^']+)'|\"([^\"]+)\"", inner)
    prog = [a or b for a, b in ops]
    # Validate: all ops must be in BASIS
    valid = [op for op in prog if op in BASIS]
    return valid if valid else None

def try_execute(prog, train_pairs):
    """Try program on all train pairs. Returns True if all match."""
    for pair in train_pairs:
        x = np.array(pair['input'],  dtype=np.int64)
        y = np.array(pair['output'], dtype=np.int64)
        try:
            result = x.copy()
            for op in prog:
                result = BASIS[op](result)
            if not np.array_equal(result, y):
                return False
        except Exception:
            return False
    return True

# ── Seed-enumeration baseline ─────────────────────────────────────────────────
def h_dist(a, b):
    return abs(a.shape[0] - b.shape[0]) + abs(a.shape[1] - b.shape[1])

def seed_solve(x, y, budget=500, max_depth=3):
    """Seed-basis BFS. Returns (solved, expansions)."""
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
            heapq.heappush(frontier, (h * 2 + depth + 1, depth + 1, ctr[0], seq + (nm,), nxt))
    return False, exp

# ── Data loading ──────────────────────────────────────────────────────────────
D2 = json.load(open('incoming/arc-agi1-visa/03_R4_transfer_wall/D2_1_final_out.json'))
ARC_EVAL = Path('incoming/arc-agi1-visa/03_R4_transfer_wall/ARC-AGI/data/evaluation')

# Sample N tasks (fixed seed for reproducibility)
rng_sample = np.random.default_rng(99)
N_TASKS = 10
K_PROPOSALS = 3
TIME_CAP = 260  # seconds (within 5-min cap)

sample_ids = list(rng_sample.choice(D2['ids'], size=min(N_TASKS, len(D2['ids'])), replace=False))

# ── Wait for llama-server ─────────────────────────────────────────────────────
print("Waiting for llama-server at :9876...")
t_wait = time.time()
while time.time() - t_wait < 120:
    try:
        req = urllib.request.Request("http://127.0.0.1:9876/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = json.loads(resp.read())
            if status.get("status") == "ok":
                print(f"  llama-server ready after {time.time()-t_wait:.1f}s")
                break
    except Exception:
        time.sleep(2)
else:
    print("ERROR: llama-server did not start in 120s. Aborting.")
    exit(1)

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\nStage-0: {N_TASKS} tasks × K={K_PROPOSALS} proposals — Qwen3.6-27B @:9876")
print(f"Pre-registered kill: 0 LLM-only solves -> coverage-expansion FALSIFIED")
print(f"Pre-registered forward: >=1 LLM-only solve -> proceed to Stage 1")
print("=" * 70)

results = []
t0 = time.time()
llm_solves = []
seed_solves = []
parse_failures = 0
total_proposals = 0
invalid_proposals = 0

for i, tid in enumerate(sample_ids):
    if time.time() - t0 > TIME_CAP:
        print(f"\n[TIME CAP] Reached {TIME_CAP}s after {i} tasks. Stopping.")
        break

    p = ARC_EVAL / f"{tid}.json"
    if not p.exists():
        continue
    task = json.load(open(p))
    train_pairs = task['train']

    # Use only first train pair's first pair for seed-baseline check (speed)
    x0 = np.array(train_pairs[0]['input'],  dtype=np.int64)
    y0 = np.array(train_pairs[0]['output'], dtype=np.int64)
    seed_solved, seed_exp = seed_solve(x0, y0, budget=500, max_depth=3)

    # K proposals from LLM
    task_solves = 0
    task_proposals = []
    for k in range(K_PROPOSALS):
        if time.time() - t0 > TIME_CAP:
            break
        temp = 0.7 + k * 0.05  # slight temp variation across proposals
        raw = llm_propose(train_pairs[:2], temperature=temp)  # limit to 2 train pairs for speed
        total_proposals += 1
        prog = parse_program(raw)
        if prog is None:
            parse_failures += 1
            task_proposals.append({"raw": raw, "parsed": None, "solved": False})
            continue
        if len(prog) == 0:
            invalid_proposals += 1
            task_proposals.append({"raw": raw, "parsed": [], "solved": False})
            continue
        solved = try_execute(prog, train_pairs[:2])
        if solved:
            task_solves += 1
        task_proposals.append({"raw": raw[:80], "parsed": prog, "solved": solved})

    llm_solved = task_solves > 0
    llm_solves.append(llm_solved)
    seed_solves.append(seed_solved)

    status = "LLM-SOLVE" if llm_solved else ("SEED-SOLVE" if seed_solved else "unsolved")
    elapsed = time.time() - t0
    print(f"  [{i+1:2d}/{N_TASKS}] {tid[:8]}: {status}  "
          f"llm_proposals={len(task_proposals)}  elapsed={elapsed:.1f}s")
    if llm_solved:
        solved_prog = next(p['parsed'] for p in task_proposals if p['solved'])
        print(f"           LLM program: {solved_prog}")

    results.append({
        "task_id": tid,
        "seed_solved": seed_solved,
        "llm_solved": llm_solved,
        "proposals": task_proposals,
        "seed_exp": int(seed_exp)
    })

# ── Verdict ───────────────────────────────────────────────────────────────────
n_tested = len(results)
n_llm_solves = sum(r['llm_solved'] for r in results)
n_seed_solves = sum(r['seed_solved'] for r in results)
total_time = time.time() - t0

print("\n" + "=" * 70)
print("E2.2' STAGE-0 — COVERAGE PROBE RESULT")
print("=" * 70)
print(f"Tasks tested:       {n_tested}/{N_TASKS}")
print(f"LLM-only solves:    {n_llm_solves}/{n_tested} ({100*n_llm_solves/max(n_tested,1):.1f}%)")
print(f"Seed-basis solves:  {n_seed_solves}/{n_tested} ({100*n_seed_solves/max(n_tested,1):.1f}%)")
print(f"Oracle ceiling:     0.7% of 395 = ~3 tasks")
print(f"Parse failures:     {parse_failures}/{total_proposals}")
print(f"Total time:         {total_time:.1f}s")
print()

if n_llm_solves > n_seed_solves:
    verdict = "FORWARD"
    print(f"FORWARD: LLM solved {n_llm_solves} tasks; seed-basis solved {n_seed_solves}.")
    print("Coverage-expansion CONFIRMED. Proceed to Stage 1 (meta-layer + R6 kill).")
elif n_llm_solves > 0 and n_llm_solves == n_seed_solves:
    verdict = "AMBIGUOUS"
    print(f"AMBIGUOUS: LLM and seed-basis tied at {n_llm_solves} solves. Tasks may be in-closure.")
    print("Check if LLM-solved tasks are actually out-of-closure in D2_1_final_out.")
else:
    verdict = "KILL"
    print("KILL: LLM-alone did not expand coverage above seed-basis on this sample.")
    print("Coverage-expansion via frozen LLM proposer FALSIFIED.")
    print("Consistent with ARC-AGI-3 sub-1% collapse across all approaches.")
    print("Report the negative honestly. Stop E2.2' LLM path.")

result_data = {
    "experiment": "E2.2-stage0",
    "issue_ref": "the-search#3",
    "date": "2026-05-29",
    "design": {
        "core": "Qwen3.6-27B-Q4_K_M frozen LLM at :9876",
        "n_tasks_sampled": N_TASKS,
        "n_tasks_tested": n_tested,
        "k_proposals_per_task": K_PROPOSALS,
        "sample_seed": 99,
        "time_cap_s": TIME_CAP
    },
    "baselines": {
        "seed_enumeration_solves": int(n_seed_solves),
        "oracle_ceiling_pct": 0.7,
        "oracle_ceiling_expected_solves": round(N_TASKS * 0.007, 2)
    },
    "llm_results": {
        "solves": int(n_llm_solves),
        "solve_rate_pct": round(100 * n_llm_solves / max(n_tested, 1), 2),
        "total_proposals": int(total_proposals),
        "parse_failures": int(parse_failures),
        "invalid_programs": int(invalid_proposals)
    },
    "verdict": verdict,
    "preregistered_kill": "llm_solves <= seed_solves and llm_solves == 0 -> coverage FALSIFIED",
    "preregistered_forward": "llm_solves > seed_solves -> coverage expanded -> Stage 1",
    "total_runtime_s": round(total_time, 1),
    "task_results": results
}

with open('incoming/arc-agi1-visa/03_R4_transfer_wall/E2_2_stage0_result.json', 'w') as f:
    json.dump(result_data, f, indent=2)
print("\nResult written to E2_2_stage0_result.json")
