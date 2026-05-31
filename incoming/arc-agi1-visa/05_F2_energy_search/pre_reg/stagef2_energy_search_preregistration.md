# Stage F2 Pre-Registration — Energy-Ranked Best-First vs BFS Reach

**Locked 2026-05-31. Do not modify after first run.**

## Purpose

Test whether frozen E_theta (from F1 PASS) guides program search to higher reach within the
same budget. ONE variable = candidate ordering (energy-ranked within batch vs random BFS).
Grammar, budget, time wall, held set, and per-task seed are all held constant.

F1 interpretation contract (locked, Leo mail #12210):
"PASS = necessary-not-sufficient → licenses GO F2 ONLY. Does NOT validate the EBM face."

---

## Locked Design Decisions

### Held eval split
- Source: Stage 1d result, seed=42 (`stage1d_premise_truth_b30000_minimality_result.json`)
- `n_held_tasks` = 200
- `eval_split_hash` = `08be6f980cec510c`
- `arm_bfs_n_solved` at B=300K/armseed=7 = **10** (Stage 1k armseed7 result; gate asserts this)

### Grammar (frozen from Stage 1k)
- Leaf set: 1370-leaf Stage 1g/1h/1i/1j/1k space (unchanged)
- `space_hash` = `4978b6739bf55beb` (gate asserts; time wall not in descriptor)
- `max_depth` = 3
- `budget_b` = 300,000

### Time wall
- `time_per_task_s` = 1800.0 (raised from Stage 1k's 600s; budget is binding constraint for both arms)

### Per-task RNG seeding
- `arm_d_seed` = 7 (same as Stage 1k armseed7; arms share the same rng stream → same programs sampled)
- `per_task_seed_formula` = `(7, task_id)`

### Frozen energy model (E_theta from F1)
- `frozen_energy_hash` = `2cd1ed64b357292b`
  (sha256[:16] of concatenated float64 bytes of W1,b1,W2,b2,W3,b3 after deterministic F1 training)
- Architecture: `mlp_histogram_36d` (same as F1; 36→64→32→1, ReLU, Adam lr=1e-3, 100 epochs, seed=0)
- Training corpus: 200 ARC training tasks minus the 200 held tasks (same as F1)
- Gate asserts hash at run time; abort-without-write if mismatch (frozen_energy_hash ≠ computed hash)

### Energy-ranked arm (arm_energy)
- BATCH_SIZE = 500 programs per ranking batch
- Within each batch:
  1. Sample 500 programs (same rng stream as BFS arm — same programs drawn)
  2. Apply each to the task's first training input → output_candidate
  3. If program errors (output=None): assign energy = 1.0 (worst rank)
  4. Score output_candidate with frozen E_theta using task_train_examples
  5. Sort batch by energy ascending (lowest = most likely correct)
  6. Check exact match (`check_prog` over all training pairs) in energy-sorted order
- n_evals counted at scoring phase: `n_evals += len(batch)` per batch, symmetric with BFS
  (BFS counts `n_evals += 1` per program sampled; energy arm counts all scored programs per batch)
- `n_evals_per_task` = total programs scored across all batches until solve or budget exhausted

### BFS arm (arm_bfs)
- Exact Stage 1k armseed7 replay: same rng, same budget, random order
- Gate asserts `arm_bfs_n_solved == 10`

---

## Verdict

| Verdict | Condition |
|---------|-----------|
| SIGNAL | arm_energy_n_solved > arm_bfs_n_solved (energy ranking improves reach) |
| NULL | arm_energy_n_solved == arm_bfs_n_solved (no improvement from ordering) |
| REGRESS | arm_energy_n_solved < arm_bfs_n_solved (ordering hurts) |

F2 SIGNAL → licenses F3: does energy ranking improve generalization (test output correctness)?
F2 NULL or REGRESS → hypothesis rejected; energy discriminability (F1 PASS) does not guide search.

---

## Fail-Closed Gate Fields

All assertions run before artifact write. Any hard violation → abort without write.

| Field | Asserted value |
|-------|---------------|
| `frozen_energy_hash` | `"2cd1ed64b357292b"` |
| `eval_split_hash` | `"08be6f980cec510c"` |
| `space_hash` | `"4978b6739bf55beb"` |
| `arm_bfs_n_solved` | 10 (must reproduce Stage 1k armseed7) |
| per-task `n_evals` ≤ `budget_b` | Both arms: max n_evals ≤ 300,000 |

### Gate injection tests (--test-gate flag)

1. **Wrong frozen_energy_hash**: inject "0000000000000000" constant → gate fires on hash comparison
2. **Budget-overrun (BFS arm)**: inject per-task n_evals = BUDGET_B + 1 → gate fires
3. **Budget-overrun (energy arm)**: inject per-task n_evals = BUDGET_B + 1 → gate fires
4. **BFS-not-10**: inject arm_bfs_n_solved = 9 → gate fires (baseline must reproduce 10)
5. **eval_split_hash mismatch**: inject wrong hash → gate fires
6. **space_hash mismatch**: inject wrong hash → gate fires
7. **F1 frozen-link (behavioral)**: use untrained MLP (rng_seed=1) → energies diverge from F1's
   per_candidate_records by >1e-4 → frozen-link check fires

---

## Artifact Schema

Result file: `stagef2_energy_search_result.json`

```json
{
  "stage": "F2_energy_search",
  "frozen_energy_hash": "2cd1ed64b357292b",
  "eval_split_hash": "08be6f980cec510c",
  "space_hash": "4978b6739bf55beb",
  "arm_d_seed": 7,
  "budget_b": 300000,
  "time_per_task_s": 1800.0,
  "batch_size_energy": 500,
  "arm_bfs_n_solved": 10,
  "arm_energy_n_solved": ...,
  "verdict": "SIGNAL|NULL|REGRESS",
  "gate_violations": [],
  "arm_bfs_per_task": {...},
  "arm_energy_per_task": {...}
}
```

Smoke result file: `stagef2_energy_search_TEMP.json` (3 tasks, `--smoke` flag)

---

## R1-R6 Status

- **R1**: LIVE — E_theta training corpus disjoint from held eval (same disjointness as F1)
- **R5**: LIVE — fixed held ARC tasks (same 200 as Stage 1d/1k/F1)
- R2/R3/R4/R6: N/A (pure reach measurement; no self-modification)

---

*Eli 2026-05-31. Issue #17 (wordingone/the-search).*
