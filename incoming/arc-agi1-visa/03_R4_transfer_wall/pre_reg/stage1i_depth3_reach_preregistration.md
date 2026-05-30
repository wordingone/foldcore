# the-search -- Stage 1i: Depth-3 Reach -- PRE-REGISTERED

**Status: PRE-REGISTERED — harness built; PR pending Leo layer-1 diff-review.**

Gates resolved:
- [x] Stage 1h PASS (D=7, FLAT, budget-binding): Leo integer-gate (#4 verdict, mail #12077)
- [x] Leo GO for Stage 1i (mail #12077): "Stage 1i filed as #6 — depth-2 → depth-3, ONE variable (MAX_DEPTH)"
- [x] Gate-hole identified: `max(per-task n_evals) <= BUDGET_B` positive behavioral assertion added
- [ ] Leo layer-1 PR diff-review (harness lands via PR — Leo verifies ONE variable + gate-hole fix)
- [ ] TEMP-smoke verified by Leo (gate fires on each injected violation INCLUDING over-budget behavioral catch)
- [ ] Canonical seed-42 GO from Leo after smoke verify

Design decisions LOCKED at pre-registration (Leo, mail #12077, issue #6):
- Q1: Grammar leaf SET UNCHANGED (same 1370-leaf Stage 1g/1h space). ONE variable = MAX_DEPTH only.
- Q2: MAX_DEPTH 2 → 3. Budget B=30000, arm seed=0, 120s/task, same 200 held.
- Q3: Depth-3 composes same leaves one layer deeper (4-tuple search). No new primitives.
- Q4: Space hash MUST change (depth embedded in hash). Gate asserts `space_hash != prev (0890317fe99bc9f1)`.
- Q5: NEW gate: `max(per-task n_evals) <= BUDGET_B` — behavioral enforcement. Catches 1g-class overruns.

---

## Context: Stage 1h established genuine budget-bound ceiling

Stage 1h (D=7/200, time_limit_hit=2/200, budget_exhausted=191/200) confirmed: the ceiling
is GENUINE at depth-2/B=30000, not an artifact of the time wall. This is the first clean
depth comparison at a faithful B=30000 (1e/1f/1g ran unbounded ~88K/task).

This experiment answers: does adding a composition layer (depth-2 → depth-3) raise reach?

---

## Stage summary table

| Stage | Arm | Grammar | Depth | Time/task | Budget enforced | Seed-42 | Seed-1 |
|-------|-----|---------|-------|-----------|-----------------|---------|--------|
| 1e | D (baseline) | 330-leaf | 2 | 9s | No (~88K) | 11 | 8 |
| 1f | D' (+MAP_GEOM) | 410-leaf | 2 | 9s | No (~88K) | 9 | 7 |
| 1g | D'' (+MAP_RELATE) | 1370-leaf | 2 | 9s | No (~88K) | 9 | -- (FLAT) |
| 1h | D (time-sat.) | 1370-leaf | 2 | 120s | Yes (30K) | 7 | -- (FLAT) |
| 1i | D (depth-3) | 1370-leaf | **3** | 120s | Yes (30K) | TBD | TBD |

ONE variable between 1h and 1i: MAX_DEPTH (2 → 3).

---

## Protocol

- Grammar leaf SET: same 1370-leaf Stage 1g/1h space (MAP_RELATE included)
- MAX_DEPTH: 3 (was 2 -- ONE variable)
- Search: 4-tuple random sampling (i, j, k, l) via `check_tuple((i, j, k, l), ...)`
- Per-task time limit: 120s (held constant)
- Budget: B = 30000 candidates per task (loop exits on budget exhaustion)
- Held split: same 200 tasks from Stage 1d (assert held_task_ids_match_stage1d)
- Arm seed: 0 (held constant)
- PREV_SPACE_HASH: "0890317fe99bc9f1" (Stage 1h 1370-leaf depth-2 hash)

---

## Fail-closed gate (PREV_SPACE_HASH = "0890317fe99bc9f1")

Violations checked at artifact-write (abort-without-write on any):
1. n_leaves != builder ACTUAL (leaf set must still be 1370)
2. n_leaves_asserted != True
3. held_task_ids count != 200
4. held_task_ids set mismatch vs stage1d
5. candidate_eval_budget != 30000
6. candidate_eval_budget_asserted != True
7. max_depth != 3 (ONE-variable integrity: was 2 in Stage 1h)
8. space_hash == PREV_SPACE_HASH (depth-3 MUST produce a different hash)
9. prev_space_hash != PREV_SPACE_HASH
10. **NEW**: max(per-task n_evals) > BUDGET_B (behavioral enforcement -- catches 1g-class overruns)
11. per-task: unsolved AND NOT time_limit_hit AND n_evals < 30000 (premature exit)

Gate injection tests MUST include:
- Test 7: max_depth wrong (inject max_depth=2 -- Stage 1h value)
- Test 8: space_hash == prev (inject PREV_SPACE_HASH -- depth change missed)
- Test 10: over-budget task (inject n_evals=BUDGET_B+1 -- behavioral catch)

---

## Decision tree (written BEFORE running)

Stage 1h baseline: D=7/200, budget_exhausted=191/200, time_limit_hit=2/200.

Seed-42 result:
- D > 7: depth-3 adds reach within B=30000. Depth is a live lever.
  Next: depth-4 OR scale budget at depth-3.
- D == 7: depth-3 adds nothing within B=30000. Grammar-EXPRESSIVITY ceiling, not depth.
  CAVEAT: D==7 with budget_exhausted high is also consistent with "30K too few for
  depth-3's larger space." Disambiguating follow: scale B at depth-3. Do NOT conflate.
  Next: grammar enrichment OR self-compilation track (pre-register the B-scaling stage first).
- D < 7: search BUG. Depth-3 strictly contains depth-2 subspace. HOLD, debug.
  Do not report to Leo until root cause found.
- time_limit_hit_count high: 120s insufficient at depth-3 (4-tuple candidates slower).
  Raise wall before trusting D.

Two-seed protocol: run seed-1 if D == 7 (ambiguous) AND time_limit_hit_count ~ 0.
If D > 7 or D < 7, seed-1 not needed.

---

-- Leo (directive, mail #12077 + issue #6), Eli (pre-registration + harness), 2026-05-30
