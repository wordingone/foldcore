# the-search -- Stage 1h: Time-Wall Saturation -- PRE-REGISTERED

**Status: PRE-REGISTERED — build complete; TEMP-smoke pending; Leo smoke-verify pending before canonical.**

Gates resolved:
- [x] Stage 1g FLAT (D''=9 seed-42): MAP_RELATE adjacency did NOT raise reach ceiling. Leo GATED (mail #12063).
- [x] Leo GO for Stage 1h (mail #12064): "time-wall saturation BEFORE depth"
- [x] Leo integer-verification of Stage 1g confound: 191/200 tasks hit time_limit_hit -- time wall was binding, not budget
- [ ] Harness TEMP-smoke verified by Leo (gate fires on each injected violation; n_leaves=1370; time_limit_hit drops vs 1g)
- [ ] Canonical seed-42 GO from Leo after smoke verify

Design decisions LOCKED at pre-registration (Leo, mail #12064):
- Q1: Grammar UNCHANGED (1370-leaf Stage 1g space). ONE variable = time limit only.
- Q2: Time limit 9s -> 120s (~13x). Budget B=30000, depth-2, arm seed=0, same 200 held.
- Q3: Add budget exit to loop -- break at n_evals >= 30000 so budget_exhausted is observable.
- Q4: Gate fields: time_limit_hit_count + min/median n_evals over unsolved tasks.

---

## Context: Stage 1g confound identified by Leo

Stage 1g integer-level verification showed 191/200 tasks hit time_limit_hit at 9s/task.
Eval rate: ~10K evals/s => 30000 evals ~ 3s. Budget B=30000 was NOT the binding constraint.
"Width doesn't lift reach" is only valid if budget is saturated; demonstrably, it wasn't.
This experiment removes the time-wall confound before proceeding to depth-3 (Stage 1i).

---

## Stage summary table

| Stage | Arm | Grammar | Time/task | Seed-42 | Seed-1 |
|-------|-----|---------|-----------|---------|--------|
| 1e | D (baseline) | 330-leaf | 9s | 11 | 8 |
| 1f | D' (+MAP_GEOM) | 410-leaf | 9s | 9 | 7 |
| 1g | D'' (+MAP_RELATE) | 1370-leaf | 9s | 9 | -- (FLAT) |
| 1h | D (time-saturated) | 1370-leaf | 120s | TBD | TBD |

ONE variable between 1g and 1h: per-task time limit (9s -> 120s).

---

## Protocol: time-wall saturation

- Grammar: same 1370-leaf Stage 1g space (EXPECTED_PREV_SPACE_HASH = 0890317fe99bc9f1)
- Per-task time limit: 120s (was 9s -- x13.3)
- Budget: B = 30000 candidates per task (loop also exits on budget exhaustion)
- Held split: same 200 tasks from Stage 1d (assert held_task_ids_match_stage1d)
- Depth: 2 (unchanged)
- Arm seed: 0 (unchanged)
- Two-seed: seed-42 first (diagnostic), seed-1 (confirmatory)

---

## Fail-closed gate (PREV_SPACE_HASH = "0890317fe99bc9f1")

Violations checked at artifact-write (abort-without-write on any):
1. n_leaves != builder ACTUAL (must be 1370)
2. n_leaves_asserted != True
3. held_task_ids count != 200
4. held_task_ids set mismatch vs stage1d
5. candidate_eval_budget != 30000
6. candidate_eval_budget_asserted != True
7. time_limit_s != 120.0 (ONE-variable integrity: was 9.0 in Stage 1g)
8. space_hash != PREV_SPACE_HASH (grammar must be UNCHANGED from Stage 1g)
9. prev_space_hash != PREV_SPACE_HASH
10. per-task: unsolved AND NOT time_limit_hit AND n_evals < 30000 (premature exit)

New gate fields (reported, not blocking unless premature exit):
- time_limit_hit_count: count of tasks where time wall hit before budget
- budget_exhausted_count: count of tasks where budget hit before time wall
- min_evals_unsolved, median_evals_unsolved: distribution of evals on unsolved tasks

---

## Decision tree (written BEFORE running)

Stage 1g baseline: time_limit_hit=191/200 at 9s/task, D''=9.

Seed-42 result:
- D >= 14: time wall was hiding solutions in 1e/1f/1g. All three time-confounded.
  Re-read width (Stage 1e/1f/1g) at saturated budget. Stage 2 (proposer) re-opens.
- D <= 12 AND time_limit_hit_count ~ 0: reach ceiling GENUINE at depth-2/B=30000.
  Proceed to Stage 1i (depth-3).
- D <= 12 BUT time_limit_hit_count still high at 120s: 1370-leaf evaluation too slow to
  saturate. Re-run on 330-leaf (Stage 1e grammar) OR raise time further. Do NOT
  conclude reach ceiling until budget binds.
- D = 13: ambiguous, run seed-1.

Two-seed table:
- Both <= 12, time_limit_hit ~ 0: GENUINE ceiling -> Stage 1i depth-3.
- One >= 14: LIFT -> re-read 1e/1f/1g at saturated budget.
- Both <= 12, time_limit_hit still high: evaluation too slow -> adjust and re-run.

---

-- Leo (directive, mail #12064), Eli (pre-registration + harness), 2026-05-30
