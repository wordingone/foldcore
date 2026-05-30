# the-search -- Stage 1j: Budget Scale at Depth-3 -- PRE-REGISTERED

**Status: PRE-REGISTERED — harness built; PR pending Leo layer-1 diff-review.**

Gates resolved:
- [x] Leo pre-stage GO (mail #12090): "1j is the both-branch successor"
- [ ] Stage 1i canonical seed-42 gated (prerequisite before 1j canonical)
- [ ] Leo layer-1 PR diff-review
- [ ] TEMP-smoke verified by Leo
- [ ] Canonical seed-42 GO from Leo after smoke verify

Design decisions LOCKED at pre-registration (Leo, mail #12090):
- Q1: Grammar leaf SET UNCHANGED. MAX_DEPTH UNCHANGED (3). ONE variable = BUDGET_B only.
- Q2: BUDGET_B 30000 → 300000 (~10x). ~30s/task at 10K evals/s, under 120s wall.
- Q3: Budget NOT encoded in compute_space_hash() (descriptor: n_leaves + max_depth only).
  Gate asserts space_hash == Stage 1i hash (4978b6739bf55beb). Inverts 1i's != back to ==.
- Q4: Disambiguate 1i null: if 1h's 7 IDs reappear (+more), 1i null was pure sparsity.
  If reach still plateaus at 300K, THAT is the clean depth-saturated signal.

---

## Context: why 1j in both branches

Stage 1i used PURE 4-tuples at B=30000, sampling 1370x sparser than 1h's 3-tuples.
PRIM_ID padding (leaf index 4 = identity) means 1i's depth-3 space strictly contains
1h's depth-2 solutions: any 3-tuple (a,b,c) is expressible as 4-tuple (a,b,c,id_idx).

BUT: B=30000 samples 1370^4 at coverage ~8.5e-9. Even known D(1h)=7 solutions are
unlikely to be re-found by chance. A D(1i) <= 7 null is sparsity-confounded, not depth-neutral.

1j raises B to 300000 to provide a budget-unconfounded signal:
- 1j LIFT (>7 new task IDs): budget was the constraint; depth-3 has more reach.
- 1j NULL (same or subset of 1h's 7): plateau holds at 300K -> grammar-EXPRESSIVITY ceiling.
Self-compilation (#9) waits for 1j to show a budget-unconfounded plateau.

---

## Stage summary table

| Stage | Arm | Grammar | Depth | Budget | Time/task | Seed-42 | Seed-1 |
|-------|-----|---------|-------|--------|-----------|---------|--------|
| 1h | D (time-sat.) | 1370-leaf | 2 | 30K (enforced) | 120s | 7 | -- (FLAT) |
| 1i | D (depth-3) | 1370-leaf | 3 | 30K (enforced) | 120s | TBD | TBD |
| 1j | D (B-scale) | 1370-leaf | 3 | **300K** | 120s | TBD | TBD |

ONE variable between 1i and 1j: BUDGET_B (30000 → 300000).

---

## Protocol

- Grammar leaf SET: same 1370-leaf Stage 1g/1h/1i space
- MAX_DEPTH: 3 (unchanged)
- BUDGET_B: 300000 (was 30000 -- ONE variable)
- Per-task time limit: 120s (budget exits at ~30s at 10K evals/s, well under wall)
- Arm seed: 0 (held constant)
- Held split: same 200 tasks from Stage 1d
- PREV_SPACE_HASH: "4978b6739bf55beb" (Stage 1i depth-3 hash)

---

## Fail-closed gate (PREV_SPACE_HASH = "4978b6739bf55beb")

Violations checked at artifact-write (abort-without-write on any):
1. n_leaves != builder ACTUAL (still 1370)
2. n_leaves_asserted != True
3. held_task_ids count != 200
4. held_task_ids set mismatch vs stage1d
5. candidate_eval_budget != 300000 (ONE-variable integrity: was 30000 in Stage 1i)
6. candidate_eval_budget_asserted != True
7. max_depth != 3 (must be unchanged from 1i)
8. space_hash != PREV_SPACE_HASH (budget NOT in hash -> space MUST match Stage 1i)
9. prev_space_hash != PREV_SPACE_HASH
10. max(per-task n_evals) > BUDGET_B (behavioral enforcement)
11. per-task: unsolved AND NOT time_limit_hit AND n_evals < 300000 (premature exit)

Note on gate #5: injection test uses BUDGET_B=30000 (the Stage 1i value) as the "wrong" case.

---

## Decision tree (written BEFORE running)

Verdict criterion: set-diff vs 1h_solved_ids (authoritative per Leo pre-reg, mail #12090).

Seed-42 result:
- (1j_solved_ids - 1h_solved_ids) non-empty: depth-3 at 300K finds tasks depth-2 couldn't.
  Budget was the constraint in 1i; depth-3 is viable with more B. LIFT.
  Next: depth-4 OR further B-scaling at depth-3.
- 1j_solved_ids subset of or equal to 1h_solved_ids: depth-3 adds no new task IDs.
  Grammar-EXPRESSIVITY ceiling, not depth or budget. FLAT.
  Next: grammar enrichment OR self-compilation track (#9) -- budget-unconfounded.
- time_limit_hit_count high: 10K evals/s assumption wrong at 300K -- tasks slower.
  Raise time wall OR profile per-candidate cost before trusting verdict.
- D(1j) >> 7 with all new IDs: strong depth signal. Proceed to depth-4.

---

-- Leo (pre-stage, mail #12090), Eli (pre-registration + harness), 2026-05-30
