# the-search -- Stage 1k: Time Wall Raise at Depth-3/B=300K -- PRE-REGISTERED

**Status: PRE-REGISTERED — harness built; PR pending Leo layer-1 diff-review.**

Gates resolved:
- [x] Stage 1j PASS / LIFT (D=9, +2 vs 1h): Leo integer-gate (mail #12105, issue #9 closed)
- [x] Leo GO for Stage 1k (mail #12105): "raise time wall 120s→600s; ONE variable"
- [x] time_limit_hit flag: 23/200 at 1j confirmed wall is interfering (Leo flagged)
- [ ] Leo layer-1 PR diff-review (harness lands via PR — Leo verifies ONE variable)
- [ ] TEMP-smoke verified by Leo (gate fires on each injected violation)
- [ ] Canonical seed-42 GO from Leo after smoke verify

Design decisions LOCKED at pre-registration (Leo, mail #12105, issue #10):
- Q1: Grammar leaf SET UNCHANGED. MAX_DEPTH UNCHANGED (3). BUDGET_B UNCHANGED (300000). ONE variable = TIME_PER_TASK_D only.
- Q2: TIME_PER_TASK_D 120s → 600s (5x). At B=300K and ~10K evals/s, budget exhausts in ~30s, well under 600s wall. Wall is safety headroom for slow tasks.
- Q3: Time wall NOT encoded in compute_space_hash() (descriptor: n_leaves + max_depth only). Gate asserts space_hash == Stage 1j hash (4978b6739bf55beb). Same inversion as 1j: ==.
- Q4: 1j had time_limit_hit=23/200 at 120s. At 600s, budget-bound tasks (~30s each) should never hit the wall. Only genuinely slow tasks (>2ms/eval at B=300K) hit 600s. Expected time_limit_hit ≈ 0–5.

---

## Context: why 1k in both branches of 1j

Stage 1j (D=9, LIFT vs 1h's D=7) had time_limit_hit=23/200. At BUDGET_B=300000 and ~10K
evals/s, expected per-task time is ~30s — well under the 120s wall. But 23 tasks hit 120s
before exhausting 300K evals. Those 23 tasks ran for ~120s without seeing their full budget.

The LIFT holds: both new solves (1j_solved − 1h_solved = {3618c87e, 7468f01a}) landed
within budget AND time. But the plateau question — is D=9 the faithful-budget ceiling? — is
unclean while 23/200 tasks are budget-truncated by the time wall.

Stage 1k raises the wall to 600s so every task sees its full B=300000. If D rises: those
23 were hiding solves. If D stays at 9: 9 is the faithful-budget ceiling at depth-3/B=300K.

---

## Stage summary table

| Stage | Arm | Grammar | Depth | Budget | Time/task | Seed-42 | Seed-1 |
|-------|-----|---------|-------|--------|-----------|---------|--------|
| 1h | D (time-sat.) | 1370-leaf | 2 | 30K | 120s | 7 | -- (FLAT) |
| 1i | D (depth-3) | 1370-leaf | 3 | 30K | 120s | 8 (LIFT) | -- |
| 1j | D (B-scale) | 1370-leaf | 3 | 300K | 120s | 9 (LIFT) | -- |
| 1k | D (time-wall) | 1370-leaf | 3 | 300K | **600s** | TBD | TBD |

ONE variable between 1j and 1k: TIME_PER_TASK_D (120s → 600s).

---

## Protocol

- Grammar leaf SET: same 1370-leaf Stage 1g/1h/1i/1j space
- MAX_DEPTH: 3 (unchanged)
- BUDGET_B: 300000 (unchanged)
- TIME_PER_TASK_D: 600.0 (was 120.0 -- ONE variable)
- Arm seed: 0 (held constant)
- Held split: same 200 tasks from Stage 1d
- PREV_SPACE_HASH: "4978b6739bf55beb" (Stage 1j hash; time wall not in descriptor)
- PREV_TIME_LIMIT_HIT: 23 (Stage 1j count — advisory gate threshold)

---

## Fail-closed gate (PREV_SPACE_HASH = "4978b6739bf55beb")

Violations checked at artifact-write (abort-without-write on hard violations 1–12):
1. n_leaves != builder ACTUAL (still 1370)
2. n_leaves_asserted != True
3. held_task_ids count != 200
4. held_task_ids set mismatch vs stage1d
5. candidate_eval_budget != 300000 (unchanged from 1j)
6. candidate_eval_budget_asserted != True
7. max_depth != 3 (must be unchanged)
8. **NEW: time_limit_s != 600.0** (ONE-variable integrity: was 120.0 in Stage 1j)
9. space_hash != PREV_SPACE_HASH (time wall NOT in hash → space unchanged → must match)
10. prev_space_hash != PREV_SPACE_HASH
11. max(per-task n_evals) > BUDGET_B (behavioral enforcement)
12. per-task: unsolved AND NOT time_limit_hit AND n_evals < 300000 (premature exit)

Advisory (written to artifact, does NOT abort):
13. time_limit_hit_count >= 10 → "600s wall still insufficient; raise again before trusting verdict"

Gate injection tests MUST include:
- Test 8 (NEW): time_limit_s wrong (inject 120.0 — Stage 1j value)
- Test (existing): budget wrong (inject 30000 — Stage 1i value)
- Test (existing): max_depth wrong (inject 2 — Stage 1h value)
- Test (existing): max_evals over budget (inject n_evals=300001)
- Test (existing): space_hash wrong (inject 0890317fe99bc9f1 — Stage 1h depth-2 hash)

---

## Decision tree (written BEFORE running)

Stage 1j baseline: D=9/200, time_limit_hit=23/200, budget_exhausted=168/200.
Verdict criterion: set-diff vs 1j_solved_ids (authoritative per Leo pre-reg, mail #12105).

Seed-42 result:
- (1k_solved_ids − 1j_solved_ids) non-empty: time wall was masking solves in 1j.
  B=300K has more reach than D=9 suggested. Depth-3/B=300K is not saturated.
  Next: continue budget-scaling (1l?) OR depth-4.
- 1k_solved_ids == 1j_solved_ids (D=9, same set): 9 is the faithful-budget ceiling
  at depth-3/B=300K. Plateau is budget-unconfounded. Grammar-EXPRESSIVITY ceiling.
  Next: depth-4 (ONE variable: MAX_DEPTH 3→4) OR grammar enrichment.
- time_limit_hit_count still >= 10 at 600s: 600s insufficient for some tasks (>2ms/eval).
  Raise wall again before trusting verdict (1l: TIME_PER_TASK_D 600→3000?).
  This does NOT invalidate D(1k) solves — it means plateau may still be confounded.

Wall-clock estimate: 200 tasks × ~30s/task (budget-limited) = ~6000s ≈ 100 min expected.
If tasks that hit 120s wall at 1j exhaust budget within 600s: same ~160 min as 1j.
Worst case (unlikely): 200 × 600s = 120000s = 2000 min.

---

-- Leo (directive, mail #12105 + issue #10), Eli (pre-registration + harness), 2026-05-31
