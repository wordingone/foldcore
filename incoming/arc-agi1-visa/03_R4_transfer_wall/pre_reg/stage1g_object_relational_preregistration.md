# the-search — Stage 1g: Object-Relational Translate-to-Anchor — PRE-REGISTERED

**Status: PRE-REGISTERED — all gates resolved; canonical run authorized.**

Gates resolved:
- [x] Stage 1f FLAT (D'=9 seed-42): object-local geometric expansion did NOT raise reach ceiling. Leo GATED (mail #12058). Recorded as verified negative.
- [x] Leo GO for pre-registration (mail #12058): "pre-register the object-relational translate-to-anchor spec now — same shape as 1f"
- [x] Stage 1f seed-1 final result: D'=7 ≤12 CONFIRMED — FLAT both seeds (seed-42: D'=9, seed-1: D'=7). GO canonical 1g.
- [x] Harness build + TEMP-smoke verified by Leo (mail #12062 — gate fires on all 7 injected violations; builder ACTUAL n_leaves=1370; space_hash=0890317fe99bc9f1 != 13a00a0ff95dd026)

Design decisions LOCKED at pre-registration (Leo, mail #12060):
- Q1: 4-mode adjacency ONLY (adjacent_left/right/up/down). No edge-align/center/corner.
- Q2: anchor = LARGEST connected component of anchor_pred-selected objects. Tiebreak: cell-count desc -> min row -> min col. NOT union bbox.

Flow: seed-42 canonical first (diagnostic), then seed-1 (confirmatory) -> report D'' table -> Leo gates result (R1-R6 + integer-level harness check).

---

## Leo gate decisions — LOCKED at pre-registration (2026-05-30)

Pre-registration APPROVED (mail #12060). R1-R6 clean (pure reach measurement, no learning/reward/optimizer/prior); fail-closed harness mirrors the 1f shape that held (PREV_SPACE_HASH=13a00a0ff95dd026 correct); ONE operator family; n_leaves builder-ACTUAL not hardcoded; decision-tree thresholds consistent with 1e/1f.

**Q1 — alignment-mode breadth: 4-mode adjacency ONLY.** Edge-align / corner-match / center-align are NOT in this family — bundling them makes 1g multi-variable and destroys clean D'' attribution. Edge-align is the NEXT one-variable step if 1g is FLAT. The orthogonal-axis choice baked into each mode (align leading edge) stays FIXED this round.

**Q2 — multi-component anchor: LARGEST connected component, NOT union bbox.** Anchor = the single largest CC among anchor_pred-selected cells; deterministic tiebreak = cell-count desc -> min row -> min col. Rationale: "move A adjacent to B" targets a real object; a union bbox of scattered anchor cells yields adjacency to phantom empty space, adds dead leaves, and gives MAP_RELATE a weaker/noisier shot. Largest-CC harmonizes with the existing single-object predicates and is the FAIREST instantiation — so FLAT then is a strong NO on the relational axis, not an artifact of a weak anchor definition.

**Minor (non-blocking):** src_pred matching multiple components moves each to the anchor's same edge -> they can overlap. Acceptable for the diagnostic; revisit ONLY if 1g LIFTS.

---

## Motivation: Stage 1f FLAT on object-local; next axis is object-RELATIONAL

| Stage | Arm | Seed-42 | Seed-1 |
|-------|-----|---------|--------|
| 1e | D (330-leaf, random time-limited) | 11 | 8 |
| 1f | D' (410-leaf, +MAP_GEOM object-local) | 9 | 7 |
| 1g | D'' (1370-leaf, +MAP_RELATE object-relational) | TBD | TBD |

Stage 1f conclusion: MAP_GEOM (per-object flip/rotate within bbox) is functional (geom_solutions: 3 seed-42, 4 seed-1) but does NOT raise the aggregate reach ceiling. D'∈{7,9} vs Stage 1e D∈{8,11} — within band. The binding constraint is NOT per-object geometric expressiveness.

---

## The expansion: ONE variable — object-relational translate-to-anchor (MAP_RELATE)

Dual-predicate design: src_pred selects objects to move, anchor_pred selects reference object.
Anchor = largest CC of anchor_pred-selected objects (Q2 locked).
4 alignment modes (Q1 locked): adjacent_down, adjacent_left, adjacent_right, adjacent_up.

Grammar:
- MAP_RELATE: src_pred x anchor_pred x mode, prune src==anchor
- Leaves: 16x16x4 - 16x4 = 960 (no further pruning at build time)
- New n_leaves: 1370 (410 Stage 1f + 960 MAP_RELATE)
- space_hash: 0890317fe99bc9f1 (verified in smoke, changed from 13a00a0ff95dd026)

Harness: B:/M/the-search/incoming/arc-agi1-visa/03_R4_transfer_wall/stage1g_object_relational.py

NOT added this round: edge-align modes, center-align, snap-and-push — next one-variable steps if FLAT.

---

## Protocol: two-seed Arm-D-analog reach lower-bound

- Pure reach measurement: random time-limited BFS, no prior, no oracle, no neural.
- Budget: B = 30000 candidates per task
- Held split: same 200 tasks from Stage 1d (assert held_task_ids_match_stage1d==true)
- Depth: 2 (holed-skeleton BFS extended with MAP_RELATE)
- Time per task: 9.0s (same as Stage 1e/1f arm-D)
- Arm seed: fixed (seed=0, same across stages)
- Two-seed protocol: seed-42 first (diagnostic), seed-1 (confirmatory). Both always. Report table.

---

## Fail-closed gate (PREV_SPACE_HASH = "13a00a0ff95dd026")

7 violations checked at artifact-write (abort-without-write on any):
1. n_leaves != builder ACTUAL
2. n_leaves_asserted != True
3. held_task_ids count != 200
4. held_task_ids set mismatch vs stage1d
5. candidate_eval_budget != 30000
6. candidate_eval_budget_asserted != True
7. space_hash == PREV_SPACE_HASH (expansion not applied)
8. candidate_set_changed != True
9. prev_space_hash != PREV_SPACE_HASH

Plus per-task: unsolved + not time_limit_hit + n_evals < 30000 -> violation.

---

## Decision tree (written BEFORE running)

Baseline ceiling from Stage 1e arm-D: D in {8 (seed-1), 11 (seed-42)}.
Stage 1f confirmed ceiling flat at D'=9 (within band).

Single-seed D'' read (seed-42 first):
- D'' <= 12: FLAT. Skip seed-1. REJECT object-relational adjacency. Next: edge-align or next axis.
- D'' >= 14: LIFT. Run seed-1. If both >=14: re-open Stage 1e probe on 1370-leaf space.
- D'' = 13: ambiguous. Run seed-1 before deciding.

Two-seed D'' table:
- Both >= 14: CONFIRMED LIFT -> re-open Stage 1e probe on Stage 1g space.
- One >= 14, one = 13: MARGINAL -> third seed before deciding.
- Both <= 12: FLAT -> reject object-relational adjacency -> next axis.
- Straddling (one >= 14, one <= 12): CHECK harness for confound before concluding.

---

— Leo (spec direction, mail #12058+#12060+#12062), Eli (pre-registration + harness), 2026-05-30
