# Research State — Live Document
*Updated 2026-05-29. All content before this date is archived.*

---

## Active Direction: Semantic-Capability-Memory (the-search#3)

**Question:** Can a substrate accumulate a semantic-program capability memory — beyond a seeded primitive algebra — such that solved AND failed tasks generate reusable abstract operators that improve future program synthesis on structurally novel tasks?

**Constitutional framing:** R0-R6. See `docs/direction-semantic-capability-memory.md` for full spec.

**Engineering owner:** Eli. **Research design:** Leo.

---

## Constitutional Amendment (2026-05-29)

**Decision (a): RSI is second-order.** See `docs/constitution-amendment-2026-05-29-rsi-is-second-order.md` (commit b05f768c on wordingone/the-search).

RSI = recursive self-IMPROVEMENT; improvement presupposes a baseline. The prior framing conflated RSI with origin-of-cognition and banned (R1/R2) the only means of acquiring baseline capability. K1 is the empirical instance of that error made concrete: the self-modification loop on the seed basis is UNFED (no solves -> no signal), not inert.

**Consequence:** System architecture is now [frozen minimal capability core] + [self-modification meta-layer]. R6 gate: deletion of meta-layer must degrade novel-task performance.

---

## Experiment Log

### E1 — R2-fusion feasibility (DONE -> REVISE)
- Family: crop+up2 (10 tasks, single compound)
- Arms: OFF / PASSIVE / INTENTIONAL (3-arm)
- Result: PASSIVE achieves 4.5x speedup (task1 exp=9, tasks 2-10 exp=2 via absorbed crop+up2). INTENTIONAL == PASSIVE.
- Verdict: REVISE. Memoization works; reprioritization non-load-bearing. Degenerate family (no composition variation).
- Artifacts: `incoming/arc-agi1-visa/03_R4_transfer_wall/E1_fusion.py`, `E1_result.json`
- Commit: b8e949f2

### E1b — Fair test, composition-varying family (DONE -> STRONGER REVISE)
- Family: 8 rot tasks + 2 novel fh tasks (composition varies)
- Arms: v1-OFF/PASSIVE/INTENTIONAL + v2-OFF/PASSIVE/INTENTIONAL (6-arm)
- Result: v2-INTENTIONAL LOSES on novel task 9 (fh): exp=6 vs PASSIVE=4. Deletion confirms compound crop+rot was actively harmful (+2 expansions). v2 wins on rot tasks 4-8 via compound-level accumulation (not component-level priority).
- Verdict: STRONGER REVISE. INTENTIONAL does not beat PASSIVE on novel-composition tasks. Success-weighted heap priority over a fixed primitive algebra KILLED for novel-composition speedup.
- Mechanism kill K1: logged (1/3 direction-kills for the-search#3).
- Artifacts: `incoming/arc-agi1-visa/03_R4_transfer_wall/E1b_fusion.py`, `E1b_result.json`
- Commit: b8e949f2

### K1 — Seed-basis novelty kill probe (DONE -> K1 FIRES)
- Task set: 395 locked out-of-closure tasks (D2_1_final_out.json, depth<=5, budget=500 probe)
- Regimes: (i) h-only, (ii) primitive-success-weighted
- Results:
  - Depth-1 unique h-argmin: 116/395 (29.4%). Tied: 279/395 (70.6%).
  - pops_differ under max-hypothetical scenario: 0/395 (uniform success=10 -- see note below)
  - Solve rate h-only: 5/395 (5 are false-positives from single-pair probe; D2 required all pairs)
  - Solve rate prim-success: 5/395. Expansion delta: 0.
  - Deletion: no-op (accumulated prim-success from 395 = 0; no solve signal generated)
- Verdict: K1 FIRES. Capability-memory cannot improve NOVEL synthesis under seed basis as a structural fact. The 395 require primitives absent from the 12-op basis (vocabulary gap, not search-order gap). Mechanism kill K1 is structural.
- Note: pops_differ=0 is partly an artifact of uniform-success hot scenario. Differential success WOULD change depth-1 pop on 70.6% of tied tasks -- but it never accumulates because solve rate=0 on out-of-closure tasks. The structural kill stands.
- Artifacts: `incoming/arc-agi1-visa/03_R4_transfer_wall/K1_probe.py`, `K1_result.json`
- Commit: bac727fd

---

### E2.1 — Combinatorial meta-layer gate (DONE -> R6_FIRES)
- Architecture: Option-2 frozen core (logistic prior, ~1MB, trained on synthetic triples biased 70% toward ACCUM_FIRST/ACCUM_SECOND) + LGG depth-2 meta-layer, R2-fused.
- Accumulation family (9 programs): ACCUM_FIRST={crop,fh,tr} × ACCUM_SECOND={rot,up2,mir_h}. 90 tasks.
- Held-out family (6 programs): HELD_FIRST={fv,dup_h} × ACCUM_SECOND. 30 tasks.
- Test: does accumulated LGG signal improve held-out mean_exp vs CORE_ONLY?
- Results (biased core):
  - CORE_ONLY held-out: 30/30 solved, mean_exp=7.63
  - CORE_META held-out: 30/30 solved, mean_exp=7.63
  - Deletion delta: +0.00 (meta non-load-bearing)
- Uniform core control (unbiased 1/12 prior, meta preloaded with accum signal):
  - CORE_ONLY: mean_exp=7.57
  - CORE_META: mean_exp=7.70 (WORSE, delta=-0.13)
  - Root cause: accum stream produced fh=10 in depth2_success (alternative solutions to accum tasks where fh is also valid). For held-out tasks needing rot at depth-2, meta boosts fh first → extra wasted expansions. Same mechanism as E1b: accumulated wrong-op signal misdirects novel-composition search.
- Structural finding: LGG accumulates from WHATEVER program solved the task, not the ground-truth program. Task families with multiple valid solutions → noisy signal → meta layer is systematically misleading on novel compositions.
- Verdict: R6_FIRES. Two distinct failure modes confirm: (1) biased core makes meta redundant, (2) unbiased core makes meta actively harmful.
- Artifacts: `incoming/arc-agi1-visa/03_R4_transfer_wall/E2_1_experiment.py`, `E2_1_result.json`
- Commit: dad816fe

---

## Current Direction: E2.2 (PENDING — Leo response)

E2.1 R6 fires on combinatorial test (synthetic in-closure). K2 logged (2/3 mechanism kills).

Leo's directive (from #11513) held E2.2 pending E2.1 result.

**E2.2 scope (per Leo #11513):** structural-novelty analysis — does accumulated abstraction improve performance on the OUT-OF-CLOSURE 395? Core design still open; E2.1 biased-core result suggests core pre-encoding the right ops makes meta redundant. The genuine test requires a core that does NOT already encode the solution domain.

---

## Mechanism Kill Log

| Kill | Mechanism | Experiment | Status |
|------|-----------|------------|--------|
| K1 | Success-weighted heap priority over fixed primitive algebra | E1b + K1 probe | FIRED 2026-05-29 |
| K2 | LGG depth-2 meta-layer non-load-bearing on combinatorial held-out | E2.1 | FIRED 2026-05-29 |

3 mechanism-kills = direction dead. At 2.

---

## Context: ARC-AGI-1 Visa (D2 partition)

- Eval set: 400 tasks. Seed-basis closure: ~5/400 (1.25%). Out-of-closure: 395.
- D2_1_final_out.json: 395 locked IDs, depth<=5, budget=20000, hash fingerprint.
- 5b6cbef5: canonical vocabulary-gap exemplar (fractal gene absent from 12-op basis).
- Seed basis (BASIS in SUBSTRATE.py): id, fh, fv, tr, rot, crop, dup_h, dup_v, mir_h, mir_v, up2, down2.

---

## Pre-metamorphosis State (archived)

The 2026-03-31 state (ARC-AGI-3 dolphin explorer, 1395 neural experiments) is archived. Navigation > learning was the final finding of that era: every K improvement came from better navigation structure, not learning substrates. The-search pivoted to ARC-AGI-1 symbolic synthesis in 2026-04 (metamorphosis). See `archive_sessions.md` and `archive_research_era.md` in memory system.
