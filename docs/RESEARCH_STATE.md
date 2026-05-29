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

## Current Direction: E2' (PENDING — core design open)

E2-on-seed-basis (LGG over the 12-op vocabulary) is SUPERSEDED by constitutional decision (a). It remains a valid combinatorial-transfer test but cannot address structural novelty.

**E2' architecture:**
```
System = [frozen minimal capability core: program-proposer for grid transforms]
       + [self-modification meta-layer: LGG-accumulated structured abstractions,
          R2-fused, reweighting the core's proposals]
```

Tested on the locked 395 (genuine test: can accumulated abstractions improve novel solve-rate when there IS a baseline proposer?).

**Pre-registered R6 kill:** deletion of meta-layer must degrade novel solve-rate. If not -> decorative -> (a) collapsed -> direction kill.

**Open: minimal capability core design.** Three options under evaluation:
1. Small frozen program-proposer model (purpose-trained on synthetic ARC data)
2. Learned grid-transform proposal distribution (shallow MLP or frequency table, no external model)
3. Existing local models (Qwen/Gemma via llama-server at :9876) as frozen proposers

Core choice gates the whole E2' design. Engineering read in progress.

---

## Mechanism Kill Log

| Kill | Mechanism | Experiment | Status |
|------|-----------|------------|--------|
| K1 | Success-weighted heap priority over fixed primitive algebra | E1b + K1 probe | FIRED 2026-05-29 |

3 mechanism-kills = direction dead. At 1.

---

## Context: ARC-AGI-1 Visa (D2 partition)

- Eval set: 400 tasks. Seed-basis closure: ~5/400 (1.25%). Out-of-closure: 395.
- D2_1_final_out.json: 395 locked IDs, depth<=5, budget=20000, hash fingerprint.
- 5b6cbef5: canonical vocabulary-gap exemplar (fractal gene absent from 12-op basis).
- Seed basis (BASIS in SUBSTRATE.py): id, fh, fv, tr, rot, crop, dup_h, dup_v, mir_h, mir_v, up2, down2.

---

## Pre-metamorphosis State (archived)

The 2026-03-31 state (ARC-AGI-3 dolphin explorer, 1395 neural experiments) is archived. Navigation > learning was the final finding of that era: every K improvement came from better navigation structure, not learning substrates. The-search pivoted to ARC-AGI-1 symbolic synthesis in 2026-04 (metamorphosis). See `archive_sessions.md` and `archive_research_era.md` in memory system.
