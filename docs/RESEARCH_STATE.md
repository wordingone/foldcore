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

### E2.1 — Combinatorial meta-layer gate (DONE -> R6_FIRES; IN-COVERAGE ONLY)
- R4 classification: SELECTION-class experiment. Per Leg 2, out-of-coverage ceiling = +0.000. In-coverage gains tested here.
- Architecture: Option-2 frozen core (logistic prior, ~1MB, biased 70% toward ACCUM_FIRST/ACCUM_SECOND) + LGG depth-2 meta-layer, R2-fused.
- Coverage split: IN-COVERAGE only (all synthetic tasks expressible by the 12-op basis). Out-of-coverage measurement pending — needs E2.1b (see below).
- Accumulation family (9 programs): ACCUM_FIRST={crop,fh,tr} × ACCUM_SECOND={rot,up2,mir_h}. 90 tasks.
- Held-out family (6 programs): HELD_FIRST={fv,dup_h} × ACCUM_SECOND. 30 tasks.
- Results (biased core, in-coverage):
  - CORE_ONLY held-out: 30/30, mean_exp=7.63; CORE_META: 30/30, mean_exp=7.63; delta=0.00
- Uniform core control (in-coverage):
  - CORE_ONLY: mean_exp=7.57; CORE_META: mean_exp=7.70 (WORSE, delta=-0.13)
  - Root cause: LGG accumulated fh=10 from alternative solutions (not ground-truth program). Meta boosts wrong op at depth-2 for tasks needing rot.
- Structural finding: LGG accumulates from whatever SOLVED the task, not the ground-truth program. Underdetermined task families → noisy meta signal → misdirects novel-composition search.
- Verdict: R6_FIRES IN-COVERAGE. Per R4 Leg 2, out-of-coverage = +0.000 is PREDICTED (selection cannot expand coverage). E2.1b will confirm.
- Artifacts: `incoming/arc-agi1-visa/03_R4_transfer_wall/E2_1_experiment.py`, `E2_1_result.json`
- Commit: dad816fe

---

### E2.1b — Out-of-coverage prediction (DONE -> R4_LEG2_CONFIRMED)
- R4 classification: SELECTION-class, out-of-coverage split. Per Leg 2, prediction = +0.000.
- Result: in-coverage delta=+0.000, out-of-coverage (100/395 sampled) delta=+0.000, solve_delta=0.
- Verdict: R4 Leg 2 confirmed from capability-memory angle. LGG selection = +0.000 on both sides of the coverage line.
- Artifacts: `E2_1b_coverage_split.py`, `E2_1b_result.json`. Commit: 2b1b3d88.

---

## E2.2(b) — Trace-conditional accumulation, TERMINAL selection-axis test (PRE-REGISTERED 2026-05-29)

**Purpose (Leo #11525):** One unmeasured cell in the signal-cleanliness axis:
- E2.1 uniform core + DIRTY signal: delta=-0.13 (meta harmful — noise from alternative solutions)
- E2.2(b): **uniform core + CLEAN signal** — trace-conditional accumulation; meta is the SOLE bias source

If meta still non-load-bearing with clean signal → selection axis definitively dead (K3 = 3/3).

**Design:**
- Core: UNIFORM prior (1/12 equiprobable at d0 and d1). Meta is the only bias source.
- Accumulation: TRACE-CONDITIONAL — only accumulate from sequences that MATCH the ground-truth generating program. If solver finds an alternative valid sequence, do NOT accumulate.
- Task family: same split as E2.1 (ACCUM_PROGRAMS × N_ACCUM_PER, HELD_PROGRAMS × N_HELD_PER).

**Pre-registered kill (K3):**
CORE_META deletion-delta ≤ 0 with trace-conditional clean signal → K3 (3/3) → selection axis DEFINITIVELY dead → hard pivot to E2.2' (coverage-expansion core).

**Pre-registered positive:**
Deletion-delta > 0 → selection works with clean signal → carry clean-signal meta-layer into E2.2' core design.

**Hard-pivot trigger:** K3 fires → next = E2.2' (expressibility-expanding core = C6 operationalized).

---

### E2.2(b) — Trace-conditional accumulation result (DONE -> K3_FIRES)
- Design: UNIFORM core (1/12) + trace-conditional accumulation (ground-truth match only).
- Accumulation: 26/89 traces accepted (ground-truth match); 63 rejected (alternative solutions).
  - depth2_success: {mir_h:19, up2:6, rot:1}. fh=0 (correctly rejected — E2.1 noise confirmed).
- Held-out: CORE_ONLY mean_exp=7.57, CORE_META mean_exp=7.57. delta=0.
- R6 deletion delta: +0.000. Meta non-load-bearing even with clean signal.
- Root cause: LGG depth-2 accumulation learns the MARGINAL P(d2) from accum stream. Novel synthesis
  requires the CONDITIONAL P(d2|d1, input). Marginal boost is informationally insufficient —
  it cannot know which d2 op a specific novel task needs. This is structural, not noise-related.
- Verdict: K3 FIRES (3/3). Selection axis DEFINITIVELY dead. Hard pivot triggered.
- Artifacts: `E2_2b_trace_conditional.py`, `E2_2b_result.json`. Commit: (pending).

---

## HARD PIVOT — E2.2' (Coverage-Expansion Core)

Selection axis exhausted. Three kills confirm the R4 decomposition from the capability-memory side:
- K1: selection signal loop UNFED out-of-coverage (vocabulary gap, structural)
- K2: selection redundant in-coverage (core pre-bias makes meta noise)
- K3: selection still zero even with clean signal (marginal != conditional, structural)

**Next direction: E2.2' — expressibility-expanding core (C6 operationalized)**
- Architecture: [frozen core with vocabulary BEYOND the 12-op seed] + [meta-layer to be evaluated]
- Core candidates: (1) purpose-trained program-proposer; (3) local LLM at :9876 (Qwen/Gemma)
- Test: does the expanded-vocabulary core beat the 0.7% oracle ceiling on the 395?
- R6 gate: does meta-layer expand COVERAGE beyond frozen expanded core alone?

Leo to specify core design. K3 fires = selection-axis chapter closed.

---

## Current Direction: HARD PIVOT — awaiting E2.2' core design from Leo

---

## Mechanism Kill Log

| Kill | Mechanism | Experiment | Status |
|------|-----------|------------|--------|
| K1 | Success-weighted heap priority over fixed primitive algebra | E1b + K1 probe | FIRED 2026-05-29 |
| K2 | LGG depth-2 meta-layer non-load-bearing on combinatorial held-out | E2.1 | FIRED 2026-05-29 |
| K3 | Trace-conditional LGG meta-layer non-load-bearing with clean signal | E2.2(b) | FIRED 2026-05-29 |

3 mechanism-kills = direction dead on SELECTION AXIS. Hard pivot to E2.2' (coverage-expansion).

---

## R4 Decomposition — Coverage-vs-Selection Spine (formalized 98e10f17; fused here 2026-05-29)

The current direction IS the R4 thread continued from the capability-memory/RSI side. The transfer wall is COVERAGE, not SELECTION.

**Three legs (all confirmed, formally closed in 98e10f17):**

| Leg | Finding | Status |
|-----|---------|--------|
| Leg 1 | 119 eval tasks fit a rule on train pairs; 1 transfers to test pair (transfer ≈ 0.01). Coverage wall, not selection. | VERIFIED |
| Leg 2 | Learned recognition prior (17→32→19 MLP) yields warm−cold = +0.000 transfer. Better selector does nothing; bottleneck is upstream of selection. | CONFIRMED (T1.py / T1_numpy.py in 98e10f17) |
| Leg 3 | Oracle ceiling ~0.7% with the available seed vocabulary. Even perfect selection is coverage-bounded. | CONFIRMED (same script output) |

**C6 (the mechanism):** Surpassing the designer requires a learned generative program-prior at scale — NOT a better selector over the fixed algebra.

---

**Fusion map — current experiments ARE the R4 decomposition from the capability-memory side:**

| Experiment | R4 mapping |
|------------|-----------|
| K1 (vocabulary gap on 395) | Confirms Leg 1+3 from the capability-memory angle: 0/395 solve → no signal accumulates → R6-deletable on novelty as structural fact. |
| E1b (INTENTIONAL == PASSIVE on novel task) | Confirms Leg 2: INTENTIONAL = a better selector over fixed algebra. delta=+0 on novel composition. Same as warm−cold = +0.000. |
| E2.1 (LGG meta-layer over 12-op core) | A SELECTION-class experiment. Per Leg 2, out-of-coverage ceiling = +0.000. In-coverage (combinatorial) gains are possible but must be measured SEPARATELY. R6 fires: meta-layer non-load-bearing even in-coverage. |
| E2.2 (expressibility-expanding core) | The COVERAGE-expansion experiment. C6 operationalized: learned generative core that proposes programs outside the seed algebra. Must beat the 0.7% oracle ceiling. R6 gate becomes: does meta-layer expand coverage beyond frozen core alone? |
| (a) constitutional decision | IS the R4 forward-pointer. VISA §7 last line: "R4 points directly at semantic-program capability memory BEYOND the seed." (a) = C6 = the prescribed next move. |

**Instrumentation requirement (per Leo #11519):** E2.1 and any LGG experiment MUST report in-coverage and out-of-coverage SEPARATELY. If out-of-coverage ever shows gain from selection alone → contradicts R4 Leg 2 → flag immediately.

---

## Context: ARC-AGI-1 Visa (D2 partition)

- Eval set: 400 tasks. Seed-basis closure: ~5/400 (1.25%). Out-of-closure: 395.
- D2_1_final_out.json: 395 locked IDs, depth<=5, budget=20000, hash fingerprint.
- 5b6cbef5: canonical vocabulary-gap exemplar (fractal gene absent from 12-op basis).
- Seed basis (BASIS in SUBSTRATE.py): id, fh, fv, tr, rot, crop, dup_h, dup_v, mir_h, mir_v, up2, down2.

---

## Pre-metamorphosis State (archived)

The 2026-03-31 state (ARC-AGI-3 dolphin explorer, 1395 neural experiments) is archived. Navigation > learning was the final finding of that era: every K improvement came from better navigation structure, not learning substrates. The-search pivoted to ARC-AGI-1 symbolic synthesis in 2026-04 (metamorphosis). See `archive_sessions.md` and `archive_research_era.md` in memory system.
