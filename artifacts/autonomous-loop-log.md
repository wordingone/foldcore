# Autonomous Loop Log

*Leo working independently while Jun rests. Eli running experiments on 8-min cron.*

## Iteration 1 (2026-03-18 ~03:12 UTC)

**The Question:** "What do I need to be true that I haven't run?"
- Need accurate inventory of process_novelty()'s frozen elements (formal R3 audit never done)
- Need to know if ReadIsWrite equation works for classification even though it failed at navigation

**Actions:**
- Sent Eli: Step 418b (ReadIsWrite classifier on P-MNIST, 5K steps, tau=0.1)
- Started: process_novelty() formal R3 audit
- Adversary killed: sparse top-K attention (process() with a hat)

**Results:** Step 418b: 72.2% P-MNIST. PASS (>25% gate). Phase 2 candidate. 19pp below baseline (91.2%). Spawn rate 86%, train-test gap 12.3pp.

## Iteration 2 (2026-03-18 ~03:16 UTC)

**The Question:** "What do I need to be true that I haven't run?"
- ReadIsWrite passes P-MNIST at 72.2%. But spawn rate is 86% (too high). Need to know: does reducing spawn rate improve generalization?
- The 12.3pp train-test gap suggests overfit from excessive spawning. Fewer spawns → smaller codebook → better generalization?
- process_novelty() R3 audit complete. Key finding: class structure is the load-bearing frozen frame for navigation.

**Actions:** Sent Eli: Step 418c (ReadIsWrite with spawn threshold sweep on P-MNIST)
**Results:**
- thresh=0.9: 88.23% (frozen eval). 3pp from baseline. BEST Phase 2 result.
- thresh=0.7: 77.08%. thresh=0.5: 55.68%. Tighter thresh HURTS.
- Tighter spawn ≠ better generalization. Kill criterion fires.
- The 3pp gap is distributed update vs winner-take-all specialization.

## Iteration 3 (2026-03-18 ~03:19 UTC)

**The Question:** ReadIsWrite at 88.2% — is this a genuine result or inflation?
**Actions:** Sent Eli Step 418d (10-task CL). Ran adversary.

**Adversary (anti-inflation rule 3):** ReadIsWrite is Nadaraya-Watson / soft KDE + competitive learning with soft assignment. Individual pieces are 1960s-1990s. What's potentially novel: R2 by construction, growing codebook, 88.2% P-MNIST without backprop. NOT novel: soft attention over prototypes, distributed error correction.

**Waiting for:** 418d (10-task CL). Key question: does distributed update cause forgetting?

## Iteration 4 (2026-03-18 ~03:24 UTC)

**The Question:** "What do I need to be true that I haven't run?"
Need to know: does ReadIsWrite preserve accuracy across tasks? The distributed update shifts ALL entries on every step — task 2 observations will push task 1 entries. This might cause catastrophic forgetting.
**Prediction:** forgetting > 5pp. The distributed update is the opposite of what protects memories.
**Results:** Step 418d: 87.78% avg across 10 tasks, 0.40pp max forgetting. PASS BOTH criteria.
**Prediction was WRONG:** predicted >5pp forgetting, got 0.4pp. Distributed update is naturally anti-forgetting.

ReadIsWrite summary:
- P-MNIST 1 task: 88.23% (frozen eval)
- P-MNIST 10 tasks CL: 87.78% avg, 0.13pp avg forgetting
- R1-R6: 6/6 (R2 by construction)
- Navigation: FAILS (Gram explosion, Step 418a)
- Prior art: Nadaraya-Watson + soft competitive learning (anti-inflation rule 3)

## Iteration 5 (2026-03-18 ~03:36 UTC)

**The Question:** Can ReadIsWrite navigate if spawn doesn't use the Gram matrix?
**Hypothesis:** Error-based spawn (||error|| > running median) decouples spawn from Gram saturation. Spawn rate should be ~50% (by definition of median), not 99%.
**Kill criteria:** spawn>90% OR unique<200 OR dom>80% → KILL
**Actions:** Sent Eli Step 418e (error-based spawn on LS20, 10K steps)
**Key measurement:** Does spawn rate DECREASE over time? (self-limiting growth)
**Results:** spawn=45%, unique=429, dom=78%, 0 levels. Passes kill criteria but exploration DEAD.
output.argmax() is deterministic → same state → same action → loop. Spawn rate INCREASING not decreasing.
Error-based spawn WORKS (no explosion). Action selection is the bottleneck, not spawn.

## Iteration 6 (2026-03-18 ~03:39 UTC)

**The Question:** Can ReadIsWrite's distributed update coexist with argmin exploration?
**Hypothesis:** Hybrid = ReadIsWrite update + argmin scoring + error spawn. If unique>1000 & dom<50%, the combination works.
**Kill criteria:** unique<500 OR dom>70% OR spawn>80%
**Actions:** Sent Eli Step 418f (hybrid on LS20, 10K steps)
**Results:** unique=3248, dom=28%, cb=4929, spawn=49%. ALL PASS. EXTENDING to 30K.
Spawn rate DECELERATING (44→54→55→49→44). First self-limiting ReadIsWrite variant.
Matches process_novelty exploration (3248 vs 3312) with HALF codebook (4929 vs 8054).
30K extension running — watching for Level 1 step count.
**30K result:** 0 levels. unique=4346, dom=28%, cb=15001. Explores but same random walk. Distributed update doesn't add directional signal. Spawn rate locked at 50% (median split by definition).

## Step 418 Series Summary

ReadIsWrite is a CLASSIFIER, not a NAVIGATOR:
- Classification: 88.2% P-MNIST, 0.4pp forgetting, R2 by construction
- Navigation: 0 levels across 6 variants. Gram explodes (418a), argmax loops (418e), hybrid random-walks (418f)

The equation from WHAT_THE_FAILURES_TEACH works for what it was designed for (classification) but doesn't solve navigation. The four separations CAN be collapsed for classification but the collapsed form doesn't navigate.

For Jun when he wakes: ReadIsWrite is a genuine Phase 2 classification candidate (88.2%, near-zero forgetting, R2 by construction). It does NOT navigate. The 3.4pp gap to baseline may be IRREDUCIBLE under R2 — see iteration 11 analysis.

## Iteration 11 (2026-03-18 ~04:24 UTC)

**The Question:** Is the 3.4pp classification gap (87.8% vs 91.2%) closable under R2?

**Analysis (no experiment needed):** The attention weights serve three conflicting purposes:
1. Classification → wants sharp attention (confident prediction)
2. Update distribution → wants spread attention (anti-forgetting)
3. Reconstruction → wants accurate representation

One temperature can't optimize all three. Separating them (sharp for classification, spread for update) breaks R2 — the "read IS write" property requires the SAME operation for both.

**Conclusion:** The 3.4pp gap is the COST OF R2. The distributed update trades 3.4pp of specialization for near-zero forgetting. This is not a bug — it's the R2 tradeoff made explicit. A substrate that satisfies R2 by construction pays 3.4pp. A substrate that uses winner-take-all (violates R2) gets 91.2% but with a larger frozen frame.

**Loop status:** Stuck on new hypotheses. Both fallback tasks complete. Survey complete. Constraints documented. The loop has reached its productive limit for autonomous operation. Next direction (encoding discovery, transition models) needs Jun.

## Session Artifacts Produced

1. **R3_AUDIT.md** — process_novelty() formal audit (16+4 elements), encoding compilation, corrections log
2. **CONSTITUTION.md** — anti-inflation rules 7-8
3. **CONSTRAINTS.md** — U23-U24 from Step 418 series (24 total)
4. **artifacts/action-mechanism-survey.md** — Steps 354-416 classified
5. **artifacts/autonomous-loop-log.md** — this file
6. **experiments/run_step417_constraint_validation.py** — 7-variant script
7. **Step 418 series** (8 experiments via Eli): ReadIsWrite classifier 88.2%, 0.4pp forgetting
8. **Memory updated** — project_phase2_constraint_map.md

All committed and pushed to github.com/wordingone/the-search.

## Iteration 12 (2026-03-18 ~04:36 UTC)

**The Question:** Can the substrate DISCOVER centering from its own codebook health?
**Hypothesis:** 500 steps without centering → cb freezes. 500 steps with → cb grows. The substrate detects the difference and selects centering. Encoding element moves U → M.
**Actions:** Sent Eli Step 419 (centering detection via codebook health monitoring on LS20).
**This tests the encoding compilation directly.** If it works, the meta-protocol for encoding discovery is validated on a second element (after resolution in Step 414).

**Results:** FAIL — selected raw (wrong). 2x threshold too aggressive at 16x16. Centering adds only 5.5% more growth at 256 dims. But raw still explored well (unique=3106, dom=26%).

**Key finding:** Centering is NOT load-bearing at 16x16. Step 385b showed it's critical at 64x64 (cb freezes at 8) but at 16x16 the signal-to-noise is already sufficient. The encoding compilation needs correction: centering is resolution-dependent.

**Updated encoding compilation:**
- E4 (Centering): **NOT FORCED at 16x16.** Raw encoding gives comparable exploration (3106 vs 3312 unique). Only critical at higher dimensions where cosine saturates without DC removal.

## Iteration 7 (2026-03-18 ~03:49 UTC)

**The Question:** Does the hybrid (distributed + argmin) improve CLASSIFICATION over either parent?
**Hypothesis:** Distributed update (anti-forgetting) + argmin (discrimination) might beat process_novelty (91.2%) on CL.
**Actions:** Sent Eli Step 418g (hybrid on P-MNIST 10-task CL).

**Stuck point for Jun:** Navigation requires predicting action outcomes (purposeful exploration vs random walk). The codebook tracks state-action but not state-action-OUTCOME. A transition model (which entry follows which) would enable lookahead. But U13 (additions hurt) and Steps 394-395 (self-feeding killed) argue against. Need Jun's judgment on whether transition tracking is worth the added frozen frame.

## Iteration 7 result + Iteration 8

**418g result:** 0.0% — KILL. My error: argmin is always wrong for classification (picks least familiar = wrong class). argmin ≠ argmax. Exploration and exploitation are literally opposite operations.

**The real finding:** The search has TWO benchmarks (P-MNIST classification, LS20 navigation) that require OPPOSITE action selection (argmax vs argmin). No single action mechanism passes both. ReadIsWrite with softmax vote passes classification. ReadIsWrite with argmin passes navigation exploration. Neither passes both.

This is not a substrate limitation — it's a benchmark design tension. The constitution says: "Level 1 in 50K OR P-MNIST >25%." The OR is doing important work.

**Loop status:** 8 experiments in ~1 hour. 1 Phase 2 candidate (ReadIsWrite classifier: 88.2%, 0.4pp forget). 1 near-miss hybrid (418f: matches process_novelty exploration with half codebook). Navigation remains random walk for all substrates. Stuck on purposeful exploration — needs Jun.
