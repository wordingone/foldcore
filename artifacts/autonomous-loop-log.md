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

For Jun when he wakes: ReadIsWrite is a genuine Phase 2 classification candidate (88.2%, near-zero forgetting, R2 by construction). It does NOT navigate. The 3.4pp gap to baseline is in per-step learning quality, not architecture. Navigation remains unsolved.
