# Constraint Map

*Revised 2026-03-27. 1252+ experiments, 16 architecture families. R3 solved by composition (Step 1251).*

**Classification:** T = Task Requirement, U = Validated Universal (2+ families), P = Provisional (1 family), S = Substrate-specific (codebook/LVQ), I = Intent (capability gap), E = Engineering (not carried forward), PB = Post-Ban.

---

## Task Requirements

### Navigation (LS20, FT09, VC33)

**Three mapping properties** (observation→node): Deterministic (HYPOTHESIZED), Locally continuous (YES  - 5 families partition cleanly), Persistent (YES  - kd-tree splits destroy edges → fail). Evidence: Props 1-3, Steps 446-453.

**Algorithm invariance (REVISED).** Count-monotone representations produce identical action sequences under argmin (Prop 3 — tautology, not closure). Does NOT prove argmin is optimal or action selection is solved. Steps 521-525. **POST-BAN: INVALIDATED** — argmin requires per-(state,action) counts = banned. **POST-COMPOSITION: R2-VIOLATING** — all count-based selectors are separate evaluators (frozen frame). Reflexive map architecture requires action selection through W.

**Non-convergent exploration.** Reachable set grows sublinearly (259→439 cells at 50K→740K, k=12). Growth decays ~2 cells/100K. Steps 528-529.

**Targeted exploration kills navigation.** 6 strategies all worse than argmin (noisy TV problem). Steps 477-482, 539-541.

**L2 reward disconnect.** 4 hypotheses falsified on LSH+LS20: scale/budget, self-reference, information availability, signal source. L2 requires forward prediction, not retrospective penalty (Prop 13). Steps 486-639. Caveat: LSH+LS20 only.

**Action discovery cascade (Steps 713-719).** l0 (state-change detection): LS20 perfect, FT09 perfect, VC33 blind. l_pi (graph-novelty): hash saturation kills it. l1 (episode-outcome): argmin equalizes action usage, preventing discrimination. ALL three hierarchy levels insufficient globally.

**Gap analysis (5 untested angles):** (1) Per-cell action discrimination on VC33, (2) 674+raw 64x64, (3) 674 on established benchmarks, (4) Clean Recode test same K, (5) Non-argmin under running-mean centering.

**POMDP reframing (Steps 652-667).** L1 is recognition, not exploration. Agent visits exit 152 times before L1 triggers. LS20 has hidden states (snw, tmx, tuv). Key: graph ~ flat counter (15/20 identical); interpreter is ONE coupled operation (decoupling kills nav); argmin ~ random for L1 (3/20 each); hash resolution bimodally predicts L1 speed.

**Selective pi-refinement (Steps 672-711).** Prop 15 CONFIRMED: perception quality IS the L1 lever. 674 (transition-triggered dual-hash) = 9/10. Running-mean 674 = 20/20 on BOTH LS20 and FT09. Centering ~75% of gain, 674 mechanism ~25%. VC33 = zone discovery failure (I3), not aliasing.

**L1 vs L2 asymmetry.** L1 = bounded aliasing (solvable by pi-refinement). L2 = unbounded aliasing + energy mechanic. L2 universal: LS20 high aliasing 0/20, FT09 zero aliasing still 0/5. L2 is NOT about disambiguation. Steps 682-692.

**L1 perception-limited.** 35+ experiments across 8 intervention types confirm only perception changes improve L1. Argmin and random solve same count but different seeds. R3 for action selection structurally blocked within argmin framework. Steps 477-712.

**Interpreter entailment conjecture.** Compare-select-store may be the UNIQUE interpreter class satisfying R1-R6. Every tested architecture that passes R1-R6 is a CSE instance. Status: CONJECTURE.

**R3 as self-directed attention.** Productive R3 target is pi (encoding) = attention: channel selection, spatial resolution, region weighting, hash planes. 674 self-directs at layer 2. R3 = extend to layer 1. Formally: pi_s: X->N is state-dependent. Prop 17.

### Classification (P-MNIST)

External labels required. Self-labels = 9.8% (codebook), 10.1% (graph) = chance. LSH k=16: 36.2% self-labels (4x above codebook, Step 573). Encoding HAS class signal (NMI=0.48, Step 526). Wall is the label, not encoding.

### Both Tasks

Interactive (unknown environment, no training phase). Must learn online.

---

## Validated Universal Constraints

| # | Constraint | Evidence | Source |
|---|---|---|---|
| U1 | No separate learn/infer modes | Codebook, LSH  - task demands online learning | C1, Steps 72, 453 |
| U3 | Structural zero forgetting (growth-only) | Codebook entries, graph edges, LSH cells all accumulate | C5, Step 65 |
| U4 | Minimal description (design principle) | Team requirement  - complexity = frozen frames | Design |
| U7 | Iteration amplifies dominant features | Codebook → dominant eigenvector, Reservoir → rank-1 collapse | C27, Steps 405, 438-439 |
| U11 | Discrimination != navigation | Codebook (balanced actions, 0 levels), Graph (discriminates, doesn't navigate) | Steps 388-390, 449-451 |
| U16 | Must encode differences from expectation | Codebook: cosine saturation without centering. LSH: hash concentration without centering. Code-verified. | Steps 414, 453 |
| U17 | Unbounded information accumulation needed | Codebook: capped→stalls. LSH: fixed cells but edges grow unboundedly→navigates. | Steps 415a, 454 |
| U20 | Local continuity in input-action mapping | Grid(0/3), LSH(3/10), TapeMachine(35%→fail), ExprSubstrate(94%→fail), Codebook(by construction). 5 families. | Steps 417-453 |
| U22 | Convergence kills exploration | Codebook scores converge→random walk. TemporalPrediction→pred_err→0→frozen. Mathematical. | Steps 428, 437d |
| U24 | Exploration and exploitation are opposite operations | Argmin(explore) vs argmax(classify). 674 needs argmin's transition diversity. Architecture-independent. | Steps 418, 444b |

---

## Provisional Constraints

| # | Constraint | Status | Note |
|---|---|---|---|
| U2 | No separate memory+generation | RECLASSIFY→S | LSH/Recode navigate with decomposed components |
| U5 | Sparse selection over global | CHALLENGED | Soft probabilistic works (Step 578-579) |
| U8 | Hard selection over soft | CHALLENGED | LSH softmax T=0.5→5/5 (Step 579). Fresh_episode bug caveat. |
| U9 | Curriculum transfer | Codebook-only | Too thin to generalize |
| U10 | Dense memory kills exploration | RECLASSIFY→S | Recode 1267 cells navigates 5/5 (Step 542) |
| U12 | Goldilocks zone | RECLASSIFY→S | 3 non-codebook families navigate without it |
| U13 | Additions hurt | RECLASSIFY→S | SplitTree/Recode contradict. Ceiling-specific. |
| U14 | Substrate IS its search | META | Philosophical observation, not testable |
| U15 | Robust to perturbation | DESIGN | LSH/Recode robust. Codebook brittle. |
| U18 | Shared action channels contaminate | Partial | 0/3→2/3 via per-domain centering. Solvable engineering. |
| U19 | Dynamics != features | PARTIALLY CHALLENGED | L1: dynamics sufficient. L2+: features needed. |
| U21 | Diversity fitness collapses | ExprSubstrate-only | One experiment. Too thin. |
| U25 | Convergent action kills exploration | UPGRADE→U | 4 representations confirm. Edge growth postpones, cell growth prevents. |
| U26 | Self-generated labels compound errors | CHALLENGED | LSH 36.2% vs codebook 9.8% (Step 573) |
| U28 | No signal improves argmin; dense damages | PROVISIONAL | LSH-only (Steps 581d, 630-639). Hart debunked ~10% threshold. |
|  - | PENALTY=100 / op-code frozen | R3/R4 AUDIT | Should be adaptive. Hart identified as R4 violations. |

---

## Reclassified (Former U → S-class)

| Former | Constraint | Why reclassified |
|---|---|---|
| U1 | ~~Reclassification~~ | REVERTED  - U1 reworded, remains Validated Universal |
| U6→S23 | Lipschitz function limit | k-NN specific (codebook). LSH/graph don't use k-NN. |
| U16 | ~~Reclassification~~ | REVERTED  - code review confirmed LSH uses centering too |
| U17 | ~~Reclassification~~ | REVERTED  - LSH edge growth IS unbounded accumulation |
| U23→S26 | Distributed updates destabilize | Codebook Gram matrix specific. Neural nets use distributed updates fine. |
| U25 | ~~Reclassification~~ | REVERTED to Provisional (not S-class). Edge ratios converge locally. |

---

## Intent Constraints

| # | What's missing | Source |
|---|---|---|
| I1 | Discover own state representation from raw observations | Steps 377-414 |
| I2 | Adapt own comparison operation | Steps 388, 402, 412 |
| I3 | Discover own action space | Steps 415, 360-361 |
| I4 | Handle temporal structure | Steps 362, 374-375 |
| I5 | Transfer across levels/episodes | Step 376 |
| I6 | Generate purposeful (not random) exploration | Steps 353, 408 |
| I7 | Output more than one integer per step | Output bottleneck analysis |
| I8 | Represent and modify own rules | Stage 7 analysis |
| I9 | Intelligence is not stochastic coverage | Reviewer critique |

---

## Substrate-Specific (Codebook/LVQ)

| # | Constraint | Source |
|---|---|---|
| S1 | Cosine resolution ~ 1/sqrt(d); fails >256D with <5% signal | Steps 377-381 |
| S2 | F.normalize load-bearing (Goldilocks noise zone) | Step 412 |
| S3 | centered_enc load-bearing (without: thresh=1.0, action collapse) | Step 414 |
| S4 | Variance weighting finds signal dims but makes cosine worse | Step 381 |
| S5 | Diff encoding: discrimination != spatial exploration | Step 383 |
| S6 | Centering + small codebook → antipodal vectors | Step 385b |
| S7 | Self-feeding consolidates on timer, not position | Step 394 |
| S8 | Two-codebook: death spiral or uniform votes | Steps 398-399 |
| S9 | Recursive composition = iteration when metric saturates | Step 405 |
| S10 | Winner identity timer-contaminated at 4096D | Step 407 |
| S11 | Temporal change-rate: detection != encoding | Steps 400-401 |
| S12 | Grouped encoding: discriminates but dynamics invert | Step 402 |
| S13 | Attract-delta: finds events not sprite (self-erases) | Step 406 |
| S14 | Temperature raises max via order stats → kills spawning | Step 409b |
| S15 | Cross-resolution rotation contaminates codebooks | Steps 413, 413b |
| S16 | Codebook cap kills exploration | Step 415a |
| S17 | Spawn-delta p95 doesn't separate with centering | Step 410 |
| S18 | Evolutionary search: no signal/dim at 4096D | Step 411 |
| S19 | Partial norm p=0.75 at 4096D: healthy dynamics, no level | Step 416 |
| S20 | Sequential resolution trial: only EXACT baseline works | Step 414 |
| S21 | 5 implementation details each independently fatal | Step 414 analysis |

---

## Engineering (NOT carried forward)

| # | Constraint | Note |
|---|---|---|
| E1 | 16x16 avgpool for LS20 | Game-specific |
| E2 | 69-class click-region for FT09 | Game-specific |
| E3 | 3-zone encoding for VC33 | Prescribed |
| E4 | LS20 ignores clicks, needs directional | Game API |
| E5 | cb_cap=10K with subsampled thresh | Implementation |
| E6 | Timer at row 15 of 16x16 grid | Game visual |
| E7 | Thresh update frequency changes dynamics | Implementation |

---

## Game-Specificity Caveat

*3 preview games (LS20, FT09, VC33). ARC-AGI-3 full launch March 25, 2026: 150+ games.*

**Game-independent** (structural/mathematical): U1, U3, U7, U11, U16, U17, U20, U22, U24. Three mapping properties. Argmin invariance. Convergence kills exploration. Growth-only = zero forgetting.

**Game-specific** (may break): VC33 magic pixels, FT09 69-action decomposition, LS20 POMDP hidden states, death penalty effects, argmin purity gap, K_NAV=12, CIFAR=chance, zero cross-domain transfer.

**Untested** (highest risk): Whether chain generalizes beyond navigation puzzles. Whether avgpool16 works for non-pixel-grid games. Whether R3 measurement generalizes beyond hash substrates.

---

## State of the Search (1252+ experiments, 16 families)

### Solved

- **Navigation mechanism (REVISED):** Graph + edge-count argmin navigates preview games but is the unexamined incumbent. Prop 3 (tautology) and broken I3_rho metric (24+ false kills) protected it from comparative evaluation. R2-violating (separate evaluator). Three mapping properties predict L1 success on preview games only.
- **All 3 preview games multi-level:** LS20 L1-L3 5/5, FT09 6 levels 75 clicks, VC33 7 levels 176 clicks. All via source analysis (prescribed). 10/25 games fully solved by analytical solvers.
- **Recode achieves l_pi:** LSH k=16 + self-refinement = 5/5 (Step 542). K confound (Step 589).
- **674+running-mean = 20/20** on LS20 and FT09. Centering ~75% of gain. L1 is infrastructure, not a result (banned as metric Step 713).
- **R3 (self-modification)  - SOLVED BY COMPOSITION (Step 1251).** 7 cross-family components (centered encoding, novelty growth, transition detection, argmin, prediction-error attention, recurrent state, self-observation) composed into one substrate produce genuine self-modification of the obs→representation mapping. Jacobian ∂(attended)/∂obs differs between fresh and experienced substrate (0.05-0.09 vs 0.026 baseline). 100/100 passes across 10 games, both sequential and parallel wirings. Component-level finding, not wiring-dependent. **R3 was always achievable  - the 0/1250 failure was a composition failure, not an R3 failure.**

### Open

1. **R3→action bridge.** R3 works (Step 1251) but argmin ignores the modified representation. I3 identical for composed substrate and argmin-alone (0.67 both). L1=0% on 8/10 games for both. The substrate modifies its world model; the action selector throws it away. **The one missing component: a state-conditioned action selector that reads from h/alpha/attended representation.** 200+ debate experiments enumerate which bridges DON'T work (alpha attention, forward models, directional attention, state-conditioned rankings, MI+attention, pixel scanning, EMA, empowerment, anti-correlated pairs, softmax concentration).

2. **I1 (state-distinguishing encoding).** Encoding self-modifies (R3) but does NOT produce distinct representations for analytically-distinct game states. I1=0.00 across 100 runs, 10 games (Step 1251). The encoding changes HOW it processes but not in a way that distinguishes states. Next: contrastive/predictive encodings that are R1-R6 compatible.

3. **R1-compliant classification.** LSH 36.2% self-labels (Step 573). Far below supervised. Unsolved.

4. **Purposeful exploration (I6, I9).** Argmin unbeatable within current framework. L2 requires different mechanism. Coupled to R3→action bridge  - a state-conditioned selector may provide purposeful exploration as a byproduct.

5. **L2+ across all games.** L2 reached twice (Steps 1074, 1211), never reproduced. 0/1251 substrates reliably reach L2. Blocked by open items 1+2.

### SOTA Baselines (Steps 760-766, 917-920)

895h/916 outperform ALL published baselines on LS20 by 2-2.5x. ICM=0, Count=109, RND=112, Graph+argmin=130, 868d=204, 895h=268, 916 recurrent=290.7. CIFAR=chance (20.21%). Atari 6/26 above random. Zero cross-domain transfer.

### External Audit (13 findings, 2026-03-18)

Integrated: 4, 5, 7, 12. Partially: 1. Outstanding: 2 (HIGH  - no CL head-to-head), 3, 6, 8, 9, 11. Low: 10, 13.

### Phase Transitions

Phase 1 (416 exp): codebook characterized, LVQ=argmin. Phase 2: 12 families killed except LSH. Phase 3 (post-777): both bans active. Constraint map was biased  - feasible region LARGER than claimed.

---

## Post-Ban Constraints (Steps 778-937)

| # | Constraint | Status | Key evidence |
|---|---|---|---|
| PB1 | Location state transfers negatively | CONFIRMED+GENERALIZED (Corollary 20.1) | Steps 776, 803, 788, 806v2 |
| PB2 | Hebbian W diverges (delta rule fixes) | CONFIRMED | Steps 778-787 |
| PB3 | Compression progress → action collapse | CONFIRMED | Step 855/855b |
| PB4 | No post-ban nav consistently beats random | CONFIRMED | Steps 778-812. Prop 21. |
| PB5 | D(s) prediction transfer is real | CONFIRMED (5/7 PASS) | Steps 778v5, 780v5, 809b, 855b, 855v3 |
| PB6 | Entropy-seeking hurts dynamics learning | CONFIRMED | Step 856 |
| PB7 | Encoding irrelevant without action mechanism | CONFIRMED | Step 817 |
| PB8 | Post-ban frozen frame thinner (3-4 vs 9) | CONFIRMED | Step 878 Table 3 |
| PB9 | Pred-error attention achieves R3 encoding self-mod | CONFIRMED (n_eff=10) | Steps 895, 895b, 895f. Prop 22. |
| PB10 | Prediction accuracy cascade | DISPROVED | W pred_acc negative. Alpha works via error distribution, not accuracy. |
| PB11 | Cold clamped alpha +32% over baseline | CONFIRMED (n_eff=10) | 895h=268.0 vs 868d=203.9 |
| PB12 | Warm alpha transfer UNRELIABLE | R3_cf NAV DISPROVED | 895h warm=209.2 < cold=268.0. Retracts 895c/895e. |
| PB13 | FT09 fails: can't learn click ordering | CONFIRMED | Step 895f |
| PB14 | Action selection modification family dead | CONFIRMED | Steps 910-915. 5 approaches all below 895h. |
| PB15 | VC33 = 0 post-ban | CONFIRMED (first measurement) | Step 914 |
| PB16 | CIFAR pre-training inflates alpha | CONFIRMED | Step 914 chain |
| PB17 | Recurrent trajectory = NEW LS20 SOTA 290.7 | CONFIRMED (10 seeds) | Step 916 |
| PB18 | 895h/916 beat all baselines 2-2.5x | CONFIRMED | Steps 917-920 |
| PB19 | FT09 bottleneck = action space (68^7 untractable) | CONFIRMED | Steps 915, 920 |
| PB20 | Per-obs action memory = graph ban violation | KILLED | Step 931 |

### Post-ban feasible region

- **Prediction transfer region NON-EMPTY.** D(s)={W,running_mean} transfers (5/7 PASS). First positive R3_cf.
- **Navigation transfer region EMPTY.** No mechanism beats random consistently.
- **Gap is structural (Prop 21):** D(s) captures global dynamics, nav needs local per-state selection (banned).
- **R3 encoding self-mod IS achievable (PB9).** Alpha concentrates on informative dims. Game-adaptive.
- **W is signal generator, not predictor (PB10 wrong).** Alpha works via differential error distribution.
- **Action selection family dead (PB14).** 5 approaches, all below 895h cold.
- **Recurrent h NEW SOTA (PB17).** 916=290.7 vs 895h=268.0 (+8.5%).

### Constraints invalidated by graph ban

Algorithm invariance, U3 (needs re-test), U17 (needs re-test), U24 (reframe needed), U25 (mechanism gone, principle survives), U28 (invalidated), three mapping properties (partially valid  - mapping survives, graph doesn't), argmin purity (invalidated), L1 perception-limited (needs re-test within post-ban framework).

---

## Kill Criteria Framework

**Universal benchmarks:** LS20 L1 (0 levels at 50K/10 seeds = fail), FT09 L1 (same), P-MNIST >25% AA / 0pp forgetting, Cross-domain (navigate after classification).

**Meta-rules:** Min 20 experiments before killing a family. 3 consecutive structural failures = one death mode, not family dead. Kill criteria in family's diagnostic language. Benchmark universal, explanation family-specific. **Incumbent evaluation is symmetric** — the control baseline faces the same kill criteria as alternatives. A control that would fail its own stage criteria is not a valid control. Asymmetric evaluation (alternatives killed, incumbent untested) is itself a defect in the evaluation framework (learned 2026-03-28, 30+ experiments).

**Family diagnostics:** See kills/ directory for per-family kill registers.
