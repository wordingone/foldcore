# Constraints — Comprehensive Extraction from Experiments

*Revised 2026-03-25. Compressed per doc-system limits. Full narratives in RESEARCH_STATE.md. Component extraction catalog in COMPONENT_CATALOG.md.*

*1060+ experiments across 16 families. Codebook banned Step 416. Graph banned Step 777. See bans/POLICY.md for extraction protocol.*

**Classification:**
- **Task Requirement**: What the TASK demands of any substrate (derived from successes AND failures across families)
- **U (Validated Universal)**: Confirmed by 2+ architecturally distinct families
- **P (Provisional)**: Single-family evidence — hypothesis to test, not law to enforce
- **S (Substrate-specific)**: Specific to LVQ/codebook/cosine-based substrates (includes 6 former U-constraints reclassified with contradicting evidence)
- **I (Intent)**: What CAPABILITY the constraint reveals is missing
- **E (Engineering)**: Implementation detail — NOT carried forward

---

## Task Requirements

*Added 2026-03-18 session 2. The task is the oracle. Architecture is the variable. These requirements are derived from what ALL successful and failed substrates tell us about the tasks themselves, not about any particular architecture.*

### Navigation (LS20, FT09, VC33)

**Three mapping properties predict L1 success/failure across 6 families:**

| Property | Required? | Evidence (brief) |
|---|---|---|
| **Deterministic** (same obs → same node) | HYPOTHESIZED | No independent evidence. |
| **Locally continuous** (nearby obs → same node) | YES | 4 families fail: grid graph, reservoir, CA, PCA grid. |
| **Persistent** (nodes don't get destroyed) | YES | kd-tree fails (splits destroy edges). |

**Algorithm invariance — QUALIFIED (Step 948).** Argmin over visit frequency is invariant across 4 graph-based representations (Steps 521-525). Hebbian RNN navigated WITHOUT argmin (1/10 seeds, Step 948) — non-argmin possible but unreliable. Post-ban: 800b delta EMA is the only working non-graph mechanism.

**Per-game best results (all time):** See COMPONENT_CATALOG.md for full performance audit.

| Game | Best pre-ban | Best post-ban | Gap |
|------|-------------|--------------|-----|
| LS20 | 20/20 (674+rm, Step 705) | 290.7/seed (916, Step 916) | Post-ban BETTER |
| FT09 | 20/20 (674+rm, Step 709) | **0/10 all mechanisms** | Complete loss |
| VC33 | 5/5 (CC discovery, Step 576) | **0/10 all mechanisms** | Complete loss |
| CIFAR | 94.48% w/ labels (Step 425) | ~chance without labels | R1-limited |

### Post-Ban Findings (LS20-focused, 230+ experiments)

**Research skew:** 230 post-ban experiments exclusively LS20. The three mapping properties (deterministic, continuous, persistent) predict LS20 L1 only. Click games demand capabilities not captured by any existing constraint.

**Findings:**
- Exploration is non-convergent (sublinear reachable set growth)
- Targeted exploration kills LS20 navigation (noisy TV)
- L1 is perception-limited, not action-limited
- R3 = self-directed attention on encoding (alpha, 895)
- CSE interpreter conjecture (5 families confirm)
- Game versions change (Steps 690+ on current versions)
- Step 1017: full graph + all bans lifted = FT09/VC33 still 0%. Generic exploration can't discover multi-step sequences.
- Step 576 (VC33 5/5, mode map + CC) = the only autonomous multi-game solve.

### Classification (P-MNIST)

Classification requires external labels. Step 432: self-labels = 9.8% (chance), external labels = 94.48%. Confirmed across codebook and graph (Step 444b). No substrate has achieved R1-compliant classification.

**Encoding HAS class signal (Steps 510, 526).** avgpool16+centered 256D produces NMI=0.48 with LSH k=12 random hyperplanes (Step 526) — better than codebook NMI=0.42 at same granularity. The partition method doesn't determine class structure — the encoding does. But no mechanism extracts this signal without external labels. The wall is the label, not the encoding.

### Both Tasks

The task is interactive (unknown environment, no separate training phase). Any substrate must learn online — not because read/write must be "one operation" (LSH separates them and navigates), but because the task doesn't offer a training set.

---

## Validated Universal Constraints (confirmed by 2+ architecturally distinct families)

| # | Constraint | Evidence across families | Source |
|---|---|---|---|
| U3 | Structural zero forgetting | Codebook: growth-only (no entries deleted). Graph: edges accumulate. LSH: cells permanent, edges accumulate. All successful substrates are growth-only or persistent. | C5, Step 65 |
| U7 | Iteration amplifies dominant features | Codebook: recursive composition → dominant eigenvector (Step 405). Reservoir: Hebbian dynamics → rank-1 collapse (Steps 438-439). Mathematical property of iterative systems. | C27 |
| U11 | Discrimination ≠ navigation | Task-level truth. Codebook at 64x64: balanced actions, 0 levels (Steps 388-390). Graph with random walk: discriminates but doesn't reliably navigate (Steps 449-451). Confirmed across all families. | Steps 388-390, 449-451 |
| U20 | Local continuity in input-action mapping | Grid graph: no local continuity → 0/3 (Steps 446-447). LSH: locality-preserving hash → 3/10 (Step 453). TapeMachine: 35% disc → fails. ExprSubstrate: 94% collapse → fails. Codebook: cosine provides it by construction. The strongest cross-family constraint — 5 families tested, clean partition. | Phase 2 Steps 417-453 |
| U22 | Convergence kills exploration | Codebook: score convergence → random walk after ~5K steps (Step 428). TemporalPrediction: pred_err → 0 → W frozen → action locked (Step 437d+). Two architecturally distinct confirmations. Mathematical: any convergent adaptation makes the environment stationary → trivially predictable → no learning signal. | Steps 428, 437d, Phase 2 |
| U1 | No separate learn and infer modes | Codebook: process() learns (attract/spawn) and acts (score/argmin) on every step. LSH: step() hashes (read), updates edges (write), and selects action in one call. No family has separate training and inference phases. The TASK demands online learning (interactive environment, no training set). Reworded from "read and write must be one operation" — the principle (no separate modes) holds across families even though read/write are decomposable sub-steps. | C1, Steps 72, 453. Adversarial review corrected over-demotion. |
| U24 | Exploration and exploitation are opposite operations | Mathematical fact: argmin (least familiar) explores, argmax (most familiar) classifies. No single mechanism serves both. Codebook: argmin = 0% classification (Step 418g). Graph/LSH: edge-count argmin explores but can't classify. Architecture-independent. **Compression note (2026-03-22):** argmin ≈ random for L1 speed (Step 653: 3/20 each). U24's PRACTICAL implication weakens if action selection doesn't matter for navigation. But 674's 20/20 requires diverse transitions (aliasing detection) which argmax would suppress. Prediction: argmax + 674 ≈ 3/20 (random walk). U24 holds practically because 674's mechanism needs argmin's transition diversity. Untested (gap #5). If encoding self-modification reduces the transition-diversity requirement, the argmin dependency may weaken — connecting R3 to the U24 tension. | Steps 418, 444b |
| U16 | The substrate must encode differences from expectation | Codebook: centered_enc load-bearing — without it, thresh=1.000 and action collapse (Step 414 runs 1-2). LSH: centered_enc applied before hashing (Step 453 code confirmed) — prevents hash concentration. Same mechanism (x - x.mean()), two architecturally distinct families, different failure modes. Centering solves a broader problem than cosine artifacts. | Steps 414, 453. Adversarial review corrected false reclassification. |
| U17 | Exploration requires unbounded information accumulation | Codebook: capped entries → exploration stalls (S16, Step 415a). LSH: fixed cells (1024) but edges grow unboundedly → navigates (Step 454). Both families require SOME form of growth — entries for codebook, edges for LSH. The principle is "unbounded accumulation needed," not "unbounded entry count." Reworded from "fixed-capacity memory exhausts" — LSH's edge growth satisfies the principle via a different mechanism. | S16, Steps 415a, 454. Adversarial review corrected over-demotion. |

### Design Requirements (from review, not from experiments)

| # | Constraint | Note |
|---|---|---|
| U4 | Minimal description | Team requirement. Complexity = frozen frames. Not experimentally derived, but valid as a design principle. |

---

## Provisional Constraints (single-family evidence — hypotheses to test, NOT laws to enforce)

*These were extracted from codebook experiments only. They may be universal, codebook-specific, or wrong. Each needs testing against non-codebook families before being used to narrow the feasible region. The codebook got 435 experiments; reservoir got 8; LSH got 4. This asymmetry means most "universal" claims are really "true for codebooks, untested elsewhere."*

| # | Constraint | Status | Why provisional |
|---|---|---|---|
| U2 | No separate memory + generation | **RECLASSIFY → S-class** | LSH navigates with decomposed components: hash function (mapping) + edge dict (memory) are separate subsystems. Recode adds a third (refinement tree). The original constraint (codebook's process() as single function) was codebook-specific. Decomposed components work fine for navigation. |
| U5 | Sparse selection over global aggregation | **CHALLENGED** | Hebbian uses global encoding (W.T@x) but SPARSE selection (argmin picks one action). Step 578: Recode k=16 with softmax(-count/T=1.0) → 3/3 L1 at 50K. Soft probabilistic action selection navigates successfully. U5 as written (hard argmin required) is CHALLENGED: the sparse selection direction (prefer low-count actions) is preserved; the hard/soft distinction is not. |
| U8 | Hard selection over soft blending | **CHALLENGED** | Codebook: soft blending → centroid convergence. Step 579: LSH k=12 T=0.5 softmax → 5/5 L1 at 50K; T=0.1 → 3/5. Soft selection WORKS for LSH navigation. Caution: note these experiments may have counted false-positive L1s (LS20 reports cl=1 for one step after reset — fresh_episode bug, see step 577 R3 work). Results likely overcount L1 "wins." Retest needed with fresh_episode fix. Provisional status: CHALLENGED but not confirmed. |
| U9 | Curriculum transfer | Codebook-only | Tested only on codebook curriculum. Neural network curriculum shows different patterns. |
| U10 | Dense memory kills exploration | **RECLASSIFY → S-class** | Recode has 1267 cells and navigates 5/5 (Step 542). LSH with 1605 cells navigates at k=20 (Step 531). The original finding was cosine saturation at 4096D, not density per se. Dense state is fine when the mapping preserves local continuity. |
| U12 | Structured noise / Goldilocks zone | **RECLASSIFY → S-class** | LSH navigates without any Goldilocks zone (Step 453). Recode navigates without one (Step 542). Hebbian navigates without one (Step 524). The Goldilocks zone is a codebook artifact from F.normalize + centered_enc interaction. 3 non-codebook families navigate without it. |
| U13 | Additions hurt | **RECLASSIFY → S-class** | SplitTree refinement splits navigate L1 (Step 537) but fail chain (Step 538). Recode self-refinement navigates 5/5 (Step 542). Two non-codebook families where additions don't hurt navigation, though SplitTree evidence is mixed. Original codebook (343-376) and reservoir (438-439) evidence was at capability ceilings. Reclassified: ceiling-specific, not universal. |
| U14 | Substrate IS its search procedure | **META** | Observation about the codebook's self-similar structure. Not a testable constraint. Recode partially confirms: the substrate's refinement IS its search (splitting confused transitions). But this is a philosophical observation, not an experimental constraint. |
| U15 | Robust to perturbation | **DESIGN** | LSH robust (navigates across random seeds, Steps 453-531). Recode robust (5/5, Step 542). Codebook brittle (5 kill switches, Step 414). Cross-family evidence exists but this is a design requirement, not a discovered constraint. |
| U18 | Shared action channels contaminate | Partial (cross-family) | Codebook: Steps 413-413b. LSH chain: Steps 523 (0/3), 533 (1/3), 546 (2/3) — contamination exists but is partially mitigable via per-domain centering. Cross-family evidence (codebook + LSH) confirms contamination EXISTS, but the 0/3→2/3 trajectory suggests it's a solvable engineering problem, not a fundamental constraint. |
| U19 | Dynamics ≠ features | **PARTIALLY CHALLENGED** | Step 416 (p=0.75): good codebook dynamics, no features at 64x64. Step 574: LSH k=12 raw 64x64 achieves L1 reliably (1191-1418/seed) — dynamics alone sufficient for L1. BUT L2=0 (sequencing needs features). **Refined: dynamics sufficient for L1 (single-touch exploration), insufficient for L2+ (sequencing requires task-relevant features).** The constraint holds for complex tasks but not for simple exploration. |
| U21 | Diversity fitness collapses | ExprSubstrate-only | One experiment on ExprSubstrate. Too thin to generalize. |
| U25 | Convergent action selection kills exploration | **UPGRADE → Validated U** | Now confirmed across 4 representations (Steps 521-525): edge dict, weight matrix, transition tensor, n-gram all produce argmin which converges locally. Steps 528-529 confirm empirically: growth rate decays to ~2 cells/100K at 740K. Coupled to U17: edge growth postpones convergence, self-refinement (Recode) expands the cell set, but convergence is projected to be eventual (not empirically confirmed for self-refining substrates). Known in literature as count-based exploration decay (Bellemare et al. 2016). |
| U26 | Self-generated labels compound errors | **CHALLENGED** | Codebook: 9.8% self-labels (Step 432). Graph: 10.1% (Step 444b). Both NN-voting = near chance. BUT Step 573: LSH k=16 achieves **36.2% test accuracy** with self-labels on P-MNIST — 4x above codebook. The self-label failure was codebook-specific (NN-voting on cosine centroids), not universal. LSH cells generalize better because random hyperplanes create more diverse partitions than cosine attractors. Note: splits=0, so this is pure LSH, not Recode. Coverage=70% (30% of test images map to unseen cells). |
| — | ~~U27 attractor hypothesis~~ | **DOWNGRADED to hypothesis** | Hart review Round 2 (2026-03-21): the 477-482 evidence was corrected to "mechanism-specific, not categorical" — which undermines using the same evidence for a categorical attractor pattern. Moved to What's Open item 3 as interpretive lens, not constraint. |
| U28 | No auxiliary signal improves argmin; dense signals damage it | **PROVISIONAL** | LSH only (Steps 581d, 630-639, 12 experiments). No signal at any density significantly improves argmin navigation (581d at n=20: p=0.63, Step 584). Dense signals (>~60% firing rate) actively damage navigation: delta PREFER at 66% → 3/5 seeds 10-18x slower (633); frontier at 94-98% → 3/5 seeds 5-20x slower (635). Sparse/moderate signals (<~55%) are inert: stale at 12% → 0.94x (637); delta at 33-42% → identical to argmin (634); stale at 54% → inert (636). Environmental events (1-5%) seed-dependent (639). Transition from inert to damage between 54% and 66%. Hart review (2026-03-21): original ~10% threshold was post-hoc fitting to 581d benefit claim, which was debunked at n=20. Single-family evidence (LSH). |
| — | PENALTY=100 and op-code definitions are frozen value judgments | **R3/R4 AUDIT QUEUE** | Step 581d: PENALTY=100 is designer-prescribed. Fails R3 (not adaptive) and R6 (removable — 4/5→3/5, not all capability lost). Should be made adaptive: the graph already encodes death information (death edges lead to start node), so the substrate COULD discover appropriate avoidance without a prescribed magnitude. Step 582: op-code assignment is self-derived (R3 partial), but op definitions (what op2 and op3 DO) remain designer-prescribed. Hart review (2026-03-21) identified these as R4 violations, not R1 violations. |

---

## Reclassified: Former U-constraints now S-class

*These were labeled "Universal" but are contradicted by non-codebook evidence. They are codebook/cosine-specific properties, not task requirements.*

| Former # | Constraint | Why reclassified | Contradicting evidence |
|---|---|---|---|
| — | ~~U1 reclassification REVERTED~~ | **U1 reworded and remains Validated Universal.** Adversarial review: the principle (no separate learn/infer modes) holds across all families. The original wording ("one operation") was too strong — read/write ARE decomposable sub-steps in LSH. But no family has separate training/inference modes. Reworded to capture the principle, not the mechanism. | Corrected 2026-03-18 session 2. |
| ~~U6~~ → S23 | Similarity-based discovery finds only Lipschitz functions | This is a mathematical property of k-NN, which is codebook-specific. LSH doesn't use k-NN. Graph uses edge counts. Note: LSH's locality-preserving hash IS a form of local similarity (nearby observations hash together with probability proportional to angular similarity). The specific theorem doesn't apply, but the underlying concern (local methods miss global structure) may resurface for LSH. | Irrelevant to non-codebook families as theorem; underlying concern persists. |
| — | ~~U16 reclassification REVERTED~~ | **U16 remains Validated Universal.** Adversarial review (Hart) confirmed via code review: Step 453 LSH uses centered_enc explicitly (x - x.mean() before hashing). Centering is load-bearing for TWO distinct families: codebook (prevents cosine saturation, Step 414) AND LSH (prevents hash concentration, Step 453). Different failure modes, same mechanism required. Original reclassification was factually wrong — introduced bias by not verifying code before claiming "no centering." | Corrected 2026-03-18 session 2. |
| — | ~~U17 reclassification REVERTED~~ | **U17 reworded and remains Validated Universal.** Adversarial review: LSH's edge growth IS unbounded information accumulation. LSH doesn't violate U17 — it SATISFIES it via edge growth instead of entry growth. The principle holds: exploration requires unbounded accumulation of some kind. | Corrected 2026-03-18 session 2. |
| ~~U23~~ → S26 | Distributed updates destabilize similarity structure | Specific to codebook Gram matrix structure. Updating all entries pushes them toward each observation. Neural networks use distributed updates (backprop) successfully. Edge-based systems don't have this problem. | Codebook-specific mechanism. Step 418 series. |
| — | ~~U25 reclassification REVERTED to Provisional~~ | **U25 moved to Provisional (not S-class).** Adversarial review: edge-count RATIOS do converge (law of large numbers). Local convergence (per-cell) is qualitatively slower than global convergence (codebook scores), but with fixed cell count, local convergence eventually exhausts all cells. U25 and U17 are COUPLED: edge growth postpones convergence, but only cell growth prevents it indefinitely. | See Provisional U25 entry for full framing. |

---

## Intent Constraints (what capability is NEEDED, mapped from ARC-AGI-3's design intent)

*ARC-AGI-3 tests: adaptation to unknown environments, learning from interaction, intelligence generated on the fly, perception + action.*

| # | Constraint | What CAPABILITY is missing | Source |
|---|---|---|---|
| I1 | The substrate must discover its own state representation from raw observations | ARC-AGI-3 provides raw pixels. The substrate must determine WHAT matters without being told. This is not "choose a resolution" — it's "discover what distinguishes states that lead to different outcomes." | Steps 377-414 (35 experiments) |
| I2 | The substrate must adapt its own comparison operation | At different scales (dimensions, signal ratios), different metrics work. The substrate must detect when its metric fails and change it. Fixed cosine = fixed capability range. | Steps 388, 402, 412, Stage 7 analysis |
| I3 | The substrate must discover its own action space | Games present different action interfaces (clicks, directions, timing). The substrate must determine what actions are available and what they do from interaction alone. | Steps 415, 360-361 |
| I4 | The substrate must handle temporal structure | Some environments require timing (VC33: WHEN to act matters, not WHERE). Codebook stores states, not sequences. Temporal reasoning is absent. | Steps 362, 374-375, C34 |
| I5 | The substrate must transfer across levels/episodes | Level transitions reset the game. Knowledge from level 1 should inform level 2. Currently: codebook resets, no transfer. | Step 376 |
| I6 | The substrate must generate purposeful exploration, not random coverage | Argmin = biased random walk. Real intelligence explores STRATEGICALLY (test hypotheses, verify predictions). Stochastic coverage scales exponentially with game complexity. | Reviewer critique, Steps 353, 408 |
| I7 | The substrate's output must express more than one integer | One action per step = 2 bits of communication with the environment. The substrate KNOWS more than it can SAY. Richer output = faster learning. | Output bottleneck analysis, session |
| I8 | The substrate must represent and modify its own rules | Stage 7: parameters adapt (thresh, alpha) but operations don't (cosine, top-K, attract, spawn). Self-improvement requires changing the learning algorithm, not just its parameters. | Stage 7 analysis, reviewer critique |
| I9 | Intelligence is not stochastic coverage | Finding a level by random walk is not intelligence. Intelligence is: forming a model, making predictions, testing them, updating the model. The substrate does none of these explicitly. | Reviewer critique |

---

## Substrate-Specific Constraints (true for LVQ/codebook, may NOT apply to other architectures)

| # | Constraint | Source |
|---|---|---|
| S1 | Cosine angular resolution ∝ 1/√d; fails above ~256D with <5% signal | C35, Steps 377-381 |
| S2 | F.normalize is load-bearing for codebook dynamics (creates Goldilocks noise zone) | Step 412 |
| S3 | centered_enc is load-bearing (without it, thresh=1.000, action collapse) | Step 414 runs 1-2 |
| S4 | Variance weighting finds signal dims but makes cosine WORSE on those dims | C36, Step 381 |
| S5 | Diff encoding discriminates but diff-novelty ≠ spatial exploration | C37, Step 383 |
| S6 | Centering with small codebook → antipodal vectors, negative thresh | C38, Step 385b |
| S7 | Self-feeding consolidates along dominant axis (timer), not target axis (position) | Step 394 |
| S8 | Two-codebook (class vote as encoding) creates death spiral or uniform votes | Steps 398-399 |
| S9 | Recursive composition = iteration when metric saturates at all scales (C27 applies) | Step 405 |
| S10 | Winner identity is timer-contaminated at 4096D | Step 407 |
| S11 | Temporal change-rate detection finds sprite region but detection ≠ encoding | Steps 400-401 |
| S12 | Grouped encoding discriminates (thresh=0.798) but dynamics invert (too few spawns) | Step 402 |
| S13 | Attract-delta variance finds game events, not sprite (surprise self-erases after learning) | Step 406 |
| S14 | Temperature on sims raises max via order statistics → kills spawning | Step 409b |
| S15 | Rotation across resolutions contaminates codebooks with incoherent actions | Steps 413, 413b |
| S16 | Codebook cap kills exploration (baseline needs uncapped growth for stochastic coverage) | Step 415a cap |
| S17 | Spawn-delta p95 doesn't separate when centering normalizes scale across dims | Step 410 |
| S18 | Evolutionary search in 4096D has no signal per dimension (1/4096 contribution undetectable) | Step 411 |
| S19 | Partial normalization (p=0.75) at 4096D creates healthy dynamics (cb=2133, dom=41%) but still no level — Goldilocks zone is narrow and resolution-dependent | Step 416 |
| S20 | Sequential resolution trial works (16x16 finds level at 26K) but only with EXACT baseline (centered_enc + F.normalize + full thresh update) — any deviation kills it | Step 414 runs 1-3 |
| S21 | Implementation details ARE the substrate: 5 differences from baseline (normalize-on-store, thresh frequency, thresh method, re-normalize attract, cb_cap) each independently fatal | Step 414 failure analysis |

---

## Engineering Constraints (do NOT carry forward — specific to current implementation)

| # | Constraint | Why NOT carried forward |
|---|---|---|
| E1 | 16x16 avgpool works for LS20 | Game-specific resolution, not universal |
| E2 | 69-class click-region encoding for FT09 | Game-specific action mapping |
| E3 | 3-zone encoding for VC33 | Prescribed from looking behind the scenes |
| E4 | LS20 ignores click coordinates, needs native directional actions | Game API detail |
| E5 | cb_cap=10K with subsampled thresh avoids OOM | Implementation detail |
| E6 | Timer at row 15 of 16x16 grid | Game-specific visual detail |
| E7 | Thresh update every step vs every N spawns changes dynamics | Implementation sensitivity |

---

## Game-Specificity Caveat (2026-03-25)

*ARC-AGI-3 full set LIVE (2026-03-25). 150+ games, 1000+ levels. Constraints derived from 3 preview games + 20+ PRISM experiments on random draws from 25-game API pool. Full 150+ pool now available.*

**Game-independent:** U1, U3, U7, U11, U16, U17, U20, U22, U24. Three mapping properties. Algorithm invariance. Mathematical arguments survive new games.

**Game-specific (may break):** VC33 magic pixels, FT09 69-action decomposition, LS20 POMDP hidden state, death penalty effects, argmin purity, LS20 action persistence (not action-0 — RNG artifact, Steps 778-784v2). K_NAV=12 tested on LS20 only. All 3 preview games are now a tiny fraction of the full 150+ set.

**Untested risks:** Non-navigation games (strategy, timing, memory). Non-pixel-grid encodings. R3 measurement beyond hash substrates. New game modalities in the 150+ set may invalidate assumptions from 3-game experiments.

**PRISM sampling update:** `--random-games 3` must draw from full 150+ pool. Infrastructure audit pending (Step 1065).

---

## The State of the Search (1060+ experiments, 16+ families, 2 bans)

*Revised 2026-03-25. See RESEARCH_STATE.md for full log, COMPONENT_CATALOG.md for extraction inventory.*

### What's Solved

- **Navigation mechanism:** Graph + argmin, confirmed across 6 families. Three mapping properties predict L1 with 100% accuracy. Post-ban: 800b + alpha (Steps 895-994).
- **All 3 games L1:** LS20 L1-L3, FT09 6 levels, VC33 7 levels — all via prescribed pipelines. 16 test cases for R3.
- **Cross-game detection (Step 576):** Mode map + isolated CC discovers interactive objects across games autonomously. Only autonomous multi-game solve.
- **Alpha R3 confirmed (Steps 895-895h):** Prediction-error attention self-modifies encoding. First post-ban R3.

### What's Open

1. **R3 (self-modification) — OPEN.** 0/4 calibration targets pass R3 (Table 1). 0/12 killed families pass R1 (Table 2). R3_counterfactual: 674 warm HURTS (cold>warm, p<0.0001, Step 776). R3_dynamic metric can't distinguish useful from random modification (Step 739). Alpha achieves R3 for encoding (PB9, Steps 895-895h). Action R3 structurally blocked within argmin. Post-ban: R4 = discriminative capacity (Ashby). alpha_conc < 30 = health diagnostic.

2. **R1-compliant classification:** Best = LSH k=16, 36.2% self-labels (Step 573). No substrate above chance without external labels.

3. **Purposeful exploration (I6, I9):** 6 targeted strategies worse than argmin (477-482). Dense auxiliary signals damage argmin (>60% firing rate). L2 requires fundamentally different mechanism.

4. **Component extraction (2026-03-24):** 33 components cataloged. 6 families under-explored (<6 experiments). FT09/VC33 action-space discovery = zero post-ban experiments. See COMPONENT_CATALOG.md.

5. **Unconstrained diagnostic COMPLETE (Step 1017):** ALL bans lifted + ALL rules suspended = FT09/VC33 still 0%. **Bans and constitution are orthogonal to the click game problem.** The constraint map describes LS20 navigation, not the full search space. The substrate lacks basic task competence (causal discovery of game mechanics) — a pre-constitutional capability that R1-R6 don't address.

6. **ℓ_π > ℓ₁ on responsive games (PB26, Steps 1042-1064).** Building sprint proved encoding modification dominates parameter modification — 10:1 advantage on same architecture. But L2+ = 0 across ALL substrates. The debate's win condition (100% ALL LEVELS, verified 25 times) = the original search problem.

7. **Opaque games (PB27).** ~1/3-2/3 of random game pool produces zero detectable change. Jun (2026-03-25): not a wall — data about what the substrate can't yet perceive. Related to ExoMDP (Efroni et al., 2022) but more extreme.

8. **ARC-AGI-3 full set LIVE (2026-03-25).** 150+ games, 1000+ levels. Pool 6x'd from 25 to 150+. $700K grand prize. PRISM infrastructure audit pending.

### External Audit Status (2026-03-18, 13 findings)

Integrated: 4, 5, 7, 12. Partially: 1 (R1 clarified). Outstanding HIGH: 2 (CL comparison). Others: 3, 6, 8, 9, 11 (mixed severity). See RESEARCH_STATE.md.

---

## Post-Ban Constraints (Steps 778-1007, 230+ experiments)

*Full details in RESEARCH_STATE.md and kill registers (kills/*.md). Extraction protocol: bans/POLICY.md.*

### Surviving: U1, U4, U7, U11, U16, U20, U22, I1-I9 (graph-independent, mathematical, or task-level)

### Invalidated or needing re-test
Algorithm invariance (argmin banned), U3/U17 (graph growth banned — what accumulates?), U24/U25/U28 (argmin-specific), argmin purity, L1-as-perception-limited (argmin framework only), three mapping properties (partially valid — encoding requirements survive, graph usage banned). Targeted exploration and noisy TV (477-482): tested against argmin, not random — post-ban baseline is random.

### Post-ban findings (PB1-PB20, Steps 778-1007)

| # | Finding | Status |
|---|---------|--------|
| PB1 | Location state transfers negatively (ANY action-coupled state) | CONFIRMED (Steps 776, 803, 788) |
| PB2 | Hebbian W diverges (delta rule fixes) | CONFIRMED |
| PB3 | Compression progress → action collapse | CONFIRMED |
| PB4 | Post-ban navigation ≈ random walk (800b = only exception, LS20 only) | CONFIRMED |
| PB5 | D(s) prediction transfer real (+73%, 5/7 PASS, first R3_cf) | CONFIRMED |
| PB6 | Entropy-seeking hurts dynamics learning | CONFIRMED |
| PB7 | Encoding irrelevant without action mechanism | CONFIRMED |
| PB8 | Post-ban R3 wall thinner (3-4 frozen vs 674's 9) | CONFIRMED |
| PB9 | Alpha prediction-error attention = R3 encoding (FT09 dims [60,51,52]) | CONFIRMED (n=10) |
| PB10 | Prediction accuracy cascade | DISPROVED (W pred_acc negative) |
| PB11 | Cold clamped alpha = 268.0/seed (+32% baseline) | CONFIRMED |
| PB12 | Warm alpha transfer unreliable (cold > warm at n=10) | R3_cf DISPROVED |
| PB13 | FT09 fails with 800b (sequential ordering bottleneck) | CONFIRMED |
| PB14 | ALL action selector modifications degrade LS20, none crack FT09 (27 kills) | CONFIRMED |
| PB15 | VC33 = 0 post-ban | CONFIRMED |
| PB16 | CIFAR inflates alpha_conc (-11% chain LS20) | CONFIRMED |
| PB17 | Recurrent h = NEW LS20 SOTA (290.7/seed) | CONFIRMED |
| PB18 | 916/895h beat ALL baselines 2-2.5× on LS20 | CONFIRMED |
| PB19 | FT09 bottleneck = temporal credit for sequential actions. Generic exploration (graph+argmin, sequence novelty, attention) ALL = 0/10 even with bans lifted (Step 1017). Only prescribed deterministic solution = 6/6 (Step 1012). No discovery mechanism works. | REVISED (Steps 920/920b/1017 vs 1012) |
| PB20 | Per-observation action memory = graph-banned | KILLED |
| PB21 | Generic exploration = 0 on click games even with ALL bans lifted (Step 1017). Gap is discovery mechanism, not constraints. | CONFIRMED (Step 1017) |
| PB22 | Direction 1: 5 extraction experiments (1007-1014) = 0 FT09/VC33 signal | CONFIRMED |
| PB23 | Game-agnostic base (no 800b/alpha/h) maintains LS20 via bootloader only | CONFIRMED (Step 1014) |
| PB24 | LS20-tuned foundation (600 steps hill-climbing) may contaminate extraction experiments | PROVISIONAL (Jun observation, 2026-03-24, no controlled test) |
| PB25 | Hardcoded game coordinates become stale across versions (572u = 0 on LS20/9607627b) | CONFIRMED (Step 1015) |
| PB26 | ℓ_π (encoding modification) > ℓ₁ (parameter modification) on responsive games. 10:1 advantage on same architecture. Prosecution v12 (attention fitness) 100%/60% L1 vs defense v13 (SPSA fitness) 10%/0%. Known in literature as ANIL result (Raghu et al., ICLR 2020) — feature reuse dominates rapid learning. Novel aspect: demonstrated on truly unknown games (no meta-training distribution). | CONFIRMED (Steps 1042-1064, 20+ experiments, 6 defense concessions) |
| PB27 | Opaque games: ~1/3-2/3 of random pool produces zero detectable change from ANY action at any timescale. Related to ExoMDP framework (Efroni et al., ICML 2022) but more extreme — entire observation appears static, not just distractors. Jun reframe (2026-03-25): "not a wall — will help us in the long run." Opaque games = data about substrate limitations, not impossibility. | CONFIRMED (Steps 1042-1064). Jun: treat as signal, not barrier. |
| PB28 | Adaptive cascade = best architecture for unknown games. Short probes (500 steps) → early exit → full budget to detected game type. Multi-timescale change detection (per-step, 5-step, 50-step windows + distribution drift). Related to EPI (Zhou et al., ICLR 2019) and PEARL (Rakelly et al., ICML 2019). Novel: tiered adaptive-depth probing without meta-training. | CONFIRMED (prosecution v10-v12, Steps 1055-1064) |
| PB29 | Jun's seed-protocol insight: substrate IS its own seed protocol. Internal restarts, internal diversity, internal failure evaluation. 10-seed protocol = researcher doing what the substrate should do internally. R1+R2+R3+R4 rolled into one. | DIRECTIVE (Jun, 2026-03-25) |

### Summary

Prediction transfer region non-empty (PB5). Navigation transfer region empty. Gap is structural (Prop 21). Building sprint (Steps 1042-1064) proved ℓ_π > ℓ₁ on responsive games but L2+ = 0 across ALL experiments. The debate's win condition (100% ALL LEVELS on 3 random games, verified 25 times) IS the original search problem.

800b = only post-ban LS20 mechanism. Theorem 4: global running mean SNR → 0 for FT09/VC33. Adaptive cascade (PB28) + multi-timescale detection + attention-weighted fitness = prosecution's best architecture. Opaque games (PB27) are data about substrate limitations, not a wall (Jun, 2026-03-25). Full 150+ games now available — random sampling pool 6x'd.
**Baselines:** 868d = 203.9/seed (true baseline). 916 = 290.7/seed (LS20 SOTA). See RESEARCH_STATE.md for full comparison table.
- **916 = 290.7/seed LS20 SOTA (PB17).** Echo-state reservoir h_t=tanh(W_h@h+W_x@enc), ext_enc=[enc,h]=320D. 895h on extended space. Beats 895h cold (268.0) by +8.5%. Published baselines ALL below 895h: ICM=0 (signal collapses), Count=109, RND=112, Graph+argmin=129.9 (PB18, Steps 917-920). Our mechanism 2-2.5× better.
- **ℓ_π > ℓ₁ CONFIRMED (PB26, Steps 1042-1064).** Same adaptive cascade architecture, different fitness function. Attention-weighted fitness (prosecution) 100%/60% vs SPSA goal-mismatch (defense) 10%/0%. The encoding IS the bottleneck — modifying it (ℓ_π) dominates tuning within it (ℓ₁). Consistent with ANIL (Raghu 2020): feature reuse > rapid learning. Novel: no meta-training distribution.
- **FT09 bottleneck REVISED (PB19, Steps 920/920b/1017 vs 1012).** Generic exploration = 0 even unconstrained. Prescribed solution = 6/6. Variable is discovery mechanism, not encoding or constraints.
- **Constraint cost measured (PB21, Step 1017).** ALL bans lifted = still 0% on click games. Gap is discovery, not constraints. LS20 cost ≈ 0.
- **Graph ban TIGHTENED (2026-03-23, Step 931 killed).** Per-observation-action memory (obs_encoding → best_action) IS per-state conditioning — banned. The observation encoding IS a state representation. ANY mechanism that conditions action selection on specific past observations is a graph in disguise. ONLY global statistics allowed: per-action delta (800b), alpha attention weights. No observation-specific recall of any kind.

---

## Kill Criteria Framework

*Added 2026-03-18 session 2. Separates universal benchmarks from family-specific diagnostics.*

### Universal Benchmarks (same for ALL families)

| Benchmark | Success criterion | Failure criterion |
|---|---|---|
| LS20 Navigation | Level 1 reached | 0 levels at 50K steps (10 seeds) |
| FT09 Navigation | Level 1 reached | 0 levels at 50K steps |
| P-MNIST Classification | >25% AA (above chance), 0pp forgetting | <15% AA or >5pp forgetting |
| Cross-domain | Navigate after classification exposure | Navigation suppressed after cross-domain |

### Family-Specific Diagnostics (different thermometer per patient)

| Family | Health indicators | Death modes |
|---|---|---|
| Codebook (LVQ) | thresh (0.85-0.95), cb_size (growing), dom (<50%), sim_stats | Thresh saturation (1.0), action collapse (dom>90%), Goldilocks zone exit |
| Reservoir (ESN) | trajectory rank (>10), spectral radius stability, action diversity | Rank-1 collapse, W saturation, action lock |
| Graph (cosine nodes) | node count (growing), edge density, unique cells/step | Edge reset, node explosion, cosine-mediated (may inherit codebook death modes) |
| LSH Graph | occupied cells / total cells, edge count growth, hash collision rate | Cell degeneration (1 cell, Step 455), hash collision saturation |
| kd-tree | leaf count, split stability, edge persistence | Edge destruction on split (Step 452) |

### Meta-Rules (universal)

- **Minimum 20 experiments before killing a family.** Codebook got 435. Reservoir got 8. This asymmetry is not acceptable. A family killed in <20 experiments was killed by insufficient exploration, not by evidence.
- **3 consecutive structural failures = one death mode confirmed, not family dead.** Document the death mode. Try a different approach within the family.
- **Kill criteria must be written in the family's own diagnostic language.** "thresh=1.0" means nothing for a reservoir. "rank=1" means nothing for a codebook.
- **The benchmark is universal. The explanation of why it fails is family-specific.**

---

## Builder's Notes (revised 2026-03-18 session 2)

*Original notes from Steps 413-416, updated with cross-family evidence. Some claims were reclassified.*

**~~S21 should be U-class.~~** Reclassified as provisional U15. True for codebook (5 independent kill switches). LSH seems more robust (works across random seeds, different k values). Robustness may be inversely correlated with operating-range narrowness, not a universal property.

**~~Centering is doing more than advertised — potentially U-class.~~** **CONFIRMED as Validated Universal (U16).** Adversarial review confirmed via code review: Step 453 LSH explicitly applies centered_enc (x - x.mean()) before hashing. Centering is load-bearing for BOTH codebook (prevents cosine saturation) AND LSH (prevents hash concentration). Two families, different failure modes, same mechanism required. Original Builder's Note was wrong — LSH does NOT navigate on raw uncentered observations.

**~~Uncapped codebook is the exploration engine — S16 might be U-class.~~** **CONFIRMED as Validated Universal (U17, reworded).** Adversarial review: LSH's edge growth IS unbounded information accumulation. LSH satisfies U17 via edge growth, not entry growth. The principle is "exploration requires unbounded accumulation of SOME kind" — codebook grows entries, LSH grows edges. Both confirmed. U17 coupled to U25: edge growth postpones convergence, but only cell growth prevents it indefinitely.

**Rotation between configurations poisons all of them (S15).** Remains provisional U18. Untested on non-codebook multi-hypothesis systems. The principle (shared action channels contaminate) seems likely to generalize, but evidence is single-family.

**p=0.75 at 4096D ≈ p=1.0 at 256D.** Codebook-specific (S19). Dynamics can be tuned independently of features. Not relevant to non-codebook families.
