# Constraints — Comprehensive Extraction from Experiments

*Revised 2026-03-18 session 2. Major restructure: task requirements first, honest universality reclassification.*

*455 experiments across 6 architecture families (codebook 435, reservoir 8, graph 8, LSH 4, kd-tree 1, CA 3). The constraint map was previously biased toward codebook — 26 "universal" constraints extracted primarily from one family. This revision reclassifies based on cross-family evidence.*

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

Every navigation experiment since Step 442 uses the same graph + edge-count mechanism. The only variable is the observation-to-node mapping. Results partition perfectly by three mapping properties:

| Property | Required? | Evidence |
|---|---|---|
| **Deterministic** (same obs → same node) | HYPOTHESIZED | No family has been shown to fail on determinism alone while satisfying the other two. Reservoir's "temporal inconsistency" (Step 448) is actually a local continuity failure — similar observations map to different cells due to hidden state history, not non-determinism. Zero independent evidence for this axis. |
| **Locally continuous** (nearby obs → same node) | YES | Grid graph fails (0/3, Steps 446-447). LSH succeeds (3/10, Step 453). PCA grid worse than random (539 vs 1869 cells). Reservoir fails (history-dependent mapping, Step 448). CA fails (degenerate mapping, Steps 449-451). 4 families fail on this axis — the strongest evidence. |
| **Persistent** (nodes don't get destroyed) | YES | kd-tree fails (splits destroy edges, Step 452). LSH/codebook cells are permanent → navigate. |

**Navigation also requires a non-convergent exploration mechanism.** Edge-count argmin converges LOCALLY (per-cell ratios approach true visit distribution). Steps 489-492 confirm empirically: at 1M steps on Level 2, cells plateau at 259/4096 — agent cycles forever. Convergence is real but only matters when the reward is beyond the reachable set.

**The 6/10 → 9/10 finding (Steps 484-485):** The apparent 6/10 ceiling on LS20 Level 1 was a STEP BUDGET ARTIFACT. Hard seeds navigate at 35K-115K vs easy seeds at 5-20K. Step 485: 9/10 at 120K. Step 484: 4/4 hard seeds navigate at 200K (tested separately). Projected 10/10 with sufficient budget but not measured in a single run.

**Smart exploration vs uniform exploration (Steps 477-482):** Every TARGETED action selection strategy (destination novelty 1/10, prediction error 0/10, softmax 2/3) performs WORSE than pure argmin. UCB1 degenerates to argmin in practice. One UNIFORM strategy — global cell novelty (Step 482) — MATCHES argmin at 6/10 with complementary seed coverage, suggesting ensemble could reach 8-9/10. Smart (signal-chasing) = worse. Uniform (coverage-maximizing) = comparable.

**Level 2 physical gap (Steps 486-493, 2 families):** Level 2's reward is in a region disconnected from the reachable states. Confirmed across LSH (259 cells, Step 489) and L2 k-means (286 cells, Step 493). Both plateau regardless of budget. Edge manipulation in any direction reduces coverage vs pure argmin. Ordering: argmin(259) > decay(241-243) > death-avoid(227) > death-seek(196). Level 2 requires purposeful exploration (I6/I9) — not a better mapping.

**LLM benchmark (Step 462) — PRELIMINARY, n=1:** 1/1 clean test (haiku on LS20) failed — action collapse (100% ACTION1, 97 steps). 2 tainted results excluded: sonnet cheated (read codebase, uninterpretable), opus tested on FT09 (a game broken for all mechanisms). Insufficient sample for strong claims. HYPOTHESIS: LLMs lack systematic exploration mechanism. Needs repeat with proper isolation before treating as confirmed finding.

**Cross-game — OPEN PROBLEM (Steps 467-469, 476, 495-499):** FT09 and VC33 expose capabilities that LS20 doesn't test. K-means maps FT09 (32 cells) and VC33 (50 cells) perfectly — the MAPPING works. **FT09 is I3-gated:** Step 499 confirmed ACTION6 is complex (requires x,y click coordinates). 5/6 actions produce deterministic transitions; ACTION6 with argmax(frame) click = self-loop. Zero reward without spatial click exploration. Codebook's 69-class encoding solved this by exploring 69 click REGIONS. The substrate must discover that click POSITION matters (I3: action space discovery). **VC33 needs diagnostic:** 1 action, 50 states, likely I4-gated (timing). Uncharacterized with k-means graph.

**Step 493 confirms Level 2 gap across families:** L2 k-means (286 cells) confirms LSH (259 cells) — both plateau, both 0/N Level 2. Growing centroids don't help. The reachable region is topologically bounded regardless of mapping architecture.

**Honest framing:** Local continuity + persistence explain ALL observed Level 1 navigation failures. Level 2 failures are explained by the SPATIAL DISCONNECT between reachable states and reward — a game topology problem, not a mechanism property. The mapping properties are necessary but not sufficient for multi-level navigation.

### Classification (P-MNIST)

Classification requires external labels. Step 432: self-labels = 9.8% (chance), external labels = 94.48%. Confirmed for both codebook and graph substrate (Step 444b: 93.34% with labels, 10.1% without). No substrate has achieved R1-compliant classification. This is an honest admission, not a constraint to work around.

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
| U24 | Exploration and exploitation are opposite operations | Mathematical fact: argmin (least familiar) explores, argmax (most familiar) classifies. No single mechanism serves both. Codebook: argmin = 0% classification (Step 418g). Graph/LSH: edge-count argmin explores but can't classify. Architecture-independent. | Steps 418, 444b |
| U16 | The substrate must encode differences from expectation | Codebook: centered_enc load-bearing — without it, thresh=1.000 and action collapse (Step 414 runs 1-2). LSH: centered_enc applied before hashing (Step 453 code confirmed) — prevents hash concentration. Same mechanism (x - x.mean()), two architecturally distinct families, different failure modes. Centering solves a broader problem than cosine artifacts. | Steps 414, 453. Adversarial review corrected false reclassification. |
| U17 | Exploration requires unbounded information accumulation | Codebook: capped entries → exploration stalls (S16, Step 415a). LSH: fixed cells (1024) but edges grow unboundedly → navigates (Step 454). Both families require SOME form of growth — entries for codebook, edges for LSH. The principle is "unbounded accumulation needed," not "unbounded entry count." Reworded from "fixed-capacity memory exhausts" — LSH's edge growth satisfies the principle via a different mechanism. | S16, Steps 415a, 454. Adversarial review corrected over-demotion. |

### Design Requirements (from Jun, not from experiments)

| # | Constraint | Note |
|---|---|---|
| U4 | Minimal description | Jun's requirement. Complexity = frozen frames. Not experimentally derived, but valid as a design principle. |

---

## Provisional Constraints (single-family evidence — hypotheses to test, NOT laws to enforce)

*These were extracted from codebook experiments only. They may be universal, codebook-specific, or wrong. Each needs testing against non-codebook families before being used to narrow the feasible region. The codebook got 435 experiments; reservoir got 8; LSH got 4. This asymmetry means most "universal" claims are really "true for codebooks, untested elsewhere."*

| # | Constraint | Status | Why provisional |
|---|---|---|---|
| U2 | No separate memory + generation | Untested | LSH has separate components (hash function + edge counts) and navigates. The claim may be too strong — coupled components may suffice. |
| U5 | Sparse selection over global aggregation | Partial | Navigation needs point selection (one action). But reservoir uses global dynamics and fails for rank-1 collapse, not global aggregation per se. Causal link unproven. |
| U8 | Hard selection over soft blending | Partial | Codebook: soft blending → centroid convergence. Both graph + LSH use hard selection. But neural networks use soft operations successfully. May be navigation-specific, not universal. |
| U9 | Curriculum transfer | Codebook-only | Tested only on codebook curriculum. Neural network curriculum shows different patterns. |
| U10 | Dense memory kills exploration | Codebook-only | Codebook at 64x64 memorized 8276/8320 states → no novelty. But the issue may be the encoding (4096D cosine saturation), not memory density. Graph also grows indefinitely and navigates. |
| U12 | Structured noise / Goldilocks zone | Codebook-only | The narrow operating zone is cosine-specific (F.normalize + centered_enc creates the right noise level). LSH has noise through hash collisions — a completely different mechanism. The TASK requires exploration noise, but the mechanism is family-specific. |
| U13 | Additions hurt | Partial | Confirmed for codebook (Steps 343-376) and reservoir (438-439). But both were at their capability ceiling. In neural networks, additions (residual connections, attention) help enormously. May be ceiling-specific, not universal. |
| U14 | Substrate IS its search procedure | Meta-insight | Observation about the codebook's self-similar structure. Interesting philosophically. Not a testable constraint for future substrates. |
| U15 | Robust to perturbation | Design practice | Good engineering. Codebook is brittle (5 independent kill switches, Step 414). LSH seems more robust (works across random seeds). But calling this a "constraint from experiments" overstates — it's a design requirement. |
| U18 | Shared action channels contaminate | Codebook-only | Steps 413-413b tested multi-resolution codebook sharing one action stream. Untested on non-codebook multi-hypothesis systems. |
| U19 | Dynamics ≠ features | Codebook-only | Step 416 (p=0.75): good codebook dynamics, no features at 64x64. Other architectures couple dynamics and features differently. |
| U21 | Diversity fitness collapses | ExprSubstrate-only | One experiment on ExprSubstrate. Too thin to generalize. |
| U25 | Convergent action selection kills exploration | Partial | Edge-count argmin convergence is LOCAL (per-cell), not GLOBAL (all scores). This makes it qualitatively slower than codebook score convergence. But with fixed cell count, local convergence eventually exhausts all cells. Coupled to U17: edge growth postpones convergence, but only cell growth prevents it indefinitely. Confirmed for codebook (Step 428); theoretical concern for edge-count family; untested at >50K steps. |
| U26 | Self-generated labels compound errors | Partial | Codebook: 9.8% self-labels vs 94.48% external (Step 432). Graph: 10.1% vs 93.34% (Step 444b). Both use nearest-neighbor voting. A genuinely different classifier might not have this problem. Confirmed for NN-voting family, not universal. |

---

## Reclassified: Former U-constraints now S-class

*These were labeled "Universal" but are contradicted by non-codebook evidence. They are codebook/cosine-specific properties, not task requirements.*

| Former # | Constraint | Why reclassified | Contradicting evidence |
|---|---|---|---|
| — | ~~U1 reclassification REVERTED~~ | **U1 reworded and remains Validated Universal.** Adversarial review: the principle (no separate learn/infer modes) holds across all families. The original wording ("one operation") was too strong — read/write ARE decomposable sub-steps in LSH. But no family has separate training/inference modes. Reworded to capture the principle, not the mechanism. | Corrected 2026-03-18 session 2. |
| ~~U6~~ → S23 | Similarity-based discovery finds only Lipschitz functions | This is a mathematical property of k-NN, which is codebook-specific. LSH doesn't use k-NN. Graph uses edge counts. Note: LSH's locality-preserving hash IS a form of local similarity (nearby observations hash together with probability proportional to angular similarity). The specific theorem doesn't apply, but the underlying concern (local methods miss global structure) may resurface for LSH. | Irrelevant to non-codebook families as theorem; underlying concern persists. |
| — | ~~U16 reclassification REVERTED~~ | **U16 remains Validated Universal.** Adversarial review (Hart) + Eli confirmed: Step 453 LSH uses centered_enc explicitly (x - x.mean() before hashing). Centering is load-bearing for TWO distinct families: codebook (prevents cosine saturation, Step 414) AND LSH (prevents hash concentration, Step 453). Different failure modes, same mechanism required. Original reclassification was factually wrong — introduced bias by not verifying code before claiming "no centering." | Corrected 2026-03-18 session 2. |
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

## The State of the Search (493 experiments, 8 families)

*Revised 2026-03-18 session 3. Updated with Steps 456-493. FT09/VC33 reframed as open problems.*

### What's Solved

**The navigation mechanism is understood.** Graph + edge-count action selection (pick least-taken action from current node). Confirmed across codebook and LSH families. The mechanism is architecture-independent — any mapping that satisfies the three properties feeds into it.

**The three mapping properties are validated.** Deterministic + locally continuous + persistent. Predicts navigation success/failure with 100% accuracy across all 6 tested families. This is the strongest empirical finding in the project.

**Three working mappings exist for LS20:** codebook (cosine — 6/10 at 50K, banned), LSH (random hyperplanes — 9/10 at 120K), L2 k-means (5/10 at 50K, 3/3 Level 1 at 200K). All satisfy the three mapping properties through different mechanisms. **But ALL only work on LS20.** Zero non-codebook solutions exist for FT09 or VC33.

### What's Open

1. **R3 (self-modification):** No substrate self-modifies its mapping. Codebook attract is prescribed. LSH hyperplanes are fixed. The mapping IS the frozen frame. Making it adaptive without reintroducing banned mechanisms is the core open problem.

2. **FT09 and VC33 — the REAL test:** No non-codebook substrate has navigated either game. FT09 requires discovering click-region actions from raw pixels (I3). VC33 requires temporal reasoning — WHEN to act, not WHERE (I4). These aren't frozen games — they're games that expose what LS20 hides: the need for action space discovery and temporal structure. Codebook solved both only with PRESCRIBED encodings (69-class click for FT09, 3-zone for VC33). The substrate must discover these encodings autonomously.

3. **R1-compliant classification:** No substrate classifies without external labels. Codebook: 9.8% self-labels vs 94.48% external (Step 432). Graph: 10.1% vs 93.34% (Step 444b). This is an honest admission — classification under R1 remains unsolved.

4. **Purposeful exploration (I6, I9):** All successful navigation is stochastic coverage with argmin bias. Not intelligence. Strategic hypothesis testing remains absent.

5. **Temporal reasoning (I4):** No substrate handles temporal structure. Reservoir was the natural candidate but rank-1 collapse killed it in 8 experiments (with codebook-biased metrics — deserves revisit with family-appropriate evaluation).

6. **Transfer (I5) and richer communication (I7):** Untested beyond single-level navigation.

### What's Changed Since Phase 1 Assessment

- "The next substrate is defined by what passes ALL constraints simultaneously" → **Constraint map was biased.** After adversarial review: 2 former U-constraints confirmed S-class (U6, U23), 3 reverted to Validated Universal with better wording (U1, U16, U17), 1 moved to Provisional (U25), 12 others marked provisional. The feasible region is LARGER than we thought, but not as large as the initial compression claimed — the adversarial review caught overcorrection.
- "LVQ is not the atomic substrate" → **The graph + edge mechanism may be.** It's the constant across all navigation successes. The mapping is the variable.
- "Self-modifying metric is needed" → **Yes, but the metric is the MAPPING, not cosine.** R3 requires adaptive mapping. Cosine is one mechanism (banned). Others unexplored.

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

**~~Centering is doing more than advertised — potentially U-class.~~** **CONFIRMED as Validated Universal (U16).** Adversarial review + Eli confirmed: Step 453 LSH explicitly applies centered_enc (x - x.mean()) before hashing. Centering is load-bearing for BOTH codebook (prevents cosine saturation) AND LSH (prevents hash concentration). Two families, different failure modes, same mechanism required. Original Builder's Note was wrong — LSH does NOT navigate on raw uncentered observations.

**~~Uncapped codebook is the exploration engine — S16 might be U-class.~~** **CONFIRMED as Validated Universal (U17, reworded).** Adversarial review: LSH's edge growth IS unbounded information accumulation. LSH satisfies U17 via edge growth, not entry growth. The principle is "exploration requires unbounded accumulation of SOME kind" — codebook grows entries, LSH grows edges. Both confirmed. U17 coupled to U25: edge growth postpones convergence, but only cell growth prevents it indefinitely.

**Rotation between configurations poisons all of them (S15).** Remains provisional U18. Untested on non-codebook multi-hypothesis systems. The principle (shared action channels contaminate) seems likely to generalize, but evidence is single-family.

**p=0.75 at 4096D ≈ p=1.0 at 256D.** Codebook-specific (S19). Dynamics can be tuned independently of features. Not relevant to non-codebook families.
