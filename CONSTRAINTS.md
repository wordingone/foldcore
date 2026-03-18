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
| **Deterministic** (same obs → same node) | YES | Reservoir fails (temporal inconsistency, Step 448). All successes are deterministic. |
| **Locally continuous** (nearby obs → same node) | YES | Grid graph fails (0/3 at 30K, Steps 446-447). LSH succeeds (3/10, Step 453). PCA grid worse than random (539 vs 1869 cells). |
| **Persistent** (nodes don't get destroyed) | YES | kd-tree fails (splits destroy edges, Step 452). LSH/codebook cells are permanent → navigate. |

**Navigation also requires a non-convergent exploration mechanism.** If action selection converges (codebook score convergence U25, prediction error convergence U22), exploration dies. Edge-count-based selection (least-taken action from current node) does not converge — counts grow monotonically, maintaining signal.

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
| U24 | Exploration and exploitation are opposite operations | Mathematical fact: argmin (least familiar) explores, argmax (most familiar) classifies. No single mechanism serves both. Codebook: argmin = 0% classification (Step 418g). Graph/LSH: edge-count argmin explores but can't classify. Architecture-independent. | Steps 418, 444b |

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
| U26 | Self-generated labels compound errors | Partial | Codebook: 9.8% self-labels vs 94.48% external (Step 432). Graph: 10.1% vs 93.34% (Step 444b). Both use nearest-neighbor voting. A genuinely different classifier might not have this problem. Confirmed for NN-voting family, not universal. |

---

## Reclassified: Former U-constraints now S-class

*These were labeled "Universal" but are contradicted by non-codebook evidence. They are codebook/cosine-specific properties, not task requirements.*

| Former # | Constraint | Why reclassified | Contradicting evidence |
|---|---|---|---|
| ~~U1~~ → S22 | Read and write must be one operation | LSH separates hash-read and edge-write. Graph separates node-lookup and edge-update. Both navigate. | LSH: 4/10 at 50K (Step 454). The TASK requires online learning, not one-operation — a weaker claim. |
| ~~U6~~ → S23 | Similarity-based discovery finds only Lipschitz functions | This is a mathematical property of k-NN, which is codebook-specific. LSH doesn't use k-NN. Graph uses edge counts. | Irrelevant to non-codebook families. |
| ~~U16~~ → S24 | Must encode differences from expectation | Centering prevents cosine saturation in high-D codebooks. LSH navigates on RAW (uncentered) observations hashed through random hyperplanes. | LSH Step 453: no centering, still navigates. Centering is a cosine-space requirement. |
| ~~U17~~ → S25 | Fixed-capacity memory exhausts exploration | LSH has FIXED capacity (1024 cells) and navigates 4/10 at 50K. The codebook needs unbounded growth because its exploration depends on spawning. LSH exploration depends on edge-count diversity, not growth. | LSH Step 454: 1024 fixed cells, 4/10. Directly contradicts. |
| ~~U23~~ → S26 | Distributed updates destabilize similarity structure | Specific to codebook Gram matrix structure. Updating all entries pushes them toward each observation. Neural networks use distributed updates (backprop) successfully. Edge-based systems don't have this problem. | Codebook-specific mechanism. Step 418 series. |
| ~~U25~~ → S27 | Score convergence and exploration bias are coupled | Specific to argmin/top-K scoring on class counts. Edge-count-based action selection (graph/LSH) doesn't suffer this — edge counts grow monotonically, maintaining relative differences. | Graph/LSH navigate without top-K scoring. Steps 426-430 are codebook-specific. |

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

## The State of the Search (455 experiments, 6 families)

*Revised 2026-03-18 session 2. Framed by task requirements, not architecture gaps.*

### What's Solved

**The navigation mechanism is understood.** Graph + edge-count action selection (pick least-taken action from current node). Confirmed across codebook and LSH families. The mechanism is architecture-independent — any mapping that satisfies the three properties feeds into it.

**The three mapping properties are validated.** Deterministic + locally continuous + persistent. Predicts navigation success/failure with 100% accuracy across all 6 tested families. This is the strongest empirical finding in the project.

**Two working mappings exist:** codebook (cosine similarity — 6/10 at 50K, banned) and LSH (random hyperplanes — 4/10 at 50K, legal). Both satisfy all three mapping properties through different mechanisms.

### What's Open

1. **R3 (self-modification):** No substrate self-modifies its mapping. Codebook attract is prescribed. LSH hyperplanes are fixed. The mapping IS the frozen frame. Making it adaptive without reintroducing banned mechanisms is the core open problem.

2. **Cross-game generality:** LSH degenerates on FT09 (1 cell, Step 455). k=10 too coarse. The codebook's adaptive density (attract concentrates entries where observations cluster) handles this automatically. Property 3 (adaptive density) is NOT needed for single-game navigation but IS needed for cross-game generality.

3. **R1-compliant classification:** No substrate classifies without external labels. Codebook: 9.8% self-labels vs 94.48% external (Step 432). Graph: 10.1% vs 93.34% (Step 444b). This is an honest admission — classification under R1 remains unsolved.

4. **Purposeful exploration (I6, I9):** All successful navigation is stochastic coverage with argmin bias. Not intelligence. Strategic hypothesis testing remains absent.

5. **Temporal reasoning (I4):** No substrate handles temporal structure. Reservoir was the natural candidate but rank-1 collapse killed it in 8 experiments (with codebook-biased metrics — deserves revisit with family-appropriate evaluation).

6. **Transfer (I5) and richer communication (I7):** Untested beyond single-level navigation.

### What's Changed Since Phase 1 Assessment

- "The next substrate is defined by what passes ALL constraints simultaneously" → **Constraint map was biased.** 6 former U-constraints reclassified to S-class. 13 marked provisional. The feasible region is LARGER than we thought — the constraints were overconstrained by codebook assumptions.
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

**~~Centering is doing more than advertised — potentially U-class.~~** **Reclassified to S24.** LSH navigates on raw (uncentered) observations (Step 453). Centering prevents cosine saturation — a codebook-specific problem. The deeper truth may be that the mapping needs local continuity (U20), and centering is one way codebooks achieve it. But it's not the only way.

**~~Uncapped codebook is the exploration engine — S16 might be U-class.~~** **Reclassified to S25.** LSH navigates with FIXED 1024 cells (Step 454). The codebook needs unbounded growth because its exploration mechanism depends on spawning novelty. LSH's exploration depends on edge-count diversity in persistent cells, not growth. Growth is a codebook exploration mechanism, not a universal requirement.

**Rotation between configurations poisons all of them (S15).** Remains provisional U18. Untested on non-codebook multi-hypothesis systems. The principle (shared action channels contaminate) seems likely to generalize, but evidence is single-family.

**p=0.75 at 4096D ≈ p=1.0 at 256D.** Codebook-specific (S19). Dynamics can be tuned independently of features. Not relevant to non-codebook families.
