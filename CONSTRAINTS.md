# Constraints — Comprehensive Extraction from Experiments

*Every constraint below is extracted from experimental failure. Classification:*
- **U (Universal)**: Required by ANY substrate for recursive self-improvement
- **S (Substrate-specific)**: Specific to LVQ/codebook/cosine-based substrates
- **I (Intent)**: What CAPABILITY the constraint reveals is missing — maps to what ARC-AGI-3 tests, not its engineering
- **E (Engineering)**: Implementation detail of current system — NOT carried forward

---

## Universal Constraints (apply to any next substrate)

| # | Constraint | What it means for the NEXT substrate | Source |
|---|---|---|---|
| U1 | Read and write must be one operation | No separate "learn" and "infer" modes. The same operation that reads also writes. | C1, Step 72 |
| U2 | Must not require separate memory + generation | One data structure serves all roles (memory, model, encoding, policy) | C3 |
| U3 | Structural zero forgetting | Learning new things must not degrade old knowledge. Architecturally, not by replay. | C5, Step 65 |
| U4 | Minimal description | The substrate must be expressible in minimal code. Complexity = frozen frames. | C9 |
| U5 | Sparse selection over global aggregation | Averaging over all stored knowledge destroys signal. Selection preserves it. | C15, Step 102 |
| U6 | Similarity-based discovery finds only locally-consistent functions | Any substrate using local similarity will fail on globally-structured problems without additional structure. | C15b, Step 286 |
| U7 | Iteration amplifies dominant features, destroys minority features | Recursive application of the same operation converges to the dominant eigenvector, losing everything else. | C27 |
| U8 | Hard selection over soft blending | Soft averaging destroys the boundaries that encode structure. Hard selection preserves discontinuities. | C18, Step 291b |
| U9 | Curriculum only helps when sub-problems are actual solution steps | Transfer from irrelevant sub-problems hurts. The decomposition must match the solution path. | C16, Step 289b |
| U10 | Dense state memory kills exploration | If the substrate memorizes every state, novelty vanishes and exploration stalls. Must forget or abstract. | C40, Step 389 |
| U11 | Discrimination ≠ navigation | Telling states apart is necessary but not sufficient. The substrate must also choose actions that REACH unseen states. | Steps 388-390 |
| U12 | The substrate works via structured noise | Exploration requires noise (randomness in action selection). Too much noise = random walk. Too little = exploitation trap. The operating zone is narrow. | Steps 408-412, session insight |
| U13 | Additions hurt | Every mechanism added to a working substrate either doesn't help or actively degrades. The substrate resists modification. | Steps 343-376, 412 |
| U14 | The substrate IS its search procedure | The process of finding the substrate (spawn hypotheses, test, select, refine) is structurally identical to the substrate itself. Fixed point. | Session insight, Step 409 |
| U15 | The substrate must be robust to perturbation | If 5 "minor" implementation changes each independently kill the system, it's brittle, not atomic. The next substrate must degrade gracefully, not catastrophically. | Step 414 failure analysis, Eli's S21 |
| U16 | The substrate must encode DIFFERENCES from expectation | Centering (subtracting the mean) is load-bearing because it converts absolute states to deviations. The substrate must represent what's SURPRISING, not what IS. | Step 414 centered_enc analysis |
| U17 | Fixed-capacity memory exhausts exploration | Any substrate with a hard memory limit will eventually run out of novelty. Must grow, forget, or abstract — not just cap. | S16, all capped-codebook experiments |
| U18 | Shared action channels contaminate multi-hypothesis systems | If multiple hypotheses (encodings, strategies) share one action stream, actions chosen by hypothesis A corrupt hypothesis B's learning. Sequential commitment or isolated channels needed. | Steps 413-413b, Eli's S15 analysis |
| U19 | Dynamics and features are independent frozen frames | Healthy codebook dynamics (growth rate, spawn/attract ratio) can be achieved at any dimensionality via metric tuning. But dynamics don't create features. Features require encoding that maps observations to action-relevant structure. Both needed. | Step 416 (p=0.75: good dynamics, no features) |
| U20 | The substrate must be locally continuous in its input-action mapping | Similar inputs must produce similar (or at least consistent) actions. This is the Lipschitz constraint applied to the substrate ITSELF. Without it: no spatial coherence, no navigation, no stable behavior. Hash-based addressing violates this (nearby inputs hash to different cells). Random tree splits violate this (nearby inputs diverge at every branch). Cosine satisfies this by construction. Any non-vector substrate must provide its own local continuity mechanism. | Phase 2 Step 417+ (TapeMachine 35% disc, ExprSubstrate 94% collapse, SelfRef 94% disc) |
| U21 | Fitness functions that reward output diversity without input similarity collapse on structured signal | ExprSubstrate's scoring (diversity x consistency) rewards any split that produces diverse outputs, regardless of whether the split aligns with input structure. On a%b (where feature selection matters most): 46% accuracy = trivial prior (label 0 is 46% of data). Signal dim fraction = 6.9% (chance = 6.2%). The tree splits on noise dims near threshold 0 (appearing diverse during training) but collapses to 1 action on clean evaluation. Without a similarity metric in the state (like SelfRef's cosine), the scoring function cannot distinguish signal-aligned from noise-aligned splits. This kills genetic programming approaches to feature selection under R1. | Phase 2: ExprSubstrate on a%b, 5 seeds. 46% = prior. 0 feature selection. Noise-dependent diversity collapse confirmed. |
| U22 | Fixed-size state that minimizes prediction error converges, and convergence kills exploration | LMS prediction on a shared matrix W converges when pred_err → 0. Any mechanism that stabilizes action selection makes the environment stationary → prediction becomes trivially easy → pred_err → 0 → update = outer(0, prev) = 0 → W freezes → action locks → exploration dies. Growth (codebook spawning, tree expansion) prevents convergence by continuously destabilizing the learned model. The baseline TemporalPrediction survived only because timer-driven action variability kept the environment non-stationary — but this variability was navigation-irrelevant. Row-normalized read: W froze at step 5K, pred_err=0.0000, action locked to 0 for 49620/50000 steps. PredictPersist: unique=1232 at 50K (< baseline 1536 at 10K). Per-action W: unique=991, pred_err=0.07. Every variant that stabilized actions triggered convergence collapse. LVQ avoids this because codebook growth prevents convergence — there is always a new entry disrupting dynamics. | Phase 2: TemporalPrediction family (4 variants, all killed). Row-norm decisive: unique=453 at 50K, W frozen from step 5K. |
| U23 | Distributed updates destabilize inter-entry similarity structure | Updating ALL codebook entries every step (soft attention + distributed error correction) pushes entries toward each observation, making max inter-entry similarity approach 1.0. Gram-based spawn thresholds saturate at 1.0 → 99% spawn rate → codebook explosion. Winner-take-all update preserves codebook structure because non-winners don't move. Error-based spawn (||error|| > median) partially mitigates (50% spawn) but doesn't improve navigation. The distributed update is COMPATIBLE with argmin exploration (hybrid 418f: unique=3248 at 10K with half the codebook) but doesn't add directional signal — same random walk, more efficient. Prior art: Nadaraya-Watson estimator, soft competitive learning. | Step 418 series (6 variants). 418a: Gram explosion (99% spawn). 418e: error spawn works (50%) but no exploration. 418f: hybrid matches baseline at half codebook but same random walk. |
| U24 | Exploration (argmin) and exploitation (argmax) are opposite operations | argmin class scoring picks the least familiar action class → novelty-seeking exploration. argmax picks the most familiar → correct classification. No single action mechanism serves both. A substrate that explores via argmin gets 0% classification accuracy. A substrate that classifies via argmax doesn't explore. The benchmark gate (P-MNIST >25% OR LS20 Level 1) uses OR precisely because one substrate cannot pass both with the same action selection. ReadIsWrite with softmax vote: 88.2% P-MNIST, 0 navigation. ReadIsWrite with argmin: 3248 unique exploration, 0% classification. | Step 418g: 0% accuracy with argmin on P-MNIST. 418b-d: 88% with softmax vote. 418e-f: exploration with argmin/reconstruction. |
| U25 | Score convergence and exploration bias are coupled in argmin/top-K systems | Argmin picks the lowest-scoring class. With raw scores, large classes score high → argmin avoids → explores. ANY normalization (count^p for p>0) penalizes large classes → their scores drop → argmin picks them → positive feedback → one class dominates. The transition is binary (p=0 vs p>0), not gradual. No scoring formula fixes the navigation wall. Navigation speed is determined by encoding quality (FT09: 82 steps with right encoding vs LS20: 26K with generic encoding, 300x). | Steps 426-430: softmax hurts (426), full norm inverts (429), fractional norm (p=0.25/0.5/0.75) all collapse to dom=100% (430). 60 prior action experiments (354-416) also failed. |

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

## The Gap (what the next substrate must close)

The 414 experiments prove: LVQ/codebook with cosine similarity is NOT the atomic substrate. It is a well-known algorithm (LVQ 1988, Growing Neural Gas 1995) that works within a narrow operating range (the Goldilocks zone) and requires prescribed encoding to reach that range.

**What the next substrate must have that LVQ doesn't:**
1. **Self-modifying metric** (I2, I8): the comparison operation adapts based on what works, not hardcoded cosine
2. **Representation discovery** (I1): the state encoding emerges from interaction, not from prescribed preprocessing
3. **Purposeful exploration** (I6, I9): strategic hypothesis testing, not stochastic coverage
4. **Temporal reasoning** (I4): sequence-aware, not just state-aware
5. **Transfer** (I5): knowledge carries across episodes/tasks
6. **Richer communication** (I7): expressive output, not just one integer

The constraint map is the specification. The next substrate is defined by what passes ALL of these simultaneously.

---

## Builder's Notes (from running Steps 413-416)

**S21 should be U-class.** The substrate's behavior is extremely sensitive to implementation details. Five "minor" differences from the baseline each independently kill the system. This isn't LVQ-specific — any substrate with tight operating margins will have this property. The real constraint: **the substrate must be robust to perturbation, not brittle to implementation detail.**

**Centering is doing more than advertised.** centered_enc doesn't just "prevent saturation." It makes the codebook into a differential encoding — each vector represents deviation from the mean. Without it, all resolutions fail. With it, 16x16 works. This suggests the substrate needs to encode DIFFERENCES from expectation, not absolute states. Potentially U-class.

**The uncapped codebook is the exploration engine.** Every capped run failed. The baseline needs to grow past 10K entries to find a level at 26K steps. The codebook IS the exploration — each new entry is a hypothesis about what's novel. Capping = capping exploration budget. S16 might be U-class: **any substrate with fixed-capacity memory will eventually exhaust its exploration.**

**Rotation between configurations poisons all of them (S15).** When codebooks learn from actions chosen by a different resolution's policy, they accumulate incoherent experience. This is broader than LVQ: any multi-hypothesis system where hypotheses share an action channel will have this contamination problem. Sequential dedication (each hypothesis gets full commitment) works. Potentially U-class.

**p=0.75 at 4096D ≈ p=1.0 at 256D in dynamics.** Partial normalization creates the same codebook growth pattern and sim statistics as full cosine at lower dimensionality. But no level found. The Goldilocks zone is necessary but not sufficient — you also need the right encoding to map visual states to action-relevant features. Partial normalization creates the dynamics; avgpool creates the features. Both are needed.
