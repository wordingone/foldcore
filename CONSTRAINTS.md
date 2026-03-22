# Constraints — Comprehensive Extraction from Experiments

*Revised 2026-03-18 session 2. Major restructure: task requirements first, honest universality reclassification.*

*612+ experiments across 12 architecture families (codebook ~435, LSH ~80, L2 k-means ~25, reservoir ~20, Hebbian 3, SplitTree 5, Recode 3, Absorb 3, graph 8, CA 3, Bloom 2, CC 1). Codebook banned 2026-03-18. Key findings: algorithm invariance (argmin across 4+ representations), centering load-bearing (U16, Step 593: 222→5 cells without it), speed not rate (Step 584: 13/20=13/20 at 50K; Step 594: random 10/20 vs argmin 13/20 NS), domain isolation necessary (Step 596), Recode K confound (Step 589: ℓ_π = ℓ₀ at same K).*

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

**Algorithm invariance (Steps 521, 524, 525).** The action-selection mechanism IS argmin over visit frequency, regardless of representation. Tested across 4 representations: edge dict (LSH), weight matrix (Hebbian), transition tensor (Markov), n-gram history. All produce equivalent navigation. The mechanism is the invariant; the data structure is a degree of freedom. (Recode also uses argmin but was not designed as an invariance test.)

**Non-convergent exploration.** Edge-count argmin converges LOCALLY (per-cell ratios approach true visit distribution). But the reachable set continues growing sublinearly: 259 cells at 50K → 439 at 740K (Steps 528-529). Growth rate decays to ~2 cells/100K at 740K; whether a true asymptote exists is unmeasured. The prior "259 ceiling" was a TIME LIMIT, not topology. At k=16, reachable set expands to 1094 cells at 200K (Step 531), 1149 at 500K (Step 532).

**Self-refinement and navigation reliability (Step 542, Recode).** LSH k=16 + passive self-refinement from transition statistics = 5/5 L1 on LS20 (vs 3/3 for LSH k=16 alone at 200K, Step 531). The observation→cell mapping is self-modified: when a cell contains observations that lead to different transition outcomes, the substrate learns a hyperplane that separates them. Cell count: 1267 vs 1149 without refinement at k=16 (+10%). The dominant factor in reachable set expansion is k (partition granularity); refinement's contribution is modest in cell count but may improve NAVIGATION RELIABILITY (5/5 at 500K vs 3/3 at 200K — budget contribution not isolated). This is the first substrate where the mapping improves from its own dynamics AND navigation succeeds reliably.

**The 6/10 → 9/10 → 5/5 progression:** 6/10 at k=12, 50K (Step 459) was BUDGET-DEPENDENT. 9/10 at k=12, 120K (Step 485). 3/3 at k=16, 200K (Step 531). **5/5 at k=16 + self-refinement, 500K (Step 542, Recode).** Navigation reliability scales with k, budget, and mapping quality.

**Targeted exploration kills navigation (Steps 477-482, 539-541).** Every TARGETED action selection strategy performs WORSE than pure argmin. Destination novelty 1/10, prediction error 0/10, softmax 2/3, entropy-seeking 0/3 (noisy TV problem — Burda et al. 2018). UCB1 degenerates to argmin. Entropy-seeking drives the agent into lethal states because death transitions have maximum irreducible entropy. **6 independent strategies tested, all worse than argmin.** This is the noisy TV problem applied to navigation: targeted exploration finds noise (death), not signal (reward).

**Level 2 reward disconnect — 4 hypotheses falsified on LSH+LS20 (Steps 486-639).** Level 2's reward is in a region the agent hasn't reached at 740K steps with 439 cells (k=12) or at 500K with 1149 cells (k=16). Four independent hypotheses tested and falsified on the same substrate+game: (1) **Scale/budget** (486-492): edge manipulation, death avoidance/seeking, stochastic exploitation, finer partitioning — reachable set grows sublinearly but L2 remains disconnected. (2) **Self-reference** (620-629, eigenform): self-observation is reflective only — computes functions of visited states, can't reach unvisited states. (3) **Information availability** (630-637, 8 experiments): graph-derived signals at any granularity — k=16 too fine, k=4 too coarse, binary over-collapses, per-edge stale 12-54% inert. (4) **Signal source** (638-639): sparse environmental signals (death ~1.3%, large_diff ~5.2%) are seed-dependent — benefit depends on game topology per seed. All converge: L2 requires forward prediction, not retrospective penalty (Proposition 13). **Caveat:** all 4 hypotheses tested on LSH+LS20 only. The L2 wall may be substrate-specific (LSH hash partitioning) or game-specific. Cross-family L2 testing is needed before claiming generality.

**LLM benchmark (Step 462) — PRELIMINARY, n=1:** 1/1 clean test (haiku on LS20) failed — action collapse (100% ACTION1, 97 steps). 2 tainted results excluded: sonnet cheated (read codebase, uninterpretable), opus tested on FT09 (a game broken for all mechanisms). Insufficient sample for strong claims. HYPOTHESIS: LLMs lack systematic exploration mechanism. Needs repeat with proper isolation before treating as confirmed finding.

**Cross-game (Steps 467-469, 476, 495-503):** FT09 and VC33 expose a key insight: **the action space IS a variable, not just the mapping.**

**FT09 SOLVED (Step 503): 3/3 wins.** K-means (32 cells) + 69-action graph (64 click positions + 5 simple) + argmin = systematic action coverage = win. Seed 2 won at step 157, seed 1 at 2035, seed 0 at 3840. No codebook learning required. Step 501 failed because it treated 64 click positions as ONE action (ACTION6). Step 503 treats each click position as a SEPARATE action — argmin covers them all systematically. The codebook's FT09 win (Step 82) was entirely about action decomposition, not cosine learning. **Confirmed across 2 families (codebook + k-means graph).**

**VC33 SOLVED (Step 505): 3/3 wins.** Zone discovery at 4x4 stride found 3 zones: 254 normal positions + 2 magic pixels at (62,26) and (62,34). 3-action k-means argmin: seed=2 won at step 18, seed=0 at 82, seed=1 at 144. Step 504 (8x8 stride) failed — too coarse, only 1 zone. The magic pixels require pixel-precise discovery. **Confirmed across 2 families (codebook + k-means graph).**

**ALL 3 GAMES LEVEL 1 SOLVED.** Unifying mechanism across 505 experiments: graph + edge-count argmin + correct action decomposition. The mapping (how observations become nodes) and the action space (how actions are decomposed) are the VARIABLES. The graph + argmin is the CONSTANT. LS20 needs 4 directional actions. FT09 needs 69 (64 grid + 5 simple). VC33 needs 3 (zone discovery). All prescribed. The frontier is: can the substrate DISCOVER the right action decomposition autonomously (I3)?

**Level 2 reward disconnect.** Dedicated L2 investigations (LSH Steps 486-493/528-529/532, k-means Step 493) confirm: L2's reward region is beyond the argmin-reachable frontier at all tested budgets and partition granularities. Incidental confirmation in codebook (Phase 1) and Recode (Step 542, 0/5 L2). The reachable set grows sublinearly but never includes L2 reward. Growing the mapping (more cells, finer partition, self-refinement) does not unlock L2.

**Centering and domain separation (Steps 543-544, 546).** Centered encoding is REQUIRED for navigation (uncentered: 62 cells, 0/5 — Step 544). Global centering kills domain separation for the chain (CIFAR and LS20 hash to shared nodes — Step 543; CIFAR accuracy 15%, 15x above chance, showing encoding has cross-domain class signal). **RESOLVED via per-domain centering (Step 546):** reset the running mean on domain switch (on_reset). Per-domain centering gives 2/3 L1 on the chain (vs 5/5 clean — partial reliability gap remains), with s0 navigating FASTER than clean (L1@12201 vs 29691). R1-compliant: on_reset is a game event, not an external domain label.

**R3 barrier precisely located (Steps 577b-577d, 2026-03-20).** Evolutionary approach (577b) fails: 0/1000 random programs have the required pipeline sequence (threshold→label_cc→nearest→navigate). No fitness gradient. Multi-buffer approach (577c): 4/5 L1 with 80%argmin+20%buffer, but argmin is doing the work — not buffer navigation. Windowed evaluation (577d): 100% buffer navigation in 500-step windows, 5000-step warmup. Result: 0/5 L1, 0 L1s across 390 windows × 5 buffers. Buffers visit MORE cells than argmin (exploration rate up) but the WRONG cells. Pixel statistics (mode/mean/var/diff/min) navigate toward visually salient features, not exit-relevant features. **R3 barrier: the substrate needs reward-correlated features, not visual statistics. Without a feedback loop linking navigation strategy to L1 outcomes, no pixel statistic discovers the exit.**

**Chain benchmark: cross-domain contamination tested (Step 577 chain, 2026-03-20).** Full sequence: CIFAR-100 (P1, 1K steps) → LS20 navigation → CIFAR-100 (P3, 1K steps). Result: CIFAR P1=1.02%, P3=1.02%, delta=0.00pp. **No contamination confirmed.** The per-domain centering fully isolates CIFAR classification from LS20 navigation — LS20 edges don't corrupt CIFAR nodes because of the separate edge dicts (G for LS20 nav, G_cif for CIFAR) and per-domain mean reset. Caveat: L2 was not reached (time cap hit at 572s with only 1 L1 cycle), so full chain (P1→L1→L2→CIFAR→L1_delta) was not measured. No contamination confirmed for the partial chain.

**Game version changes (confirmed 2026-03-22, Jun).** Games DO change. LS20: cb3b57cc → 9607627b (different sprite definitions). FT09: also changed. VC33: unchanged. Pre-change and post-change results are on DIFFERENT games — both valid tasks. The substrate must win ANY version of ANY game (Jun's directive). Game change = new task, not baseline invalidation. All Steps 690+ verified on current versions (LS20/9607627b, FT09/0d8bbf25). Pre-690 results on LS20/FT09 are on prior versions and not directly comparable.

**Self-observation mechanism (Step 620, eigenform, 2026-03-21).** Eigenform self-observation on L1: substrate reads its own edge count distribution every 500 steps, computes percentile thresholds, assigns AVOID/PREFER/NEUTRAL op codes. L1=5/5 (matches baseline — no performance improvement). Op distribution: 94-99% NEUTRAL, 0-2% AVOID, 1-10% PREFER. AVOID grows over time (0% → 8-10% across all seeds). **Mechanism produces non-zero output. No performance effect detected on L1.** Whether AVOID growth is meaningful self-calibration or a statistical artifact (percentile calculation over accumulating distribution → tails grow mechanically) is unknown until tested on L2+ where argmin fails and trap avoidance could matter. Combined with 582 (event-driven ops, SIGNAL 4/5): two independent R3 approaches both produce non-zero output.

**L1 as hidden-state conjunction — the POMDP reframing (Steps 652-667, 2026-03-22).** L1 is NOT an exploration problem — it is a recognition problem. The agent visits the exit cell avg 152 times before L1 triggers (Step 652). First visit occurs at steps 37-82. The bottleneck is being at the exit cell WITH the right hidden state, not reaching the exit cell.

LS20 is a POMDP: hidden state variables (snw, tmx, tuv) are inaccessible to the substrate (Step 654). L1 triggers when a hidden-state conjunction is met at the exit cell. The substrate can't observe the conjunction condition directly.

**Key findings:**
- **Graph ≈ flat counter (Step 660).** 15/20 seeds: flat counter (per-cell action counts, no transitions) = full graph. Transitions rarely decisive for argmin — 1/20 seeds (s10) where transitions help, 3/20 where transitions hurt (flat counter better), 15/20 identical. Inference: the graph's primary value is cell persistence (U3), not topology.
- **Graph hurts random walk (Step 663).** Graph memory reshapes exploration paths. Pure random: 4/20. Graph+random: 2/20. Hash refinement interferes with lucky random trajectories.
- **Interpreter is ONE coupled operation (Step 658).** Decoupling compare from select (store at frame cell, select from delta cell) kills navigation completely (0/10). The three-operation decomposition (compare-select-store) is a theoretical artifact.
- **Argmin ≈ random for L1 (Step 653).** Argmin-only: 3/20. Random-only: 3/20. Both: 1/20. Perfectly symmetric — argmin and random each unlock different hidden-state conjunctions. L1 action sequences are near-uniform (H=1.948/2.0, Step 657).
- **Hash resolution strongly predicts L1 speed (Steps 664-665).** BIMODAL: fast seeds (s3,s4,s6) have exit cell entropy ~2.0 (hash resolves hidden states, L1 in <1000 steps). Slow seeds (s8,s9) have exit cell entropy 0.0 (hash conflates ALL hidden states, L1 at 22K-24K steps). Hidden state IS visible in frames (two clusters at exit cell: 227 vs 194, center distance 0.612), but k=12 LSH hashes all visits to same cell. **The POMDP is opaque at the hash level, not the pixel level** (cf. perceptual aliasing, Whitehead & Ballard 1991; hash granularity, Tang et al. 2017). Causal confirmation not yet established — correlation only.
- **Outcome conditioning right concept, wrong granularity (Step 667).** (cell, prev_outcome) tuples create combinatorial explosion — most keys visited once. The arrival path DOES carry hidden-state information, but raw tuple keys are too sparse.

**Out-of-framework result (Step 662).** Evolutionary sequence search (R3-violating: mutation operator frozen) achieves L1 7/10 on LS20 — NO graph, NO hash, NO perception. Fitness = L1/death (R1-compliant). 58 unique sequences, 595 episodes. Budget not directly comparable to graph argmin. This demonstrates L1 is solvable by brute-force sequence search, but does not inform constraints on R1-R6-compliant substrates. The three mapping properties remain necessary within the graph/LSH framework.

**POMDP intervention results (Steps 668-671):**
- **Visit-count belief (668).** MARGINAL 5/10. Random break at >=20 visits disrupts argmin gradient. Lost 3 baseline seeds. Signal real, intervention too coarse.
- **Gaussian variance (669).** MARGINAL 5/10, 145x speedup on successes. Variance identifies exit cell for fast seeds (s3: 285x). But too many hi_var cells for other seeds → floods argmin with novelty.
- **Alternating argmin/random (670).** 5/20. Finds 3 NEW seeds (2,5,18) that neither pure approach found — temporal interleaving creates a genuinely different exploration regime. But loses 5 union seeds. Budget-limited at 5s cap.
- **Splatter world model (671).** KILL 0/10. Noisy TV: death = max frame difference. Same wall as Steps 477-482. Prospective prediction fails; retrospective avoidance (581d) works.

**Pattern across all interventions (Steps 668-670):** Every mechanism helps some seeds and LOSES others. The intervention disrupts argmin's gradient for seeds where argmin was already succeeding. No intervention captures the full union of argmin + random seeds.

**Selective π-refinement series (Steps 672-711, 40 experiments):**

**Proposition 15 CONFIRMED:** Perception quality (π-refinement) IS the L1 lever. Step 674 (transition-triggered dual-hash: k=12 coarse + k=20 fine at cells with inconsistent transitions, |successor_set|≥2, min_visits=3). Conceptually related to CEGAR (Counterexample-Guided Abstraction Refinement) for POMDPs — transition inconsistency = counterexample showing the hash is too coarse. Our implementation is a concrete instantiation for LSH navigation.

**Complete cross-game characterization (Steps 690-711, all verified on current game versions):**

| Game | Method | Budget | L1 | vs plain | Step |
|------|--------|--------|----|----------|------|
| LS20 | Plain k=12 | 25s | 11/20 | — | 697 |
| LS20 | 674 frame-local | 25s | 17/20 | +6 | 690 |
| LS20 | 674 running-mean | 25s | **20/20** | +9 | 705 |
| LS20 | Plain k=12 | 120K | 16/20 | — | 701 |
| LS20 | 674 frame-local | 120K | 20/20 | +4 | 699 |
| LS20 | 674 running-mean | 120K | 20/20 | +4 | 708 |
| LS20 | 674 chain | 120K | 20/20 | +4 | 700 |
| FT09 | Plain k=12 | 25s | 8/20 | — | 711 |
| FT09 | Plain k=12 | 120K | 8/20 | — | 706 |
| FT09 | 674 frame-local | 120K | 17/20 | +9 | 702 |
| FT09 | 674 running-mean | 120K | **20/20** | +12 | 709 |
| VC33 | 674 any centering | 25s | 0/5 | — | 681/707 |

**Key structural findings:**
- **Plain k=12 has a hard ceiling.** LS20: 16/20 at 120K — 4 seeds never solve (s2,s9,s11,s17). FT09: 8/20 — identical 8 seeds at 25s AND 120K (binary — budget irrelevant, Step 711).
- **674 provides genuine COVERAGE improvement**, not just speed. Rescues seeds plain k=12 never solves at any budget.
- **Running-mean centering universally rescues deterministic seeds.** FT09 s11,s13,s18 had aliased=0 under frame-local (674 mechanism never engaged). Running-mean creates aliasing even on previously frozen seeds (0→2-3 aliased cells, Step 709). Running-mean = universal mechanism activation.
- **Running-mean 674 = 20/20 on BOTH games** at sufficient budget. Best short-budget: 20/20 at 25s on LS20.
- **Capping HURTS** (674b/c/d all 8/10, 695 freeze=8/10, 696 death=6/10). All modifications worse than uncapped.
- **VC33 = zone discovery failure**, not aliasing. 674 irrelevant — VC33 needs room/zone mapping (I3).

**Series ranking (original 10-seed sweep):** 674 (transition-triggered) 9/10 > 673 (entropy) 8/10 = 677 (multi-res vote) 8/10 > 675/678/679 all 7/10 > 672/676 5/10.

**L1 vs L2 asymmetry (Steps 682-686).** L1 aliasing is BOUNDED — seed 8 has 87 aliased cells, sufficient for count accumulation. L2 aliasing is UNBOUNDED — aliased cells grow monotonically (2→439 over 572K steps, seed 8; 48→445, seed 3). The transition-triggered mechanism detects new aliased cells post-L1 (dynamic adaptation), but the cell count never plateaus. Fine graph becomes too sparse for argmin when 400+ cells compete for budget. **L2 requires navigation DESPITE unresolved aliasing, not resolution of all aliasing before navigation.** This connects to the energy mechanic (iri sprites, Steps 556-557) — L2 requires purposeful energy-seeking, not exhaustive coverage.

**Implications for the constraint map:**
1. **Hash resolution at exit cell CAUSALLY determines L1 speed** — upgraded from "correlation only." Step 674's 9/10 via targeted resolution increase at aliased cells confirms the causal mechanism. Fast seeds: aliased<120 (bounded, counts accumulate). Slow seeds: aliased>170 (sparse, counts diluted).
2. **Argmin is locally optimal for L1** within the graph framework. 35+ experiments across 8 intervention types (477-482, 581d, 620-639, 652-686) — only π-refinement (not g-modification) consistently improves L1.
3. **Prospective prediction = noisy TV** for navigation (477-482, 671). Retrospective avoidance works sparsely (581d). Confirmed on LS20.
4. **L1 and L2 have qualitatively different walls.** L1 = bounded aliasing (solvable by π-refinement). L2 = unbounded aliasing + energy mechanic (requires purposeful navigation). L2 wall confirmed UNIVERSAL across games: LS20 L2=0/20 with high aliasing (Steps 686-689, 692, 699), FT09 L2=0/5 with ZERO aliasing (Step 691: aliased cells frozen at 1-4 post-L1 over 1.9M steps). L2 is NOT about disambiguation capacity — FT09 is fully disambiguated and still can't reach L2.
5. **Centering is an active variable, not just preprocessing (Step 698).** Running-mean vs frame-local centering changes which cells get aliased. s4: frame-local→NO_L1 (189x regression), running-mean→L1@15544 (rescued), chain (running-mean + CIFAR pre-pop)→L1@2002 (12.5x faster than standalone). Two separable effects: centering type changes hash distribution; CIFAR pre-population seeds aliased cells before LS20 starts. Connects to U16 (centering load-bearing) and I1 (representation discovery).
6. **674 is a trade, not a uniform improvement.** Helps seeds where exit-cell aliasing is the bottleneck (s8: 198x faster). Hurts seeds where exit cell is already resolved (s4: 189x slower on current game). Net: +6/20 at 25s. Chain context reverses: s4 12.5x faster in chain, s8 158x slower.

**Honest framing (revised 2026-03-22, post-711):** L1 is SOLVED for LS20 and FT09 within the graph+LSH framework. Running-mean 674 = 20/20 on both games. The mechanism: transition-inconsistency identifies cells where the hash conflates hidden states; fine hash resolves the ambiguity; running-mean centering ensures the mechanism engages even on deterministic seeds. L2 remains a qualitatively different wall — 0/20 on both games regardless of aliasing dynamics (LS20 high-aliasing, FT09 zero-aliasing). VC33 requires zone discovery (I3), not disambiguation. **Cross-game verification complete:** LS20/9607627b (Steps 690-712), FT09/0d8bbf25 (Steps 680, 702, 706, 709, 711). **Audit queue (027a5b0):** FT09 re-verified on current version. Pre-change results (572u L3=5/5, 610 VC33 7 levels, 572j L2=5/5) remain flagged — on prior game versions.

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
| — | ~~U27 attractor hypothesis~~ | **DOWNGRADED to hypothesis** | Hart debate Round 2 (2026-03-21): the 477-482 evidence was corrected to "mechanism-specific, not categorical" — which undermines using the same evidence for a categorical attractor pattern. Moved to What's Open item 3 as interpretive lens, not constraint. |
| U28 | No auxiliary signal improves argmin; dense signals damage it | **PROVISIONAL** | LSH only (Steps 581d, 630-639, 12 experiments). No signal at any density significantly improves argmin navigation (581d at n=20: p=0.63, Step 584). Dense signals (>~60% firing rate) actively damage navigation: delta PREFER at 66% → 3/5 seeds 10-18x slower (633); frontier at 94-98% → 3/5 seeds 5-20x slower (635). Sparse/moderate signals (<~55%) are inert: stale at 12% → 0.94x (637); delta at 33-42% → identical to argmin (634); stale at 54% → inert (636). Environmental events (1-5%) seed-dependent (639). Transition from inert to damage between 54% and 66%. Hart debate (2026-03-21): original ~10% threshold was post-hoc fitting to 581d benefit claim, which was debunked at n=20. Single-family evidence (LSH). |
| — | PENALTY=100 and op-code definitions are frozen value judgments | **R3/R4 AUDIT QUEUE** | Step 581d: PENALTY=100 is designer-prescribed. Fails R3 (not adaptive) and R6 (removable — 4/5→3/5, not all capability lost). Should be made adaptive: the graph already encodes death information (death edges lead to start node), so the substrate COULD discover appropriate avoidance without a prescribed magnitude. Step 582: op-code assignment is self-derived (R3 partial), but op definitions (what op2 and op3 DO) remain designer-prescribed. Hart debate (2026-03-21) identified these as R4 violations, not R1 violations. |

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

## The State of the Search (612+ experiments, 12 families)

*Revised 2026-03-21 session. Updated with Steps 494-612. 16 levels via source analysis (R3 specification). GRN killed (607). LS20 L4 bootstrap in progress (612).*

### What's Solved

**The navigation mechanism is understood.** Graph + edge-count argmin. Confirmed across codebook, LSH, k-means, Hebbian families. Architecture-independent — any mapping satisfying three properties feeds into it. Argmin advantage over random is speed, not exclusive access (13/20 vs 10/20, Step 594, NS).

**The three mapping properties are validated.** Deterministic + locally continuous + persistent. Predicts L1 success/failure with 100% accuracy across 6 tested families.

**All 3 games solved at multiple levels.** LS20 L1-L3=5/5 (mgu pipeline, Steps 572j-572u). FT09 all 6 levels, 75 clicks deterministic (Step 608b). VC33 all 7 levels, 176 clicks, analytical BFS (Step 610). **16 total levels — but all via source analysis (prescribed pipelines). These are R3's SPECIFICATION: 16 concrete test cases for what an R3-compliant substrate must discover autonomously.**

**Cross-game detection generalizes.** Mode map + isolated CC pipeline discovers interactive objects across all 3 games without game-specific tuning (Step 576). FT09 trivially solvable by action coverage alone (Step 575).

**Recode achieves ℓ_π.** LSH k=16 + passive self-refinement = 5/5 L1 on LS20 (Step 542). First substrate where the observation→cell mapping self-modifies AND navigation succeeds. But K confound (Step 589): Recode(K=16) = LSH(K=16) 18/20. Self-modification provides mid-budget speed advantage only.

### What's Open

1. **R3 (self-modification of operations):** The frozen frame and navigation capability are structurally coupled (R3_AUDIT.md). R3 audit Rounds A-B (2026-03-17, codebook-specific) found: most "unjustified" elements in the codebook are actually FORCED by the constraint map — every tested alternative is killed by a specific universal constraint. This is analogous to frozen variables in CSP theory (Zdeborová & Krzakala 2007), though the R3 audit enumerates a finite set of designer-chosen alternatives rather than proving all alternatives fail. **Round C (the decisive test) was never completed** — blocked by codebook ban. The methodology (enumerate alternatives, kill with constraints) is substrate-agnostic; the specific results are codebook-specific. **LSH frozen element enumeration is queued as the next R3 audit target.**

   **Encoding determines performance (I1), not R3.** The encoding pipeline (avgpool, centering, normalization, action mapping) accounts for 300x difference in step count across games (LS20: 26K, FT09: 82, VC33: 283 — same substrate). This is an I1 issue (representation discovery), not R3 (operational self-modification). R3_AUDIT encoding compilation: 3I/2M/1 narrow U — the encoding is mostly R3-compliant. The 300x impact is about WHAT to process (I1), not HOW to process (R3). Anti-inflation rule 7 still applies: audit the full system (substrate + encoding) for frozen frame accounting (R4).

   **No Phase 2 substrate has been formally R3-audited** — expected given kill velocity (SelfRef, Tape, Expr, Temporal, GRN all killed in <5 experiments). LSH has been active since Step 453 (~180 experiments) with frozen elements (random hyperplanes, k, centering, argmin) not yet enumerated. Recode's self-refinement (Step 542) is the closest to R3 compliance. Op-code definitions (Steps 581d, 582: PENALTY=100, death/surprise semantics) are frozen value judgments — R4 says they must be adaptive or irreducible. Hart debate (2026-03-21): computational parameters (R4) vs value parameters (R1) are different axes.

   **The R3 barrier precisely located (Steps 577b-577d):** Evolutionary R3 (577b): 0/1000 random programs find the pipeline. Buffer navigation (577d): 0 L1s across ALL buffers — pixel statistics navigate to MORE cells but WRONG cells. R3 requires reward-correlated features, not visually salient features. Without a feedback loop linking strategy to outcomes, no pixel statistic discovers the exit.

   **Eigenform series (Steps 620-629, 10 experiments):** Self-observation via graph statistics — substrate reads its own edge count distribution, computes percentile thresholds, assigns AVOID/PREFER/NEUTRAL op codes. Mechanism works mechanically (AVOID grows 0→8%), but is INERT for L1 (argmin-solvable, no room for improvement). Self-observation self-terminates at L1 (Step 621: M→2000). L2=0/5 (Step 629: reflective only, can't reach unknown territory). P3 7-53x SLOWER (Step 625: AVOID contaminates known paths). **Eigenform points INWARD (avoid known graph) not OUTWARD (toward unknown states).** L2 requires predicting unseen states — a world model. The graph stores the past; L2 requires the future. This is the same wall as the exploration series (477-482) from a different angle: introspection ≠ foresight.

   **R4 testable formulation:** R4 (frozen frame must be minimal) is tested by the R3 audit methodology — enumerate every frozen element, classify as M (modified), I (irreducible), or U (unjustified). R4 passes when U=0. The R3 audit IS the R4 benchmark. No separate R4 test needed.

2. **R1-compliant classification:** No substrate classifies without external labels. LSH k=16 achieves 36.2% with self-labels (Step 573, 4x above codebook's 9.8%), but still far below supervised. Classification under R1 remains unsolved.

3. **Purposeful exploration (I6, I9):** All navigation is stochastic coverage with argmin bias. 6 targeted strategies (Steps 477-482) all worse than argmin — failures were mechanism-specific (noise, locality traps, noisy TV), not categorical. **Argmin modification closed (U28, Steps 581d + 630-639, 12 experiments, LSH only):** No auxiliary signal at any density significantly improves argmin navigation (581d debunked at n=20, Step 584: p=0.63). Dense signals (>~60% firing rate) actively damage navigation by overriding argmin decisions (633: 66% → 3/5 seeds 10-18x slower; 635: 94-98% → 3/5 seeds 5-20x slower). Sparse/moderate signals (<~55%) are inert. L2 requires a fundamentally different mechanism, not argmin + auxiliary signal. Attractor hypothesis (2026-03-21, downgraded by Hart) partially correlated: death penalty is both sparse AND attractor-removing, delta is both dense AND attractor-creating — but correlation doesn't establish subsumption. Both lack positive confirmation at scale.

4. **Population-level R3:** GRN substrate (Step 607) killed — encodings not diverse enough. Price equation framework predicts success requires behaviorally diverse encoding candidates, not just parametric variation.

5. **LS20 L4+:** Blocked at L3 transition (puq_wall_set bootstrap, Step 611-612). Fix structurally correct but needs longer run to verify.

### External Audit Status (2026-03-18, 13 findings)

Integrated: Finding 4 (R3 I-classification — anti-inflation rule 8, forced ≠ unjustified), Finding 5 (constraint map induction — reclassified 2026-03-18), Finding 7 (adversary process — Hart debate formalized), Finding 12 (research-algorithm isomorphism — PAPER.md Section 4.4).

Partially addressed: Finding 1 (R1/R5 clarification — R1 clarification fixed via Hart debate 2026-03-21, but labeled vs unlabeled comparison NOT added, both numbers NOT reported in CONSTRAINTS.md).

Outstanding: Finding 2 (no CL head-to-head comparison — HIGH), Finding 3 (random walk propagation — codebook-specific, LSH argmin stays effective), Finding 6 (R2-performance tradeoff — not documented), Finding 8 (ANIMA connection — not explored), Finding 9 (survival benchmark — partially tested via chain Steps 543-577), Finding 11 (paper framing — strategic). Finding 10 (ghost projects) and Finding 13 (FOLD_EVOLUTION.md) are LOW severity.

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
