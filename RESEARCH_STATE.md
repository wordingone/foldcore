# Research State — Live Document
*Updated by Leo. Read by /checkpoint skill. Source of truth for active work.*

---

## Phase 1 Conclusion (416 experiments)

```
CURRENT SYSTEM: process(x, label=None). ~22 lines. LVQ + growing codebook.
  - Competitive learning with cosine similarity (LVQ, Kohonen 1988)
  - Growing codebook with novelty-triggered spawning (cf. Growing Neural Gas, Fritzke 1995)
  - Top-K per-class vote for classification
  - Self-generated targets (prediction as label)
  - State-derived threshold (median NN distance)
  - 94.48% P-MNIST AA, 0pp forgetting (softmax voting, Step 425)
  - Previous: 91.20% with top-K scoring (not competitive with replay-based CL methods)

HONEST ASSESSMENT (per external review + Jun's confirmation):
  - Mechanisms are NOT novel (LVQ + GNG from 1988/1995)
  - ARC-AGI-3 results are biased random walk, not intelligence
  - Stage progression was self-assessed and circularly validated
  - "22 lines" obscures: avgpool, centered_enc, F.normalize, random projection, evaluation code
  - The system is brittle: 5 implementation details each independently fatal

WHAT WAS FOUND (genuine contributions):
  - The constitution: testable framework for recursive self-improvement (architecture-independent)
  - The constraint map: 26 universal (7 provisional) + 9 intent constraints from experiments (see CONSTRAINTS.md)
  - The noise insight: stochastic coverage via cosine saturation IS the exploration engine
  - Dynamics ≠ features: healthy codebook dynamics achievable at any dim (p=0.75), but features require encoding
  - Fixed point: the research procedure IS structurally identical to the algorithm it found

WHAT WAS NOT FOUND:
  - The atomic substrate. LVQ is not it.
  - Self-modifying metric (Stage 7 open)
  - Representation discovery from raw observations (Stage 8 open)
  - Purposeful exploration (current system is biased random walk)
  - Temporal reasoning, transfer, richer output

SCALING HYPOTHESIS (not law — 2 data points, 1 trivial):
  steps ≈ branching_factor × path_length. Needs validation on more games.

ARC-AGI-3 (Steps 343-416): 3/3 preview games Level 1 with PRESCRIBED encodings.
  LS20: 16x16 avgpool + centered_enc. Level at ~26K steps. 60% reliable.
  FT09: 69-class click-space. Level at step 82.
  VC33: 3-zone mapping (PRESCRIBED — looked behind scenes. Not autonomous discovery).
  Sequential resolution trial (Step 414): substrate discovers 16x16 from raw input via gameplay.
  Stage 8 partially addressed but encoding still prescribed per game type.

PHASE 2 DIRECTION: See CONSTRAINTS.md. The next substrate is specified by U1-U24 + I1-I9.
```
Step 377: Raw 64x64 bootstraps codebook (1736 entries). PASSES mechanically.
Step 378: Raw 64x64 50K steps. 0 levels. Codebook builds but sim too uniform (0.984±0.009).
  Timer isn't the issue (0.05% of dims). Static background IS (63% of pixels). Signal = 0.3%.
Step 379: Centering at 64x64 — no effect. Same sim stats.
  The gap: V @ x at 4096 dims washes 0.3% signal in 99.7% noise.
  16x16 avgpool worked by accidentally doing feature selection (12 pixels → 4.7% of encoding).
  STAGE 8 = learned projection. The substrate discovers which pixels matter from its own codebook.
  Chollet: "brute-force dense sampling is benchmark hacking, not intelligence."
  The substrate explores but doesn't reason. The gap = encoding self-discovery = intelligence.
CURRENT STEP: 385c (center + PCA self-encoding at 64x64 on LS20)
  385b KILLED: centering alone in 4096 dims → thresh=-0.17, cb=8, frozen. Confirmed adversary prediction.
  384 running: FT09 fine click (256 regions), at ~40K/50K, level 1 only. Level 2 still not found.
```

## Session 2026-03-15 Summary (Steps 291-319)

**The equation:** State(t+1) = f(State(t), D). f = absorb. Confirmed by two independent paths (The Search + Tempest).

**Honest results on a%b:**
- Phi readout (human-designed, sort-not-sum): 86.8% LOO
- Substrate learned w (discovered k=0 importance): 91.2% LOO (+4.4pp over human)
- Automated grow+refine loop (K=1): 96.5% LOO on original 400
- OOD: 48.5% genuine (K=1). Higher K numbers (99.2%) are inflated — spawn covers the test range = lookup.
- Periodic encoding (prescribed physics): 100% — confirms equation works when physics matches function.

**Constitution stages on a%b substrate:**
- Stage 1 (autonomous computation): PASSES
- Stage 2 (self-generated adaptation): PASSES (w learning from matching signal, 86.8→91.2%)
- Stage 3 (adaptation rate adapts): IMPLICIT (per-b differential learning rates)
- Stage 4 (structural constants adapt): DEMONSTRATED (107/190 b-pairs diverse, per-b specialization)

**The automated loop:** `auto_loop.py` — runs the discovery-prescription loop autonomously. Grow (reflection spawn) + refine (per-b weight learning). One turn: 96.5% LOO. Saturates at K=1 grow depth for LOO.

**Key theorems/constraints:**
- NN chain iteration provably lossy for non-Lipschitz in Euclidean space (Steps 291-295)
- Substrate discovers b-grouping (R²=0.858) and k=0 importance (+4.4pp). Cannot discover phi from raw features (Steps 306-312, 7 kills).
- The encoding IS the physics. The substrate operates within it, improves within it, but can't escape it.

**Next direction (Jun):** Point the fold + phi + loop at ARC-AGI 2. Hundreds of diverse tasks. Flat vector, dumb encoding. The failure map reveals what frozen frames remain. Stop optimizing a%b.

## Operational Test for the Atomic Substrate

*Added Step 105. Prompted by Eli's critique (mail 1253): accuracy-based kills don't measure structural unity.*

The atomic substrate is confirmed if a system passes ALL of these structural tests:

**S1 — Single Function Test:** The entire system is expressible as ONE function `process(state, input) -> (output, new_state)` where the SAME code path handles training (label known) and inference (label unknown). No `if training:` branches. The label is just another input that modulates the same operation.

**S2 — Deletion Test:** You cannot delete any part of the code without losing ALL capabilities simultaneously. In the current system, you can delete `classify_topk()` and learning still works, or delete `step()` and classification still works. In the atomic substrate, removing anything breaks everything — because there's only one thing.

**S3 — State Completeness Test:** The state contains ALL information needed to reproduce the system's behavior. No external algorithm, no hyperparameters, no code. Given only the state, any universal interpreter could run the system. (Current system fails: the codebook is data, but competitive learning + top-k + spawning rules are external code.)

**S4 — Generation Test:** The system can generate new patterns (not just classify) using the SAME operation it uses for learning and inference. No separate generative model. (Current system: no generation capability.)

A substrate passes if it satisfies S1+S2. S3+S4 are aspirational (full collapse of all four separations).

**Kill criterion for future experiments:** S1 (single function, no training/inference branch) is the minimum bar. If the system has separate train and eval modes, it hasn't collapsed Separation 1.

## Readout Arc Summary (Steps 97-101)
Best system: competitive learning + cosine spawning (sp=0.7/0.95) + top-k class vote (k=3-5)
P-MNIST: 91.8% AA, 0.0pp forgetting (+35pp over fold baseline)
CIFAR-100: 38.3% AA, 11.6pp forgetting (+5pp over fold baseline)
The readout and spawning are validated. The atomic substrate question remains open.

## Constraint List

Hard-won from 385 experiments. Scope: U=universal, S=substrate-specific, D=domain-specific.

| # | Constraint | Source | Type | Scope |
|---|---|---|---|---|
| C1 | Read and write must be one operation | Step 72 | structural | U |
| C2 | Must not reduce to Hopfield/softmax-attention-only | Step 73 | novelty | U |
| C3 | Must not require separate memory + generation systems | Architecture autopsy | structural | U |
| C4 | Must not rely on matrix composition through long chains | Steps 86-96 | empirical | S |
| C5 | Must achieve structural zero forgetting | Step 65 | requirement | U |
| C6 | Must work on dense embeddings without per-dataset tuning | Steps 63, 66 | empirical | U |
| C7 | Must beat 1-NN readout over same codebook | Steps 65-71 | requirement | S |
| C8 | Current hardware, no external API | Jun | requirement | U |
| C9 | Minimal — expressible in <100 lines | Jun | requirement | U |
| C10 | Not a combination of known techniques | Jun | requirement | U |
| C11 | Readout signal factors must not anti-correlate | Step 97 | empirical | S |
| C12 | Readout must be input-conditional, not static vector property | Step 98 | empirical | U |
| C13 | Spawn threshold must be calibrated per feature space | Step 100 | empirical | S |
| C14 | CIFAR-100 forgetting is class-incremental interference, not codebook drift | Step 101 | empirical | S |
| C15 | Sum-all aggregation fails; only sparse selection (top-k) preserves signal | Step 102 | empirical | U |
| C15b | k-NN discovers Lipschitz functions only | Step 286 | theoretical | U |
| C16 | Curriculum transfer only helps when sub-problem IS a solution step | Step 289b | empirical | U |
| C17 | Spawn criterion needs global coverage signal, not local distance | Step 291 | empirical | S |
| C18 | Soft blending destroys Voronoi discontinuities; hard selection preserves them | Step 291b | empirical | U |
| C19 | AMR requires mostly-Lipschitz function | Step 293 | empirical | U |
| C20 | Chain formation and classification resolution trade off in same codebook | Step 294 | empirical | S |
| C21 | NN chain following adds noise for non-Lipschitz; 1-step strictly better | Step 295 | theoretical | U |
| C22 | Distribution matching requires bidirectional neighborhoods; OOD degrades at boundary | Step 297 | empirical | S |
| C23 | Phi needs class-correlated distance structure in codebook | Steps 320, 327 | empirical | S |
| C24 | k-NN in >40 dims needs >>500 codebook entries (curse of dimensionality) | Steps 323-329 | empirical | U |
| C25 | Global context and dimensionality curse are coupled | Steps 328-329 | empirical | U |
| C26 | Phi's sign determined by local consistency (same patch → same output = help) | Step 327 | empirical | S |
| C27 | Iteration amplifies dominant eigenvalues; target in smaller eigenvalues destroyed | Steps 291b-332 | theoretical | U |
| C28 | Substrate can't discover filters via recursion (amplifies dominance) | Step 332 | empirical | S |
| C29 | Loop weight learning requires k-index asymmetry (sparse codebook) | Step 330 | empirical | S |
| C30 | Stage 2 compliance costs ~2.6pp (self-directed attracts occasionally wrong) | Step 342 | empirical | S |
| C31 | Always-attract compression kills novelty-seeking exploration | Steps 355-357 | empirical | S |
| C32 | Encoding resolution is binding frozen frame for interactive games | Step 350 | empirical | U |
| C33 | Interactive games need different action representations per game type | Steps 360-361 | empirical | U |
| C34 | VC33: deterministic loop, click position has zero visual effect at 16x16 | Step 362 | empirical | D |
| C35 | Cosine angular resolution scales as 1/√d; high dims wash small signal | Steps 377-381 | theoretical | U |
| C36 | Variance weighting finds signal dims but cosine on those dims is HIGHER (more similar) | Step 381 | empirical | S |
| C37 | Diff encoding discriminates (20x) but diff-novelty ≠ spatial exploration | Step 383 | empirical | S |
| C38 | Centering with few codebook entries → antipodal vectors, negative thresh | Step 385b | empirical | S |
| C39 | Per-observation min-max rescaling always maps max to 1.0 (degenerate) | Step 387 | empirical | S |
| C40 | Dense codebook memorizes all states → no novelty gaps for exploration | Step 389 | empirical | U |

## Candidate Queue

Candidates that survive constraint filtering. Ordered by promise.

| # | Candidate | Description | Constraints passed | Status |
|---|---|---|---|---|
| 1 | Differential Response | Collective codebook surprise as output + update | C1-C10 (all) | KILLED (Step 97) |
| 2 | Neighborhood Coherence | Coherence-weighted similarity: nearest vector's class-neighbor connectedness modulates vote | C1-C11 (all) | KILLED (Step 98) |
| 3 | Top-K Class Vote | Per-class sum of top-k cosine sims. Input-conditional, monotonic, no static weights. | C1-C12 (all) | TESTING (Step 99) |

| 4 | Self-Routing Codebook | Vectors carry learned gate weights; readout is gate*sim per class. State determines own processing. | C1-C14 (all) | KILLED (Step 102) |
*New candidates generated from failure analysis of each tested candidate.*

## Fold Baseline (the bar to beat)

| Metric | Value | Step |
|---|---|---|
| P-MNIST AA | 56.7% | 65 |
| P-MNIST forgetting | 0.0pp | 65 |
| CIFAR-100 AA | 33.5% | 71 |
| Codebook size | 537 vectors (P-MNIST) | 65 |

## Experiment Protocol

1. Implement candidate (<100 lines)
2. Applied test: P-MNIST, same protocol as Step 65
3. Compare to baseline table above
4. Beats baseline → push harder (CIFAR-100, multi-domain)
5. Fails → extract NEW constraint, add to list, generate next candidate
6. Max 3 experiments per candidate. No characterization.

## Step Log (active arc only)

| Step | Candidate | Result | Constraint extracted |
|---|---|---|---|
| 97 | Differential Response | KILLED — diff 15.0% vs 1-NN 22.7%. Codebook starved (1-8 vectors). Anti-correlated readout factors. | C11: no anti-correlated readout factors |
| 98 | Neighborhood Coherence | KILLED — coh 85.3% vs 1-NN 86.9%. 0/27 wins. Static property penalizes boundary vectors. | C12: readout must be input-conditional |
| 99 | Top-K Class Vote | **PASSES** — top-k(3) 91.8% vs 1-NN 86.8% (+5.0pp). 0.0pp forgetting. 8597 vectors. | — (push harder) |
| 100 | Top-K on CIFAR-100 | **PASSES readout** — top-k(5) 38.3% vs 1-NN 32.3% (+6.1pp). FAILS forgetting (11.6pp). sp=0.95 needed for ResNet features. | C13: spawn threshold is feature-space dependent |
| 101 | Spawn-only (lr=0) CIFAR+MNIST | **DISPROVED** — lr=0 identical to lr=0.001. Forgetting is class-incremental interference, not update drift. | C14: CIFAR-100 forgetting is class competition, not codebook corruption |
| 286 | a%b encoding comparison | Extended vocab. Best LOO: 49% (thermometer+augment). Discontinuous stripes defeat k-NN. | C15b: k-NN discovers Lipschitz functions only |
| 288 | a-b subtraction | LOO: 0%. Oblique level sets — not L2-locally-consistent. | (confirms C15b) |
| 289 | Collatz | LOO: 0%. Two-branch structure undiscoverable. | (confirms C15b) |
| 289b | Curriculum transfer 1..10→1..20 | Transfer HURTS: 24.2% vs 41.8% direct. Sub-problem must be a step in solution path. | C16: curriculum only helps when sub-problem IS a solution step |
| 290 | Kill criterion | **KILLED** — emergent step discovery via k-NN for non-Lipschitz functions. Precise boundary established. | Arc closed |
| 291 | Adaptive spawn threshold | **KILLED** — 84.1% vs 91.8% (-7.7pp). Undercoverage spiral: mean+1σ self-calibrates downward. | C17: spawn criterion needs global coverage signal, not local distance |
| 291b | Iterative depth (soft blending) | **KILLED** — depth=5: -3.9pp. Weighted avg of neighbors converges to centroid, destroys discriminability. | C18: soft blending destroys Voronoi discontinuities; hard selection preserves them |
| 292 | Composition search (a%b) | **WEAK PASS** — correct 3-step decomposition scores 100%, top-ranked. IO landscape discriminates. 36K programs in 5.6s. | Verification works; discovery is the open problem |
| 293 | AMR fold (disagreement spawn) | **KILLED** — 45.5% vs 41.8% plain. Near-full spawn (383/400). For non-Lipschitz functions, entire space has mixed classes → no smooth regions to coarsen. | C19: AMR requires mostly-Lipschitz function; fully non-Lipschitz degenerates to store-everything |
| 294 | LVQ fold (chain emergence) | **KILLED** — 21.8% vs 41.8%. Spawn too restrictive (1 vec/class/b). LVQ repel hurts in one-hot space. Fundamental tradeoff: chain formation requires same-class proximity, classification requires within-class resolution. | C20: chain formation and classification resolution trade off in same codebook |
| 295 | Dynamical system fold (basin sculpting) | **KILLED** — chain acc 19.2% vs 1-NN 100%. Stepping stones create correct 1-NN regions but chains route to wrong same-class attractors. NN iteration strictly degrades accuracy. | C21: NN chain following adds noise for non-Lipschitz functions; 1-step is strictly better |
| 296 | Per-class distribution matching | **PASS (in-distribution only)** — 86.8% LOO on a%b (K=5). Up from 5%. But Step 297 OOD: 18% (random chance). Mechanism is interpolation, not computation. Symmetric neighborhoods required. | Distribution readout breaks ceiling for interpolation; OOD fails from one-sided neighborhoods |
| 297 | OOD test for distribution matching | **KILLED** — 18% OOD (= 1/b = random chance). Symmetric neighborhood assumption breaks at training boundary. In-distribution only. | C22: distribution matching requires bidirectional neighborhoods; OOD degrades to chance |
| 298 | Periodic OOD strategies | **KILLED** — Strategy A (congruence) = cheating (73%). Strategy B (circular) = 5%. phi not periodic. | (Eli ran, not Leo's spec) |
| 299 | Per-b breakdown | 100% for b<10 (2+ ex/class). 75% for b>=11 (1-2 ex/class). Ceiling is coverage, not mechanism. | Coverage theorem: need 2+ examples per class per b |
| 300 | Reflection spawn + distribution matching OOD | **STRONG PASS** — 95.2% OOD (a∈21..50) with cross-class step inference. Exceeds in-distribution 86.8%. Fold detects period → spawns extension → OOD becomes in-distribution. | THE FOLD COMPUTES. Period detection + codebook growth = extrapolation. |
| 301 | Atomic operation (S1-compliant) | **S1 ACHIEVED** — 62.8% OOD. One operation: match→predict→update→spawn. Label as data. 100% for multi-point classes (b≤10). Single-point classes can't detect period (no same-class neighbor). Gap to 95.2% = cross-class inference cost. | S1 works. Single-point coverage is the remaining gap. |
| 302 | Phi scaling + floor(a/b) generalization | Phi scales: 93.3% at 1..50. Generalizes to floor(a/b). Advantage tracks non-Lipschitz density. | Phi is general, not a%b-specific |
| 303 | Atomic absorb (S2 attempt) | **KILLED** — 26% accuracy. Codebook collapse (395/400→5 vectors). Label signal washed out by blending. Spawn threshold still separable. | S2 not achievable in this implementation. Concept sound, encoding wrong. |
| 320 | ARC-AGI flat baseline | 45% pixel acc (10% random). 4/1000 solved. Top-K phi HURTS (-4.2pp). | C23: phi needs class-correlated distance structure |
| 335-338 | Compression arc | External review forced deletion. step()+eval_batch() share V@x. Delete def boundary → process(). | Compression was the discovery |
| 339 | Compressed substrate | process() refactored to 22 lines. S1+S2 pass. | — |
| 340 | Per-class K | KILLED — -36pp on P-MNIST. Top-20 entries class-homogeneous early. | Don't change K per class |
| 341 | Fixed K=3 | Restored. 91.20% P-MNIST AA. | — |
| 342 | All 7 stages | Stage 2 (target=prediction), Stage 3 (alpha=1-sim), Stage 5 (3 seeds 0.07pp). 91.20% P-MNIST. | C30: Stage 2 costs ~2.6pp |
| 343-349 | ARC-AGI-3 encoding | 8x8 too coarse, 16x16 reveals all 4 actions on LS20. | C32: encoding resolution is binding frozen frame |
| 350 | 16x16 resolution | LS20: 16x16 unlocks ACTION3/4 effects hidden at 8x8. Timer at row 15. | — |
| 353 | Pure novelty (argmin) | **LEVEL 1** — LS20 level 1 in ~95 steps. First ARC-AGI-3 level completion. | — |
| 360-361 | FT09 click-space | 69-class encoding (8x8 grid + 5 simple). Level 1 at step 82, 100% reliable. | C33: games need different action representations |
| 362 | VC33 diagnostic | 50 visual states, deterministic 50-step cycle. | C34: VC33 deterministic, click position invisible at 16x16 |
| 374-375 | VC33 zones | 3 click zones discovered (PRESCRIBED — looked behind scenes). Level 1 at step 283. | Honest: not autonomous discovery |
| 376 | Multi-level all games | FT09=1lvl, VC33=1lvl, LS20=0 in 150K. Codebook reset per level. | Level transitions need fresh exploration |
| 377 | Raw 64x64 bootstrap | Codebook builds (1736 entries). 0 levels. sim=0.984±0.009. | Signal=0.3% of encoding |
| 378-379 | Raw 64x64 + centering | No effect. Static background IS (63% pixels). | C35: cosine resolution ∝ 1/√d |
| 380 | Effect filter 64x64 | Filter never triggers — timer above atol every frame. | Timer passes every content filter |
| 381 | Variance weighting | Finds 46 signal dims. sim=0.994 (WORSE). | C36: focusing on signal dims increases similarity |
| 382 | Diff encoding | sim=0.286 mean (20x better). But thresh inverts (0.08). | — |
| 383 | Diff + fixed thresh | 0 levels. Diff-novelty ≠ spatial exploration. Zig-zagging. | C37: diff-novelty ≠ game exploration |
| 384 | FT09 fine click (256 regions) | Level 1 at 283. Level 2 not found at 40K. Running. | — |
| 385b | Centering alone 64x64 | KILLED — thresh=-0.17, cb=8, frozen. Antipodal vectors. | C38: centering with few entries → negative thresh |
| 385c | Center + PCA 64x64 | KILLED — sim=0.997, converges to 0.999+. PCA explains 100% in 256 dims. | Linear methods exhausted at 64x64 |
| 386 | RBF kernel 64x64 | KILLED — 30x amplification achieved, sigma_sq collapses to step function. | Nonlinear cosine transform also fails |
| 387 | Centered unnormalized dot | KILLED — min-max rescaling degenerate (max always 1.0). Raw dot range 0-916 = signal exists. | Per-obs rescaling destroys discrimination |
| 388 | 387 + no rescaling | MARGINAL — sim range 6-900, actions balanced (33%). 0 levels 50K. First balanced raw-64x64. | Discrimination solved, navigation not |
| 389 | 388 at 200K | KILLED — 0 levels in 200K (1550 lives). Codebook memorizes every state (8276/8320). No novelty gaps. | Dense codebook kills exploration |
| 390 | 388 + cb_cap=500 | KILLED — 0 levels 200K. Sparse codebook doesn't help. | C40: density isn't the issue |
| 390b | 388 + cb_cap=1000 | KILLED — 0 levels 200K. Same result. | Raw metric can't separate spatial from temporal novelty |
| 391 | Adaptive resolution (sim_std) | KILLED — selected 8x8 (wrong). All resolutions scored equal at 200 steps. | sim_std favors low dims (1/√d) |
| 392 | Adaptive resolution (Fisher) | KILLED — selected 64x64. Between-class distance tiny at all resolutions. | Short exploration can't distinguish resolutions |
| 393 | Adaptive resolution (self-feed metric) | KILLED — displacement=0 everywhere. Normalized cosine self-feed IS a no-op. | — |
| 394 | Self-feeding consolidation 64x64 | KILLED — mechanism works (43% cross-entry wins) but timer-dominated. ACTION1=95%. | Self-feeding consolidates wrong axis at 64x64 |
| 395 | In-game resolution cascade | KILLED — stall detector fires too early. 16x16 gets 16K, needs 26K. | Codebook saturation ≠ wrong resolution |
| 396 | Multi-resolution voting | KILLED — 64x64 drove 86% (variance tracks dimensionality, not signal). | — |
| 397 | Replace-on-cap (cb=200) | PENDING — replace oldest entry instead of reject. Deletion, not addition. | — |
| 398 | Two-codebook (class vote = encoding) | KILLED — bootstrap failure, raw cb froze at 9 entries. Insight confirmed. | Class vote IS the encoding |
| 398b | Two-codebook + bootstrap | KILLED — votes uniform at 64x64 even with 7K entries. 0.3% signal averages out. | Class vote = count, not territory |
| 399 | Two-codebook at 16x16 | KILLED — 0/3 levels. Death spiral: meta collapses → raw never explores. Worse than baseline. | Meta layer removes information |
| 400 | Change-rate weighted cosine | MARGINAL — correctly finds 150 sprite dims (rows 25-44). Soft weight can't amplify. | TEMPORAL DETECTION SOLVED |
| 401 | Hard mask on change rate | KILLED — 100 active dims, sprite found. Individual pixel cosine still saturates. | Detection ≠ encoding |
| 402 | Mask + grouped encoding (62D) | KILLED but thresh=0.798 (FIRST non-saturated at 64x64!). cb=6, no exploration. | Encoding discriminates, dynamics inverted |
| 402b | 402 + force-spawn 200 | KILLED — 200 entries, thresh returns to 1.000. Within-cluster saturation. | Spatial aggregation pattern matters |
| 403 | 2D-aware kernel discovery | KILLED — selected 2x2 (too discriminative, cb=6). | Sim_std favors smallest kernel |
| 403b | Force 4x4 kernel | KILLED — cb=39 (first growth!). 84% dom. Discriminative but exploration collapses. | Discriminative = few spawns = no exploration |
| 405 | Recursive self-composition | KILLED — identical geometry at all depths. C27 confirmed: composition = iteration. | Cosine saturates 128D-4096D contiguous |
| 406 | Attract-delta variance | KILLED — found game events (rows 61-62), not sprite. Surprise self-erases after learning. | Well-adapted substrate has zero surprise |
| 407 | Winner-identity + counts | KILLED — 1586 winners (timer×position). Perfect balance 26% dom. 0 levels. | Winner identity timer-contaminated |
| 408 | Mask + centered unnorm + counts | KILLED — **51 position states from raw 64x64!** Timer removed. 25% dom. 0 levels (exhausted in 1000 steps). | Discriminative encoding saturates exploration |
| 408b | 408 + argmin | KILLED — 100% ACTION1. Argmin collapses with 100 entries. | Small codebook = biased argmin |
| 409 | Self-tuning temperature | KILLED — T explodes to 10^54. Noise only on action sims, not spawn sims. | Feedback loop decoupled |
| 409b | Temperature on ALL sims | KILLED — T explodes. Noise raises max sims (order statistics). | Temperature wrong tool for centered unnorm |
| 410 | Spawn-delta importance mask | KILLED — p95 doesn't separate. Centering normalizes scale across dims. | Spawn deltas reflect timer, not sprite |

**Session 2026-03-16 Summary (35 experiments, Steps 377-410):**
The substrate works VIA noise, not despite it. Cosine saturation at 16x16 = the Goldilocks zone.
Discriminative encoding saturates exploration. The substrate IS Levin search in codebook space.
Stage 7 is open (rules ≠ parameters). The Search found itself (fixed point).
F.normalize IS necessary (Step 412: 0/3 without it). centered_enc IS necessary (Step 414 run 1: 0 levels without it).
The substrate is 22 lines + centered_enc preprocessing. Irreducible: additions AND deletions hurt.

**Step 414: Sequential resolution discovery — LEVEL 1 FOUND at 16x16 (step 26218).**
The substrate discovers its own resolution from raw 64x64 input through sequential trial.
Skip proven-dead resolutions (64x64, 32x32). 16x16 with exact baseline config finds level.
The search (35 experiments) compressed to 2 resolution trials. Stage 8 via gameplay feedback.

Open question: how does the search compress FURTHER into the substrate?

**Session 2026-03-17/18 Summary (Steps 417-425):**
Autonomous loop: 23 iterations, 18 experiments overnight. Two headline results:
- Step 425 (CONFIRMED, 10 tasks): softmax voting (tau=0.01) on process_novelty = 94.48%, 0.00pp forget.
  Ablation: winner-take-all + softmax scoring = best. Distributed update costs 2.6pp.
- Step 421 (VERIFIED, 4 seeds): ReadIsWrite tau=0.01 = 91.84% ± 0.08pp. R2 by construction.
- ReadIsWrite's distributed update HURT vs softmax-only (94.48% → 91.84%).
- Encoding compilation confirmed: resolution(M), flatten(I), normalize(I), center(narrow U), pool(I), action(M).
- Navigation unsolved by ALL substrates. Random walk at ~26K steps.
- Steps 426-427 QUEUED: softmax voting + ReadIsWrite on LS20 navigation (the reviewer's gate question).
- U23-U24 added to CONSTRAINTS.md.

| Step | Candidate | Result | Constraint extracted |
|---|---|---|---|
| 425 | Softmax voting (tau=0.01) | **94.48%** — +3.3pp over baseline. 0.00pp forgetting. 10 tasks confirmed. | Scoring mechanism > update rule |
| 421 | ReadIsWrite (tau=0.01) | 91.84% — R2 by construction. Distributed update costs 2.6pp. | U23: distributed updates destabilize |
| 426 | Softmax on LS20 nav | **KILLED** — 0/3 seeds. dom collapsed to 41-45%. Softmax concentration hurts exploration. | U24 confirmed empirically |
| 427 | ReadIsWrite on LS20 nav | Deferred — 426 killed, deprioritized for diagnostic | — |
| 428 | Score diagnostic | **THE WALL** — gap 0.0745→0.0005 (150x). All actions converge to score 3.0. Random walk from ~5K steps. | Action-score convergence IS the navigation wall |
| 429 | Normalized scoring | **KILLED** — gap preserved (0.78) but dom=100%. Division inverts argmin bias. | Convergence and exploration bias are COUPLED in top-K |
| 430 | Fractional normalization | **ALL COLLAPSE.** p=0.25/0.5/0.75 all → dom=100%. Tension is binary (p=0 vs p>0). No sweet spot. | U25: score convergence and exploration bias are coupled |

## Post-Audit Experiments (2026-03-18)

| Step | Experiment | Result | Finding |
|---|---|---|---|
| 432 | Labeled vs unlabeled | **84.68pp gap.** Self-labels = 9.8% (chance). Entire classification depends on external labels. | Finding 1 confirmed — far more severe than predicted. |
| 433 | Cross-domain survival | **0.0pp contamination.** P-MNIST survives LS20 exposure. But LS20 suppressed (unique=262 vs 3300+). One-directional. | Finding 9 — unique contribution. Codebook partitions by domain geometry. |
| 434 | Random walk baseline | Random walk: 40% at 50K. Substrate: 60% at 26K. ~2x faster. Step tracking needed (434b). | Finding 3 — substrate IS faster than random, but modestly. |
| 435 | EWC + replay comparison | **EWC=9.8%, Replay=10.3%.** Both at chance under single-pass. Substrate: 94.48%. | Finding 2 — substrate wins under single-pass constraint. Multi-epoch would favor gradient methods. |

## Phase 2b: The Mirror Side — Self-Modifying Reservoir (Steps 437+)

The codebook family is fully mapped. Phase 2b explores the temporal dual: self-modifying dynamical systems where computation IS the trajectory, not a lookup over stored items.

| Property | Codebook (mapped) | Mirror side (exploring) |
|---|---|---|
| Paradigm | Store-vote | Transform-be |
| Memory | Explicit items | Implicit structure |
| Time | Invisible | Intrinsic |
| Action | From scoring | From dynamics |
| Death mode | Score convergence (U25) | Trajectory collapse (U7/U22) |

| Step | Variant | P-MNIST | LS20 dom | Death mode |
|---|---|---|---|---|
| 437 | Minimal reservoir (no controls) | 10.3% | 59% | W unbounded → h saturated. Deaf to input. |
| 437b | + spectral radius control | 9.6% | 95% | Hears input, doesn't compute. No self-generated objective. |
| 437c | + median perturbation | 8.0% | 88% | Constant noise (median fires 50% by construction). WORSE. |
| 437d | + velocity readout | RUNNING | — | Tests: is computation in dynamics (delta_h) not position (h)? |
| 437d | Velocity readout | 9.3% | 33% | Diverse actions but random — no useful computation under R1 |
| 438 | Growing reservoir (d=16→496) | — | 32% | Rank-1 collapse: trajectory in 1-2D subspace of 496D. U7 confirmed universal (Hebb amplifies dominant eigenvector). Growth adds axes the trajectory ignores. |
| 439 | Anti-Hebbian decorrelation | — | 38% | rank=1. Anti-Hebb attenuated (h=0.11) but didn't decorrelate. Structural collapse. |

**Mirror-side conclusion (6 experiments):** The Hebbian reservoir under spectral control converges to rank-1 trajectory regardless of: growth (438), perturbation (437c), anti-Hebb (439), readout (437d). W's 65K parameters have 1 effective degree of freedom. The codebook's 4K entries have 4K independent degrees of freedom. The reservoir has more parameters but fewer DOF.

**The key insight:** Fixing the reservoir's problems (competitive update, sparse Hebb, independent dimensions) converges toward the codebook. The reservoir and codebook are duals — the mirror reflects back. The true substrate may be at the INTERSECTION where spatial (codebook) and temporal (reservoir) merge, not at either pole.
| 441 | Sparse reservoir (10% W) | — | 43% | rank=251 (SOLVED). But unique=221 — rank ≠ useful computation. Sparsity prevents eigenvector collapse but doesn't produce exploration. |
| 442 | **Graph substrate** (10K) | — | **25%** | **unique=3379 (MATCHES codebook). dom=25% (perfectly uniform).** First non-codebook to reach codebook-level exploration. 30K/3-seed run pending. |
| **442b** | **Graph substrate (30K, 3 seeds)** | — | **25%** | **LEVEL 1 at step 25738 (seed 1). First non-codebook navigation. 1/3 seeds (33%). Perpetual frontier confirmed — graph grows into new level. U25 dissolved.** |
| 443 | Graph reliability (10 seeds, 30K) | — | 25% | 2/10 at 30K. Both wins at ~25.7K (coverage threshold). Needs 50K comparison. |
| 444a | Graph on FT09 | — | — | DEAD. Threshold mismatch — 32 states collapse to 1 node at 0.99. Fixable. |
| 444b | Graph on P-MNIST | 93.34% (labels), 10.1% (no labels) | — | 1-NN via 5000 nodes. Same label dependency as codebook (U26). |
| 445 | Graph 50K reliability (10 seeds) | — | 25% | 3/10 at 50K. Steps=[25708, 25738, 44020]. Codebook: 6/10, median 19K. Graph: half reliability, no fast seeds, systematic not bimodal. |
| **446** | **Grid graph (no cosine)** — random proj→8D, percentile bins, fixed cells | — | **25%** | 0/3 at 10K. Dynamics healthy (dom=25%, unique=1544, kill criteria not hit). Edge mechanism intact without cosine nodes. |
| 446b | Grid graph 30K | — | 25% | **0/3 at 30K.** unique=1869, edges=6200. Cosine graph navigated at same timescale (25.7K). Grid has similar node count (~1877 vs ~1984) but doesn't navigate. Data-aligned partitioning may be load-bearing. |
| 447 | PCA grid graph (data-aligned projection) | — | 25% | **0/3 at 30K. unique=539 (WORSE than random 1869).** PCA concentrates variance → fewer cells. Three-way comparison: random 0/3 (1869 cells), PCA 0/3 (539 cells), cosine 1/3 (1984 nodes). Only adaptive placement navigates. **Adaptive spatial partitioning confirmed necessary for graph navigation. Adaptive placement IS the codebook mechanism (attract + nearest-prototype). Graph navigation is codebook-mediated.** |
| 448 | Reservoir-graph hybrid (sign(h) → cells) | — | 26% | **0/3 at 10K. unique/steps ratio=0.942.** Temporal inconsistency confirmed: 94% of steps produce unique cell. Edges are singletons. Reservoir and graph are incompatible — temporal state prevents revisitation needed for edge accumulation. |
| 449 | CA-graph (Rule 110 + edges) | — | 25% | 1/10 at 10K (449b). **RETRACTED: CA rule irrelevant (449c: all rules identical). Binarize-only gives same result (450). Pure random walk also 1/10 (451, step 1329 = 3x faster). CA-graph IS random walk.** |
| 450 | Binarize-only graph (no CA) | — | 25% | 1/10, seed 2, step 4081 — identical to CA. CA transformation adds nothing. |
| 451 | Pure random walk baseline 10K | — | — | 1/10, seed 8, step 1329. Random walk FASTER than CA-graph. Navigation at this rate is seed luck. |
| **452** | **kd-tree partition graph (family #5)** | — | **25%** | **0/3 at 30K. KILL.** Healthy dynamics (dom=25%, ratio=0.024, ~730 leaves). Adaptive density via splitting: ✓. Navigation: ✗. Three death modes: (1) edge resets on split destroy accumulated action statistics, (2) axis-aligned splits misclassify similar observations, (3) cells are ephemeral (split→destroyed). |
| **453** | **LSH graph (fixed random hyperplanes, k=10)** | — | **25%** | **3/10 at 30K. NAVIGATES.** Steps=[4997, 14737, 18244]. Zero codebook DNA. 89-354 unique cells (of 1024). Properties 1+2+4 sufficient. Cell persistence is the key variable. |
| **454** | **LSH reliability (50K, 10 seeds)** | — | **25%** | **4/10 at 50K.** Steps=[4997, 14737, 18244, 38345]. Beats cosine graph (3/10 at 50K, Step 445). Median step ~18K (comparable to cosine ~19K). LSH is a legitimate non-codebook navigation substrate. |
| **455** | **LSH on FT09** | — | **17%** | **DEGENERATE. 1 unique cell.** All FT09 observations hash to same 10-bit code. dom=17%=1/6=random. k=10 too coarse for FT09's visual structure. **P3 (adaptive density) is not needed for single-game navigation but IS needed for cross-game generality.** The codebook's attract mechanism provides per-game adaptation that fixed LSH cannot. |
| **456** | **Multi-resolution LSH (time-based growth)** | — | **25%** | **1/3 at 30K. Growth HURTS.** Success at step 4997 (before first growth event at 5K). Growth creates exponential cell fragmentation: 348→12206 occupied at k=16. Most cells visited once → no argmin signal. ratio=0.31 (vs fixed 0.003-0.012). Death mode: cell fragmentation from exponential doubling. |
| **457** | **LSH durability 200K (fixed k=10)** | — | **25%** | **Convergence bifurcated by navigation.** Seeds that navigate early: signal stable (0.11-0.13), cells keep growing. Stuck seeds: signal drops 58% by 200K, convergence is real. U25-U17 coupling confirmed: navigation success is self-reinforcing (new cells → fresh signal), stagnation is self-reinforcing (cell exhaustion → convergence). No growth mechanism needed before 50K. The question shifts from "prevent convergence" to "increase reliability from 4/10." |
| **458** | **LSH k-sweep (k=8,10,12,14)** | — | **25%** | **k is NOT the reliability lever.** All produce 1-2/3 at 3 seeds. BUT k=12 has 2x signal quality (0.236 vs 0.116) and the fastest navigation ever: step 471 (seed 0). Occupancy: k=8=42%, k=10=17%, k=12=6%, k=14=2.5%. Sparser cells = less argmin ambiguity = faster discrimination. Reliability lever is not cell count but startup discriminability — how quickly early edge counts differentiate actions. |
| **459** | **LSH k=12 reliability (10 seeds, 50K)** | — | **25%** | **6/10 at 50K. NEW LSH BASELINE.** Steps=[471, 16771, 19604, 27010, 35607, 41905]. k=12 beats k=10 (4/10) by 50%. Signal quality: navigating seeds all sig_q>0.170, non-navigating all <0.165. Finer cells (4096 max, 339 occupied, 8.3% occupancy) = less argmin ambiguity = faster discrimination. Step-471 win confirmed (seed 0). |
| **460** | **Reservoir-LSH hybrid (sr=0, 0.5, 0.9)** | — | **25%** | **NAVIGATES despite 70-90% chg_rate.** sr=0: 1/3 at 30K (step 13207). sr=0.5: 0/3. sr=0.9: 1/3 (step 12703). Step 448's failure was hashing resolution (256-bit=94% unique), not reservoir output. k=10 LSH coarsening rescues local continuity. Signal quality HIGHER at sr=0.5/0.9 (0.29-0.31 vs 0.12). Reservoir family NOT dead — was killed by wrong evaluation. Experiment 9/20 of reservoir family. |
| **461** | **Reservoir-LSH k=12 (sr=0, 0.9)** | — | **25%** | **No improvement from k=12.** sr=0: 1/3 (step 22634). sr=0.9: 1/3 (step 430). Signal quality up (sr=0.9: 0.403) but reliability unchanged. k is not the lever for reservoir either. Occupancy 25% at sr=0.9 (vs 8% for pure-LSH k=12) — reservoir dynamics explore h-space more broadly. Experiment 10/20 of reservoir family. |
| **462** | **LLM benchmark (haiku/sonnet/opus on LS20)** | — | — | **0/3 models navigated. 0 levels.** Haiku: 97 steps, 100% dom ACTION1, wrong mental model. Sonnet: 4 steps, spatial awareness but cheated. Opus: 11 steps, confused, gave up. **KEY FINDING: LLMs lack systematic exploration mechanism.** Intelligence without exploration = exploitation. |
| **463** | **Reservoir-LSH dim sweep (32-256)** | — | **25%** | **res_dim NOT the lever.** 1/3 across all dims. Root cause: chg_rate=85-90% regardless of dim → occupancy 37-49% → cells flood → no argmin signal. Experiment 14/20 of reservoir family. |
| **464** | **Reservoir quantization (q=0.5, 0.1, 0.05)** | — | **25%** | **Quantization BACKFIRES.** Coarser quant increases chg_rate (discrete jumps interact badly with LSH). 0/3 at q=0.5/0.1. Only q=0.05 (≈no quant) matches baseline 1/3. Experiment 15/20. |
| **465** | **Low-sr reservoir sweep (0.0-0.2)** | — | **25%** | **RESERVOIR KILLED at 20 experiments.** chg_rate monotone increasing with sr. Every unit of memory makes navigation worse. sr=0.0 (no recurrence) is best at 1/3 — but that's tanh(W_in @ obs), a random nonlinear projection, not a dynamical system. Reservoir family dead as temporal architecture. |
| **466** | **tanh(W_in) vs centered_enc encoding shootout** | — | **25%** | **centered_enc WINS.** A (centered_enc): 3/5 at 50K, sig_q=0.211. B (tanh nonlinear): 2/5, sig_q=0.113 (46% lower). Nonlinear projection loses discriminative structure by compressing 256D→64D. Reservoir/nonlinear thread fully closed. centered_enc + k=12 LSH is the confirmed optimal configuration. |
| **467** | **LSH k=12 on FT09** | — | **17%** | **Still degenerate. 1 cell / 4096.** avgpool16 + centered_enc produces identical vectors for ALL FT09 frames. Not a k problem — preprocessing destroys FT09's visual structure. |
| **468** | **LSH k=12 on VC33** | — | **100%** | **Degenerate. 1 cell / 4096.** Same as FT09. Cross-game summary: LS20 6/10, FT09 0/5 (1 cell), VC33 0/5 (1 cell). |
| **469** | **Raw 64x64 + k=16 LSH on FT09** | — | **17%** | **Still degenerate. 1-2 cells even with raw 4096D.** The GAME is frozen — no visual variation without correct clicks. Task-level constraint. |
| **470** | **Raw 64x64 + k=16 LSH on LS20** | — | **25%** | **1/5 at 50K.** LSH handles 4096D but avgpool16 is better (6/10 vs 1/5). avgpool is needed signal compression, not codebook-specific. |
| **471** | **Diff-frame encoding + k=12 on LS20** | — | **25%** | **2/5 at 50K.** Diff frames non-degenerate (453 cells) but weaker than baseline (6/10). Transitions less discriminable than positions. |
| **472** | **Concat encoding (pos+diff) + k=12 on LS20** | — | **25%** | **2/5 at 50K.** Extra dims disperse LSH partitioning. U13 confirmed for encoding. |
| **473** | **Multi-hash LSH (L=1,3,5 tables) on LS20** | — | **25%** | **ALL 2/5. Neutral.** LSH at plateau. |
| **474** | **L2 k-means centroids + graph (FAMILY #8)** | — | **25%** | **5/5 at 50K** (lucky seeds). L2 nearest-centroid (n=300, frozen after 1K warmup). Data-aligned. Near-full occupancy (97%). |
| **475** | **L2 k-means reliability (10 seeds)** | — | **25%** | **5/10.** Seeds 0-4 win, 5-9 fail. 474's 5/5 was lucky. L2 k-means = LSH (6/10) in reliability, not better. The 5-6/10 ceiling is in the argmin mechanism, not the partitioning. |
| **476** | **L2 k-means on FT09 + VC33** | — | **varies** | **FT09: 32 visual states, 0/3. VC33: 50 visual states, 0/3.** k-means captures game structure perfectly but argmin can't discriminate actions. Action selection is the bottleneck. |
| **477** | **Softmax action selection (T=0.1, 1.0, 10.0) on k-means** | — | **25%** | **Argmin wins. Stochasticity hurts.** Deterministic tie-breaking is optimal. |
| **478** | **Destination-novelty action selection on LSH** | — | **25%** | **1/10. Much worse.** Local exploration trap. |
| **479** | **UCB1 action selection (C=0.5-5.0) on LSH** | — | **25%** | **All C identical = argmin.** UCB degenerates. |
| **480** | **Projection selection (10 probes, pick best sig_q)** | — | **25%** | **1/5. sig_q at 2K not predictive.** Trajectory luck is upstream of hash function. |
| **481** | **Prediction-error action selection** | — | **25%** | **0/10. Smart exploration KILLS navigation.** Local trap. Argmin's uniform coverage is optimal for any single mechanism. |
| **482** | **Global cell novelty (anti-revisitation penalty)** | — | **25%** | **6/10, different seeds.** Complementary to argmin. |
| **483** | **Ensemble (argmin + global novelty parallel)** | — | **25%** | **6/10. Appeared to be game ceiling.** Seeds 1,2,5,6 fail for both mechanisms at 50K. |
| **484** | **Hard seeds (1,2,5,6) at 200K** | — | **25%** | **4/4 NAVIGATE.** 6/10 was step budget artifact. Hard seeds need 35K-115K. |
| **485** | **LSH k=12 at 120K, 10 seeds** | — | **25%** | **9/10.** Mechanism is ~100% reliable given sufficient budget (5K-150K per seed). |
| **486** | **Multi-level (no reset, 200K)** | — | **25%** | **L1: 5/5. L2: 0/5.** Graph contamination from Level 1 edges. |
| **487** | **Multi-level with edge reset** | — | **25%** | **L1: 3/3. L2: 0/3.** Edge reset insufficient — the hash CELLS are the problem. Level 1's random projection doesn't discriminate Level 2's visual structure. ~2300 game-overs = agent IS exploring Level 2 but cells don't provide useful navigation. I1 (discover own representation) fails. Fixed LSH can't adapt across levels. |
