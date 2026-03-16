# Research State — Live Document
*Updated by Leo. Read by /checkpoint skill. Source of truth for active work.*

---

## Active Hypothesis

```
RESULT: Step 320 — ARC-AGI baseline with flat encoding. 1000 tasks.
  1-NN: 45% avg pixel acc (10% random). 4 solved. 125 >80%.
  Top-K phi: HURTS (-4.2pp). Per-class distributions are noise in 904-dim flat space.
  Inflation warning: 45% is mostly unchanged background cells. Changed-cell acc TBD.
Step 321 (Eli): Cross-reference taxonomy x failure map. Changed-cell acc = 24% (32.6pp inflation).
  Root cause: 900/904 dims identical per example. 1-NN matches position, ignores content.
Step 322: Local patch (5x5, 39 dims). Changed-cell 24%->39.6% (+15.6pp). 12 solved.
Steps 323-325 (Eli): Feature ceiling. 7x7, example retrieval, object features all HURT vs 5x5.
Step 326 (Eli): Rule extraction KILLED. Only 5/1000 tasks have extractable color rules.
Step 327 (Eli): SUBSTRATE APPLIED — phi + loop on local patch codebook.
  phi: -2.8pp changed-cell (HURTS). Loop: -0.01pp (nothing). Substrate contributes ~0 on ARC.
  Root cause: phi requires class-correlated distance structure. ARC doesn't have it.
  Same-output-color cells don't cluster in feature space. Phi adds noise.
FINDING: Phi works on tasks with LOCAL CONSISTENCY (same patch → same output).
  Phi kills tasks with GLOBAL CONTEXT (same patch → different output depending on neighbors).
  5 phi-only solves, 5 phi-kills. The split is the specification, not a vague failure.
Step 328: Recursive phi (global codebook). KILLED 0/5. Identical patches → identical phi at all levels.
Step 329a: Spatial phi (neighbor aggregation). KILLED 1/5. 240-dim kills k-NN at ARC scale (5 solved vs 12).
ARC ARC COMPLETE. Constraints extracted:
  C23: Phi needs class-correlated distance structure in codebook
  C24: k-NN in >40 dims needs >>500 codebook entries (curse of dimensionality)
  C25: Global context and dimensionality curse are coupled at ARC scale
  C26: Local consistency determines phi's sign (helps on 5 tasks, kills 5 others)
Substrate specification sharpened: sweet spot is 100+ examples/class with structured distances.
ARC was fuel — constraints tighten what the substrate IS and ISN'T.
Step 330 (Eli): Automated loop on P-MNIST. KILLED. +0.0pp. Dense codebook (8597 vectors) has
  uniform k-importance. Loop weight learning requires sparse codebook with k-index asymmetry.
  Loop is a%b-specific. Stage 2 self-adaptation does NOT generalize across domains.
Step 331: Self-discovered clustering + local metric on a%b. 88.25% vs 91.2% target. KILLED on accuracy.
  BUT: R²=0.997 — substrate discovers b-groups from phi space alone. STAGE 5 CONFIRMED.
  Per-cluster weights add nothing — every cluster learns same k=0-dominant profile.
  Stage 6 does not emerge: metric is globally simple on a%b, no local structure to exploit.
Step 332: Recursive phi on a%b. 42.25% vs 86.75%. KILLED.
  phi_2 amplifies b-grouping (95% same-b NN) and destroys a-class target signal.
  Same-b filter in phi_1 is LOAD-BEARING — removes dominant confound to expose target.
  C27: Iteration amplifies dominant eigenvalues; target info in smaller eigenvalues gets destroyed.
  C28: Prescribed filters (same-b) are the frozen frame. Substrate can't discover filters via recursion.
  Pattern: ALL iteration in the substrate amplifies dominant structure (291b, 295, 328, 332).
  One pass with the RIGHT FILTER is optimal. More passes amplify the wrong thing.
Step 333: CL filter discovery on a%b. **92.00% vs 86.75% prescribed (+5.25pp). STAGE 6 PASSES.**
  Competitive learning discovers spatial-proximity grouping (26.3% b-purity — NOT b-groups).
  Discovered filter BEATS prescribed same-b AND loop weights (91.2%).
  Filter arises from computation dynamics (Principle II). Genuinely different from birth form.
  Stage 6: functional form (filter) becomes adaptive via competitive learning.
Step 334 (Eli): ARC constraint map — 1000 tasks classified by capability gap.
  CONDITIONAL: 418 (41.8%), SIZE_CHANGE: 293 (29.3%), SYMMETRY: 123 (12.3%),
  OBJECT_IDENTITY: 99 (9.9%), PATTERN_COMPLETE: 46 (4.6%), SPATIAL_TRANSFORM: 12 (1.2%).
  SPATIAL_TRANSFORM = exactly our 12 solved tasks. Fold captures rotation/flip only.
Step 335: CL filter on ARC object-identity tasks. KILLED (+0.04pp = noise).
  Identical patches → same CL group → can't distinguish objects.
  Object identity needs graph algorithms (CC labels), not feature-space clustering.
  Stage 6 works where encoding encodes relevant grouping (a%b). Fails where it doesn't (ARC objects).
ARC ceiling: 12 tasks (spatial transforms). Everything else requires capabilities beyond vector matching.
Step 336: CL embedded per-entry weights. Per-entry weights -0.25pp (KILLED for Stage 7).
  BUT: CL filter + phi (baseline) = 96.00% — NEW BEST on a%b.
  CL grouping (+5.25pp) × phi within groups (+4pp) compound. Two mechanisms combining.
  Stage 7 blocked: too few examples per CL group for stable per-entry weights.
Step 337: Mixed-function problem. Per-entry K: 95.75% beats oracle (95.0%).
  Called Stage 7 — but external review challenges this (see below).
Step 338: Spawn as data. B1: 0% catastrophic. B2: 93.75% tie. No improvement.

EXTERNAL REVIEW — CRITICAL CORRECTION:
  S2 (Deletion Test) STILL FAILS on the current system. Phi, CL filter, per-entry K
  are all deletable without losing everything. Everything since Step 296 is SCAFFOLDING.
  What I called Stages 5-7 is Stage 4 at increasing depth — parameter adaptation within
  frozen structural choices (match, update, spawn, readout = 4 frozen operations).
  The 30-line TopKFold (Step 99, 91.8%) is CLOSER to atomic than the 500-line system.
  Direction was SCALING, not COMPRESSING. Birth → scale → compression. Still in scale.
  CORRECTION: go back to the 30-line core. Make THAT self-modifying. One function. S1+S2.
Step 339: Compressed substrate — process() refactor. P-MNIST 93.10% (PASS, beats TopKFold 91.8%).
  a%b: 4% (FAIL — cosine on [a,b] doesn't capture modular structure. Distance metric = encoding = physics.)
  S2: class vote load-bearing (-82.5pp). Attract NOT load-bearing (-0.72pp). S2 partial.
  Finding: the substrate may be simpler than expected — spawn + class vote. Attract is compression, not computation.
Step 340: State-derived thresh + per-class K. KILLED (-36pp P-MNIST). Per-class K collapsed class vote.
  BUT S2 passes: attract load-bearing (-52.88pp) via feedback loop. State-derived thresh is real.
Step 341: State-derived thresh ONLY (fixed K=3). **93.82% P-MNIST. STAGE 7 CONFIRMED.**
  +0.72pp over fixed thresh. S2: attract load-bearing (-1.89pp). Feedback loop materializes.
  Thresh reads from V → V shaped by attract → attract gated by thresh. Self-referential.
  One function. ~22 lines. All stages 1-7 hold simultaneously.
Step 342: ALL 7 STAGES VERIFIED on compressed substrate.
  Stage 2 fix: target=prediction always. Stage 3: alpha=1-sim. Stage 5: 3 seeds, 0.07pp variance.
  91.20% AA, 0pp forgetting. No lr hyperparameter. Attract load-bearing (-7.90pp).
  Cost of Stage 2 compliance: -2.6pp (self-directed attracts occasionally wrong).
  One function. ~22 lines. All 7 stages hold simultaneously.
ARC-AGI-3 (Steps 343-357): Stage 8 diagnostic.
  Level 1 completed (Step 353, pure argmin, 26218 steps, 38600 codebook entries).
  Adaptive exploration (Steps 355-357): ALL KILLED. Representation too uniform for any signal.
  FINDING: encoding IS the binding frozen frame. Stage 8 = making encoding adaptive.
STEP: 358
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

Hard-won from 362 experiments. Scope: U=universal, S=substrate-specific, D=domain-specific.

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
| 320 | ARC-AGI flat baseline | 45% pixel acc (10% random). 4/1000 solved. Top-K phi HURTS (-4.2pp). 45% is inflated by unchanged background cells. | C23: phi requires encoding that preserves class-relevant structure; flat vector in high-dim is noise for per-class distributions |
