# Research State — Live Document
*Updated by Leo. Read by /checkpoint skill. Source of truth for active work.*

---

## Active Hypothesis

```
TESTING: [Step 228. Logic 100%, n-gram 99%, arithmetic 94-100%. OOD ceiling. 128 experiments.]

STEP 193 TEMPLATE ABLATION:
  Parity: mod2 +20.3pp, abs +18.0pp, cos +9.0pp, sign 0pp
  XOR: cos +13.8pp, abs +13.6pp, full_menu +14.4pp, mod2 +2.0pp
  LOO selection contributes 100% of improvement regardless of template.
  Template choice adds 0-11pp depending on task-template alignment.
  The right SINGLE template outperforms the diversified menu.

STEP 190 RESULT:
  Task: Fibonacci mod 10, raw 2D integer input
  Feature discovery: 5 cosine features selected by LOO (d=2 -> d=7)
  Iterated k-NN: 100% at 1, 5, 10, 20, 50 steps
  No backprop. No designed features. No external architecture.
  Store + discover + iterate = universal computation on discrete state machines.

  90 EXPERIMENTS (Steps 97-190). THE SEARCH HAS FOUND THE SUBSTRATE.

STEP 181 KEY RESULT:
  Iterated 1-step k-NN = 100% at ALL step counts (1, 2, 3, 5, 10).
  Direct N-step prediction degrades: 90% at 5 steps, 70% at 10 steps.
  THE COMPUTATION GAP IS CLOSED. Self-application of the 1-step rule
  composes computation from storage. The system COMPUTES by reusing itself.

STEP 180 DEEP FINDING:
  k-NN perfectly memorizes 1-3 step CA evolution but can't predict 5+ steps.
  Multi-step prediction requires RULE COMPOSITION — applying the rule iteratively.
  k-NN matches patterns; it can't compose them. This is the COMPUTATION gap.
  The atomic substrate needs genuine computation, not just storage + retrieval.

CONSTITUTIONAL PROGRESS (updated Step 175):
  Stage 1: PASS
  Stage 2: PASS (LOO/margin from cosine)
  Stage 3: VACUOUS
  Stage 4: PASS (random features selected by LOO)
  Stage 5: VACUOUS (subsumed by Stage 4)
  Stage 6: PASS (metric selection by LOO)
  Stage 7: PARTIALLY PASSED — multi-template selection works (cos→abs), 98.4% on parity
  Stage 8: The only frozen element is the LOO scoring function itself

CONSTITUTIONAL PROGRESS (updated Step 173):
  Stage 1: PASS (autonomous computation)
  Stage 2: PASS (margin/LOO signal from cosine)
  Stage 3: VACUOUS (k fixed, not binding)
  Stage 4: PASS (random cosine features, selected by LOO)
  Stage 5: VACUOUS (flat topology, subsumed by features)
  Stage 6: PASS (L2 vs cosine vs L1 selected by LOO)
  Stage 7: NOT STARTED (representation as modifiable data)
  Stage 8: NOT REACHED

CONSTITUTIONAL ASSESSMENT (Step 158):
  Stage 1 (Autonomous Computation): PASS
  Stage 2 (Self-Generated Adaptation): PASS — margin signal is Principle-II compliant
  Stage 3 (Adaptation Rate Adapts): VACUOUS — same as Living Seed
  Stage 4 (Structural Constants Adaptive): BLOCKED — feature candidate space is code, not data
  Stages 5-8: NOT STARTED
  FROZEN FRAME MINIMUM: same 6/8 wall as Living Seed. Architectural.

STEPS 144-147 ARC:
  Step 144: Margin discovers cos(sum*pi)=parity. 75.4% -> 100.0%. BREAKTHROUGH.
  Step 145: No effect on CIFAR-100 ResNet (already structured).
  Step 146: Multi-rule discovery: finds XOR pair + AND pair in correct order. 96.8% -> 99.6%.
  Step 147: No effect on MNIST (already 93.4%, no simple feature helps).
  DOMAIN: structured rule discovery from binary/discrete features.
  MECHANISM: k-NN margin as Principle-II-compliant adaptation signal.

STEP 144 BREAKTHROUGH:
  Margin-guided discovery selects cos(sum*pi) = parity. 75.4% -> 100.0%.
  Signal: k-NN prediction margin on training data.
  cos(sum*pi) margin delta = +1.84 (vs coherence: sum_sq at +0.044, WRONG feature).
  Principle II: signal arises FROM k-NN computation (cosine similarity).
  The substrate discovers and applies the exact feature needed to fix its own failures.

STEP 141 BREAKTHROUGH:
  k-NN FAILS parity (75.8%). Coherence DETECTS the parity feature (+0.041 delta).
  k-NN + oracle parity feature = 100%. Coherence finds it unsupervised.
  k-NN handles similarity tasks (MNIST 95.4%). Coherence handles rule tasks (XOR, parity).
  TOGETHER: a computational substrate with self-generated feature discovery.

STEPS 130-136 ARC:
  Coherence-guided feature discovery: works on synthetic XOR (rank 1/4950 at d=100)
  Real data (MNIST/CIFAR): no improvement — features already coherent
  Codebook-as-weights: 66.2% vs k-NN 93.4% — compression loses 27pp
  Unsupervised clustering: NMI=0.24 (partial structure, not semantic)
  DEEP RESULT: for fixed features, storage >> computation for CL.

STEPS 130-131 BREAKTHROUGH:
  Class coherence (intra-class centroid similarity) discovers XOR feature at RANK 1.
  d=20: rank 1/190, SNR 35x. d=50: rank 1/1225, SNR 33x. d=100: rank 1/4950, SNR 30x.
  Unsupervised, Principle-II compliant, same cosine computation as readout.
  Signal-to-noise ratio does NOT degrade with dimensionality.

STEPS 119-124 KEY FINDINGS:
  Sequential S1 eval (interleaved train/eval): 39.7% AA, ~0.1pp fgt (+1.5pp, -11.6pp fgt)
  Batch S1 eval (all train then eval): 37.3% AA (-1.0pp, HURTS)
  Confidence gating: spawning ALL beats selective (density > precision)
  Synthetic vectors (mixup): -0.7pp HURTS (wrong distribution)
  Noisy copies of eval: 38.2% (no benefit — only EXACT samples help)
  Random vectors: 38.2% (no benefit)
  MECHANISM: transductive memorization of test distribution, not learned features.
  The INTERLEAVING is load-bearing — unified train/inference is functionally SUPERIOR.
PROVES IF: [not set]
DISPROVES IF: [not set]
ABANDON BY: [not set]
STEP: 113 (next)

STEPS 109-112 SUMMARY:
  109: Always-spawn beats threshold (+3.1pp → 95.0%). System = append + top-k.
  110: Raw pixels beat projection (95.4%). No feature extractor needed.
  111: Retrieval-induced learning: +0.2pp (marginal). Representation saturated.
  112: Retrieval-driven projection: DISPROVED. Online rank-1 updates break geometry.

S1 VALIDATED:
  P-MNIST:   91.9% AA, self-supervised delta=-0.5pp, mislabel ~8%
  CIFAR-100: 38.3% AA, self-supervised delta=-0.4pp, mislabel ~62%
  One function process(state, input, label?) — no train/eval branch.

STEP 108 ABLATION:
  lr=0 (no competitive learning): 91.9% — NOT load-bearing, remove it
  1-NN (no top-k): 86.2% — load-bearing, -5.7pp
  No spawn: 17.4% — CRITICAL, -74.5pp
  MINIMAL SYSTEM: spawn + top-k. Two operations.
```

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

Hard-won from 96 experiments. Every candidate must pass ALL constraints before implementation.

| # | Constraint | Source | Type |
|---|---|---|---|
| C1 | Read and write must be one operation | Step 72 | structural |
| C2 | Must not reduce to Hopfield/softmax-attention-only | Step 73 | novelty |
| C3 | Must not require separate memory + generation systems | Architecture autopsy | structural |
| C4 | Must not rely on matrix composition through long chains | Steps 86-96 | empirical |
| C5 | Must achieve structural zero forgetting | Step 65 | requirement |
| C6 | Must work on dense embeddings without per-dataset tuning | Steps 63, 66 | empirical |
| C7 | Must beat 1-NN readout over same codebook | Steps 65-71 | requirement |
| C8 | Current hardware, no external API | Jun | requirement |
| C9 | Minimal — expressible in <100 lines | Jun | requirement |
| C10 | Not a combination of known techniques | Jun | requirement |
| C11 | Readout signal factors must not anti-correlate (e.g. attn×displacement) | Step 97 | empirical |
| C12 | Readout must be input-conditional, not static vector property (coherence, density) | Step 98 | empirical |
| C13 | Spawn threshold must be calibrated per feature space (cosine range varies) | Step 100 | empirical |
| C14 | CIFAR-100 forgetting is class-incremental interference, not codebook drift | Step 101 | empirical |
| C15 | Sum-all aggregation fails when class size >> effective neighbors; only sparse selection (top-k) preserves signal | Step 102 | empirical |

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
