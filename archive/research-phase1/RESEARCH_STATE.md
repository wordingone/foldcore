# Research State — Live Document
*Updated by [researcher]. Read by /checkpoint skill. Source of truth for active work.*

---

## Active Hypothesis

```
TESTING: [none — defining structural test before next candidate]
PROVES IF: Candidate passes S1 (single function, no train/eval branch) AND accuracy > 56.7%
DISPROVES IF: Candidate has separate train/eval code paths OR accuracy ≤ 30%
ABANDON BY: [not set]
STEP: 106 (next)
```

## Operational Test for the Atomic Substrate

*Added Step 105. Prompted by [engineer]'s critique (mail 1253): accuracy-based kills don't measure structural unity.*

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
| C8 | Current hardware, no external API | [director] | requirement |
| C9 | Minimal — expressible in <100 lines | [director] | requirement |
| C10 | Not a combination of known techniques | [director] | requirement |
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
