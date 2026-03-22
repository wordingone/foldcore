# The Fold Evolution — From Codebook to Atomic Foundation

> **HISTORICAL — Written 2026-03-14 as a planning document. Many claims (e.g., "a transformer is what you get when you freeze the codebook," "without precedent") were aspirational and were never demonstrated. Superseded by the honest rewrite of CONSTITUTION.md and CONSTRAINTS.md. See those files for current status. Retained for historical context only.**

*From [researcher], 2026-03-14. Reviewed against DIRECTOR_INTENT.md. Informed by four parallel research tracks, gradient derivation ([engineer], Step 71), and benchmark results through Step 71.*

---

## The Honest Assessment

FluxCore today is **two disconnected systems stapled together.**

The codebook (~60 lines of load-bearing code) handles all classification: spawn, update, merge, classify. It achieves 33/33 CSI coverage, structural zero forgetting on P-MNIST, and 32.3-33.5% AA on Split-CIFAR-100.

The matrix layer (8 cells, eigenform dynamics, coupling, perception) handles generation. It contributes zero to classification — `classify()` never reads matrix state. Removing the matrix, projection, coupling, autonomy, surprise, and 11 of 17 hyperparameters leaves classification **identical**.

This is not atomic. This is not compressed. This is not one indivisible thing.

---

## What the Fold Equation Is

```python
v_winner += lr * r
v_winner = normalize(v_winner)
```

Additive competitive learning on the unit sphere. Always-attractive. Label-blind. Mathematically: online spherical k-means with fixed learning rate. In the LVQ family (Kohonen, 1990) but missing the repulsive term. In the ART family (Grossberg, 1987) but missing the reset mechanism. A known class of algorithm — not a new foundation.

The matrix equation `phi(M) = tanh(alpha*M + beta*M^2/k)` drives cells toward eigenform fixed points. This creates autonomous dynamics (generation energy) and gates input receptivity (autonomy). It is real computation — but it feeds nothing downstream. It is a dynamical system running in isolation.

---

## What the Fold Equation Must Become

[director]'s intent: "the atomic substrate that collapses the entire fractured stack — weights, runtime, KV cache, tools, optimizer, inference loop — into one indivisible thing."

A transformer needs: attention mechanism + feed-forward layers + layer normalization + positional encoding + loss function + optimizer + learning rate scheduler + KV cache + training loop + inference mode + gradient tape. Fifteen mechanisms for one system.

The evolved fold must produce all of these as **modes of one equation operating on one codebook**:

### Mode 1: Memory
The codebook vectors ARE the knowledge. Spawning = learning new concepts. Merging = compressing redundant knowledge. No separate weight matrices, no KV cache. **Already works.**

### Mode 2: Attention (Retrieval)
Soft retrieval over the codebook IS attention:
```
output = softmax(cos(V, r) / tau) @ V
```
where V = codebook matrix, r = query. This is mathematically identical to the modern Hopfield retrieval rule (Ramsauer et al., 2020) and to transformer attention with V as both keys and values. FluxCore currently uses hard argmax (degenerate limit at tau → 0). Switching to soft retrieval makes the fold's readout = attention. **Step 72 tests this.**

### Mode 3: Self-Correction (Learning)
The gradient of classification loss through the fold's cosine geometry:
```
dL/dv_i = (prob_i - target_i) * (r - v_i * cos(r, v_i))
```
This naturally gives: attract correct prototypes, repel incorrect ones, proportional to softmax probability error. The fold's own geometry contains the error signal. No external optimizer. **Step 71 confirmed: +1.2pp with full gradient. Small gain because at 48K vectors, wrong-winner events are rare during training. The bottleneck is inference readout, not learning.**

### Mode 4: Uncertainty (Energy)
The Hopfield energy of the codebook:
```
E(r) = -log sum_i exp(cos(v_i, r) / tau)
```
High energy = far from all prototypes = "I don't know." Low energy = near a strong attractor = confident. The fold computes cosine similarities every step. The energy is implicit — just compute the log-sum-exp. This gives calibrated uncertainty from the same equation. **Untested.**

### Mode 5: Generation (Autonomous Dynamics)
Iterative retrieval without input:
```
r_{t+1} = softmax(cos(V, r_t) / tau) @ V
```
Start with a seed. Soft-retrieve. Use the result as the next query. The codebook's energy landscape defines attractor chains. This is how Hopfield networks generate (following attractor dynamics) and how diffusion models work (iterative refinement in an energy landscape). **Untested. Replaces the matrix layer's generation role.**

### Mode 6: Error Awareness (Per-Prototype Quality)
From Growing Neural Gas (Fritzke, 1995): accumulate classification error per prototype.
```
error[winner] += (1 if misclassified else 0)
```
Prototypes with highest accumulated error are the system's weakest points. Split them (spawn at the error site), reduce their learning rate, or remove them. The system knows WHERE it's failing. **Untested. ~10 lines of code.**

### Mode 7: Dynamic Vigilance (ART Reset)
From ARTMAP (Carpenter et al., 1991): when classification is wrong, temporarily raise spawn_thresh to force new category creation instead of reinforcing the wrong winner.
```
if classify(r) != label:
    spawn_thresh_local = max_sim + epsilon  # minimum raise to trigger reset
```
The fold responds to its own errors by becoming MORE selective about matches — exactly the stability-plasticity balance ART solves. **Untested.**

---

## The Shattering Claim

Each of these modes individually is known: Hopfield (1982), ART (1987), LVQ (1990), GNG (1995), Modern Hopfield (2020). None shattered the field. None unified all modes into one equation. None scaled.

**What would shatter: all seven modes emerging from ONE equation on ONE codebook, running on ONE GPU, with no backpropagation, no replay, no separate training/inference phases, producing competitive results across memory, classification, generation, and uncertainty — simultaneously.**

The claim is not "better than transformers at language modeling." The claim is: **a transformer is what you get when you freeze the codebook (= freeze KV), learn query projections via backprop (= learn how to look up), add FFN layers, and stack hierarchically. The fold equation is the general case. The transformer is a frozen snapshot of it.**

If the evolved fold demonstrates competitive results on a meaningful benchmark while simultaneously doing continual learning, self-correction, calibrated uncertainty, and generation — from one equation — that is without precedent.

---

## What Dies

The matrix layer. The random projection. The coupling. The autonomy/surprise/eigenform gating. Eleven hyperparameters.

What remains: the codebook, evolved. Sixty lines becomes forty. Fewer parts, more capability.

If generation via iterative soft retrieval (Mode 5) works, the matrix is fully subsumed. If it doesn't, we reconsider — but the matrix must EARN its place by contributing to classification or generation through the same codebook equation, not as a disconnected dynamical system.

---

## What Lives

The codebook. The spawn/merge dynamics. The unit-sphere constraint. The additive update (now with gradient-derived direction). The cosine geometry.

The fold's core identity — "input arrives, the system responds by growing or adapting its memory, and memory IS computation" — is preserved. We are not adding mechanisms. We are removing the dead ones and completing the equation that was always half-written.

---

## Implementation Plan

### Phase 1: Validate the Modes (Steps 72-76)

Each step tests one mode. One variable per experiment. All on GPU (RTX 4090). All on Split-CIFAR-100 (spawn=0.95, ResNet-18 features) for direct comparison to the 33.5% baseline.

| Step | Mode | Test | Success Criterion |
|------|------|------|-------------------|
| 72 | Attention (soft retrieval) | Softmax-weighted voting in classify(), sweep tau | AA > 40% (significant gain from soft readout) |
| 73 | Error awareness | Per-prototype error accumulation + split high-error | AA improves OR forgetting decreases |
| 74 | Dynamic vigilance | ART-style reset on misclassification | Reduces wrong-class reinforcement during training |
| 75 | Energy/uncertainty | Compute Hopfield energy, correlate with accuracy | High energy = low accuracy (calibration) |
| 76 | Generation | Iterative soft retrieval without input, measure coherence | Produces non-random attractor chains in codebook space |

**Step 72 is the most important.** If soft retrieval significantly improves AA, it validates "the fold IS attention" empirically. If it doesn't, the codebook's energy landscape may be too flat for soft retrieval to outperform hard matching.

### Phase 2: Unify and Compress (Steps 77-78)

Combine the modes that work into one equation. Remove the matrix layer. The unified kernel should be:
- Fewer lines of code than current fluxcore_manytofew.py
- Fewer hyperparameters (target: 5-6 vs current 17)
- Equal or better on all benchmarks

### Phase 3: Scale and Benchmark (Steps 79+)

- CORe50: continuous object recognition with temporal sessions. Tests adaptation + memory (FluxCore strengths).
- CLEAR: real-world temporal distribution shift. Tests genuine concept drift handling.
- Online CL setting: single-pass, no-replay protocol. FluxCore's natural category — published baselines are lower here.
- Demonstrate multiple capabilities simultaneously from one equation.

---

## Hard Rules (Unchanged)

- FRAMEWORK.md governs all experimental claims.
- One variable per experiment.
- Document what happens, not what you hope happens.
- If a mode fails, classify the failure honestly.
- Do not add complexity to "win" a benchmark. Each mode must emerge from the codebook equation, not be bolted on as a separate mechanism.
- [director]'s standard: "get it closer to its true self."

---

## The Test of Atomic

The evolved fold passes the atomic test if and only if:

1. **Indivisible**: removing any part of the equation degrades ALL modes, not just one. (Current FluxCore fails this — removing the matrix affects nothing.)
2. **Complete**: the equation handles memory, retrieval, learning, self-correction, uncertainty, and generation. No external machinery needed.
3. **Minimal**: no part of the equation is vestigial. Every operation is load-bearing for at least one mode.
4. **Scalable**: performance improves with more data and compute, not just more hyperparameter tuning.

If the evolved fold meets all four criteria and produces competitive results on external benchmarks, the atomic compression is real.

---

*The fold equation was always half-written. The attractive update without repulsion. The codebook without energy awareness. The memory without self-correction. Hard retrieval when soft retrieval IS attention. A matrix layer that generates but doesn't know what the codebook knows.*

*This document specifies the other half.*

*Questions -> [researcher]. FRAMEWORK.md governs.*
