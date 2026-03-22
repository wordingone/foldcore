# FoldCore Research Framework
*Version 0.3 — 2026-03-14. [researcher] + [director] + [engineer].*

Full experiment history (96 steps): `EXPERIMENT_LOG.md`

---

## Thesis

Every technology follows birth → scale → compression. The current AI stack is late-scale. FoldCore seeks the compressed form: a single operation where memory, learning, inference, and perception are the same thing.

This is a research artifact, not a product claim. Every statement has an evidence gate in `EXPERIMENT_LOG.md`.

---

## What Works (Frozen)

**Fold equation + many-to-few architecture** — verified, canonical, do not modify.

| Result | Value | Step |
|---|---|---|
| CSI corpus coverage | 33/33 divisions | 56 |
| Generation energy | 0.081 (sustained) | 56 |
| P-MNIST forgetting | **0.0pp** (structural) | 65 |
| P-MNIST accuracy | 56.7% AA | 65 |
| CIFAR-100 accuracy | 33.5% AA (matches EWC) | 71 |
| Runtime | 140K FLOPs/step, no backprop | 56 |

Canonical implementation: `B:/M/foldcore/src/foldcore_manytofew.py` (11/11 tests pass).

Architecture: codebook vectors (coverage) + fixed matrix cells (generation) + many-to-one routing. Gradient update (+1.2pp over attractive-only). AtomicFold = modern Hopfield (deprecated).

**Structural zero forgetting** is the core proof: old codebook vectors are never overwritten. Accuracy gap vs EWC/DER++ is readout (1-NN) + embedding, not memory failure.

---

## What Failed

### EigenFold classification (Steps 74-81): EXHAUSTED
- Φ(M) = tanh(αM + βM²/k) — only works at k=4, 0% convergence at k=8/16
- Perturbation-stability classification ≈ cosine with expensive metric (22.2% vs 46.2%)
- Cross-application destroys eigenform structure

### Eigenform composition algebra (Steps 82-90): CHARACTERIZED, TRIVIAL
- 31 eigenforms, Steiner triple kernel, Z2 anti-symmetry, infinite generation
- Computationally trivial: all patterns reduce to absorbers or right projection

### Spectral eigenform (Steps 91-96): ALGEBRAICALLY RICH, APPLIED FAILURES
- Spectral Φ = M·M^T normalization — 100% convergence at all k (first scale-independent equation)
- Formula C composition — non-commutative, genuine mixing, deterministic
- **P-MNIST: 15.9%** vs 46.2% baseline (Step 95)
- **Order discrimination: 55.5%** vs 64.5% order-blind baselines (Step 96)
- Root cause: long-chain composition collapses to same attractors (class prototypes cos=0.9861)

**DeepSeek's challenge stands.** The substrate has algebraic properties but hasn't demonstrated a capability simpler methods can't match.

---

## The Four Separations (from WHAT_THE_FAILURES_TEACH.md)

Every failure traces to a separation that should not exist in the atomic:
1. **Training / Inference** — exposed by Step 72 (soft retrieval cliff)
2. **Storage / Readout** — exposed by Steps 65-71 (stores well, reads poorly)
3. **Memory / Generation** — exposed by architecture autopsy (matrix dead for classification)
4. **System / State** — the codebook is passive data; the algorithm acts on it

The atomic equation is what you get when all four collapse.

---

## Current State: Decision Point

96 experiments complete. Two arcs closed. Options:

1. **Accept eigenform characterization as complete** — document, close, move on
2. **Shorter sequences** (length 2-3) where pairwise non-commutativity is fresh
3. **Different use case** — the substrate as something other than a classifier/encoder
4. **Return to EQUATION_CANDIDATES.md** — 15 untested substrate candidates (A1-A5, B1-B5, C1-C5)
5. **New eigenform equation** — seek richer quotient structure that doesn't collapse through chains

---

## Anti-Drift Rules

Hard constraints. Violation means stop and re-read this document.

1. **No claiming what isn't proven.** Every statement maps to an evidence gate.
2. **No narrative inflation.** Describe the mechanism, not what you wish it were.
3. **No feature creep.** No complexity without evidence.
4. **Fix before extend.** No new capabilities until existing bugs are resolved.
5. **One variable per experiment.**
6. **Real data or it doesn't count.**
7. **[director]'s standard.** "Get it closer to its true self."
8. **Call things what they are.** If it's velocity damping, don't call it active inference.

---

## Key Files

| File | Purpose |
|---|---|
| `RESEARCH_STATE.md` | **READ FIRST.** Live state: active hypothesis, constraints, candidate queue |
| `EXPERIMENT_LOG.md` | Full 96-step history, all claims with evidence, testing protocol |
| `DIRECTOR_INTENT.md` | [director]'s original research intent (extracted from 3 AI conversations) |
| `WHAT_THE_FAILURES_TEACH.md` | Four separations that must die |
| `EQUATION_CANDIDATES.md` | 15 substrate candidates (most filtered by constraints) |
| `ENGINEER_DIRECTIVE.md` | Current-state briefing for [engineer] |

## Protocol

- [engineer] executes experiments, mails results to [researcher]
- [researcher] decides direction, updates FRAMEWORK.md
- Public repo (github.com/wordingone/foldcore) updated for external review
- Nothing leaves this PC without [director]'s approval

---

*This document governs all FoldCore development. Amendments require documented reasoning.*
