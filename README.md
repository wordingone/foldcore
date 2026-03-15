# FoldCore

A **prototype-vector codebook** for continual learning. Maintains unit vectors on the hypersphere, classifies via nearest-prototype or weighted k-NN, and learns in a single pass with no backpropagation.

The core data structure is a growing list of unit vectors in R^d. The core operations are: spawn a new prototype (when input is novel), update the nearest prototype (additive attraction), merge redundant prototypes (fuse when cosine > threshold), and classify (vote across nearest prototypes). All benchmarks use frozen feature extractors — the system operates on pre-extracted embeddings, not raw pixels.

**This is active research with known structural problems. We are looking for aggressive, constructive review.** Read the source code — there are four files in `src/`, all under 400 lines. If you can identify what this system actually is (or isn't), where the math breaks down, or whether any of this is genuinely new — open an issue. See [Known Issues](#known-issues) and [Open Questions](#open-questions).

## What It Does

FoldCore maintains a growing set of prototype vectors on the unit hypersphere. When input arrives:

- **Spawn**: if no existing prototype is similar enough (cosine < threshold), a new prototype is created
- **Update**: the nearest prototype moves toward the input
- **Merge**: if a new prototype is too similar to an existing one, they fuse
- **Classify**: label of the nearest prototype (or weighted vote across k-nearest)

Every input modifies the system. There is no read-only inference mode.

## Architecture

Two implementations:

### `src/foldcore_manytofew.py` — Canonical kernel (CPU, pure Python)
- Codebook layer: unit vectors in R^d, spawn/update/merge dynamics
- Matrix layer: fixed cells with eigenform dynamics (RK), coupled. Handles generation.
- Many-to-few routing: codebook vectors assign to matrix cells at spawn time
- Dependencies: `src/rk.py` (matrix utilities)

### `src/atomic_fold.py` — Hopfield-equivalent kernel (GPU, PyTorch) [DEPRECATED]
- Single codebook with per-prototype confidence weights (kappa)
- Softmax attention for both training updates and classification
- Energy-gated spawning via Hopfield energy
- **This is mathematically identical to a modern Hopfield network** (Ramsauer et al. 2020). The logsumexp energy, softmax attention, and attractor dynamics are Hopfield's math. Kept for reference, not active development.

### `src/eigenfold.py` — Matrix codebook with eigenform dynamics (CPU, pure Python) [ACTIVE]
- Codebook elements are k×k matrices (not vectors), each seeking eigenform Φ(M) = tanh(αM + βM²/k)
- Classification by **perturbation stability**: input is cross-applied with each element, most stable element (smallest perturbation) wins
- Matrix interactions are **noncommutative** (M_i·M_j ≠ M_j·M_i) — this breaks the symmetry that makes vector-based systems equivalent to Hopfield
- Fold lifecycle: spawn when no element is stable, update winner via cross-application, eigenform recovery after update
- Dependencies: `src/rk.py` (matrix utilities)

## Benchmark Results

All results use frozen feature extractors (no end-to-end training).

### Permuted-MNIST (10 sequential tasks, d=384)

| Method | Avg Accuracy | Forgetting |
|--------|-------------|------------|
| FoldCore (attractive-only, 1-NN) | 56.7% | 0.0pp |
| FoldCore (full gradient, 1-NN) | 84.1% | 11.4pp |
| Fine-tune baseline | ~52.5% | ~47pp |
| EWC | ~95.3% | ~2pp |

### Split-CIFAR-100 (20 tasks, d=512, frozen ResNet-18)

| Method | Avg Accuracy | Forgetting |
|--------|-------------|------------|
| FoldCore (full gradient, k=10 weighted) | 36.6% | 12.9pp |
| FoldCore (full gradient, 1-NN) | 33.5% | 12.6pp |
| FoldCore (attractive-only, 1-NN) | 32.3% | 12.5pp |
| EWC | ~33% | ~16pp |
| DER++ | ~51% | ~8pp |

### Notes on results
- The attractive-only update rule produces structural zero forgetting (0.0pp) because prototypes are never overwritten. The gradient update breaks this by repelling wrong-class prototypes.
- Accuracy gaps vs EWC/DER++ are due to nearest-prototype readout vs learned decision boundaries, not memory failure.
- Published baselines use 60K samples/task (MNIST) and full training pipelines. FoldCore uses 6K/task with random projection.

## Requirements

- Python 3.8+
- `src/foldcore_manytofew.py`: no dependencies beyond stdlib
- `src/atomic_fold.py`: PyTorch with CUDA
- Tests: pytest, numpy

## Running

```bash
# Run tests (skip slow benchmarks)
pytest tests/test_manytofew.py -m "not slow"

# Run all tests including benchmarks
pytest tests/test_manytofew.py
```

## License

CC BY-NC 4.0 — free for non-commercial research and educational use. See [LICENSE](LICENSE).

## Known Issues

### Vector codebook (`foldcore_manytofew.py`, `atomic_fold.py`)
- **Matrix layer is dead for classification.** The `classify()` method reads only the codebook. Removing all matrix cells produces identical results.
- **`atomic_fold.py` is a modern Hopfield network.** The softmax attention, logsumexp energy, and attractor dynamics are mathematically identical to Ramsauer et al. 2020. This was identified by external review. Any results characterize Hopfield behavior, not a new system.
- **Zero forgetting is trivial in attractive-only mode.** Append-only storage preserves old prototypes by construction.
- **Readout is the bottleneck.** 1-NN over prototypes is a weak classifier. The codebook stores well but reads poorly.

### Matrix codebook (`eigenfold.py`) — classification exhausted
- **Classification doesn't work.** 22.2% AA on P-MNIST 2-task vs 46.2% for vector cosine baseline. Perturbation-stability classification ≈ prototype matching with an expensive metric (confirmed by DeepSeek review + Step 76 head-to-head).
- **Cross-application destroys eigenform structure.** Ψ(M*, R) for true eigenform M* and input R lands in non-converging matrix space (0% convergence). Basin routing is inaccessible.
- **Collective coupling doesn't help.** Element-element coupling adds -0.2pp accuracy, +0.8pp forgetting, 27x slower (Step 77).
- **k=8 and k=16 landscapes are barren.** 0% convergence at both scales. The eigenform structure is k=4 specific with Φ(M) = tanh(αM + βM²/k).

### Eigenform composition algebra (tanh equation) — genuine mathematical finding, computationally trivial
- **31 distinct eigenforms at k=4** (α=1.2, β=0.8), frob≈2.64, in ~15 families {M*, -M*}.
- **Basins are rock-stable.** 0% crossing under small perturbation (frob=0.1). Structural zero-forgetting is real.
- **Composition is deterministic.** Ψ(M_i*, M_j*) → new eigenform. 10/10 consistency under noise (ε=0.001).
- **Negation distributes 100%.** M_a ∘ M_b = M_c implies (-M_a) ∘ M_b = -M_c. Perfect Z2 anti-symmetry.
- **Steiner triple kernel.** Three eigenform families {D,E,J} form a closed sub-algebra: any two distinct compose to the third.
- **Algebra is infinite.** A single eigenform generates 108+ distinct eigenforms through iterated composition (not closed after 5 rounds). Novel eigenforms are genuine fixed points (self-idempotent).
- **Not associative** (36.4% of triples). Non-commutative at some parameter settings (α=1.1, β=0.5: 8 non-commutative pairs, 89% convergence).
- **Computationally trivial.** All composition patterns reduce to absorbers (left/right zeros) or right projection (x∘y=y). Chains break: 0% convergence for length-4 chains. The algebra has structure but can't perform useful computation at k=4.

### Spectral eigenform substrate — scale-independent but applied failures
- **Spectral Φ(M) = M·M^T / ||M·M^T||_F · target_norm** achieves 100% convergence at k=4, 8, 16. First equation to be scale-independent.
- **Formula C composition:** Ψ(A,B) = Φ(A + B - A·B/||A·B||·target). Genuine mixing (result ≠ either input), 15/15 non-commutative pairs (cosine ≈ 0.796), deterministic, 100% convergence.
- **Pairwise properties are strong:** 15/24 permutations distinct at length-4, 12/12 reversed pairs differ, 93% of ordered pairs non-commutative with 8-element alphabet.
- **P-MNIST classification FAILS:** 15.9% compositional encoding vs 46.2% vector cosine baseline. Wrong task for the substrate (no sequential structure in permuted images).
- **Temporal order discrimination FAILS:** 55.5% vs 64.5% for order-blind baselines. Class prototypes converge to nearly identical eigenforms (cosine = 0.9861). Long-chain composition (length 7-8) collapses to same small attractor set regardless of input order.
- **Root cause:** Pairwise non-commutativity exists but doesn't accumulate constructively through chains. The quotient structure dominates — the algebra maps long sequences to a handful of fixed points that don't align with task-relevant boundaries.

## Resolved Questions

- **Perturbation-stability classification vs cosine nearest-prototype:** Vector cosine wins by 24pp (46.2% vs 22.2%). Perturbation stability ≈ prototype matching with an expensive metric.
- **Basin structure of Φ(M) = tanh(αM + βM²/k):** 31 eigenforms at k=4, ~8 orthogonal families, 0% basin crossing at frob=0.1. Only 1.2% of random matrices converge.
- **Structural zero-forgetting:** Yes, within basins. But inputs can't be routed to basins for classification.
- **Can spectral eigenform composition encode sequences?** Partially. Pairwise non-commutativity is strong (93% of pairs), and short sequences (length 2-3) produce distinct compositions. But long chains (length 7+) collapse to a small attractor set. The quotient structure maps many distinct sequences to the same eigenform. On an order-discrimination task (A-before-B vs B-before-A, length 5-10), the substrate scored 55.5% — below order-blind baselines (64.5%).

## Open Questions

1. **Is the eigenform composition algebra a known algebraic structure?** A non-associative, non-commutative, idempotent magma with Steiner triple kernel and Z2 anti-symmetry, generated by continuous dynamics on 4×4 matrices. Does this exist in the literature?
2. **Why is the tanh eigenform structure k=4 specific?** k=8 and k=16 produce 0% convergence with scaled parameters. What property of 4×4 matrices under tanh(αM + βM²/k) creates fixed points that larger matrices don't have?
3. **Can the spectral quotient structure be enriched?** Formula C's quotient maps long sequences to a handful of attractors. Is there a composition formula that preserves more information through chains — producing O(n!) distinct outputs for length-n sequences rather than collapsing?
4. **What IS the right task for eigenform composition?** The substrate has non-commutative pairwise composition, deterministic dynamics, and scale independence. These are strong algebraic properties. But classification, sequence discrimination, and order encoding have all failed. What computational task would genuinely benefit from non-commutative matrix composition?

## How to Contribute

The most valuable contribution is honest analysis. Run the code, read the math, and tell us what's wrong.

- **Identify the algebra.** The eigenform composition structure (Steiner triple kernel, Z2 anti-symmetry, infinite generation from finite seed) may be a known algebraic object. If you recognize it, cite it.
- **Explain the k=4 specificity.** Why does Φ(M) = tanh(αM + βM²/k) produce eigenforms only at k=4? Is this a spectral property of the map?
- **Propose a richer composition.** Formula C (spectral Φ + additive-multiplicative blend) gives non-commutative pairwise composition but collapses through long chains. We need composition that preserves more information through sequences.
- **Identify the right task.** Non-commutative deterministic matrix composition — what computational problem does this actually solve? Not classification, not sequence ordering.
- **Break the results.** If the algebra characterization doesn't reproduce, or if a simpler explanation exists, that's important.

Open an issue or submit a PR. Blunt feedback is preferred over polite encouragement.

## Status

Active research, 96 experiments completed. Two arcs:

**Arc 1 (fold equation):** Verified perceptual primitive. Many-to-few architecture achieves 33/33 coverage + generation. Structural zero forgetting on Permuted-MNIST (0.0pp). Matches EWC on Split-CIFAR-100 (33.5% AA). Canonical implementation frozen.

**Arc 2 (eigenform substrate):** Tanh equation exhausted at k=4 (17 experiments). Spectral eigenform achieves scale-independent convergence and non-commutative composition, but fails all applied tasks (classification, sequence discrimination, order encoding). The substrate has interesting algebraic properties but hasn't demonstrated a capability that simpler methods can't match.

**Arc 3 (readout + substrate search, Steps 97-105):** Systematic search for mechanisms that beat 1-NN readout over the same codebook. 15 constraints extracted from 105 experiments total.

Key results:
- **Top-K Class Vote:** Per-class sum of top-k cosine similarities. +5.0pp over 1-NN on P-MNIST (91.8% AA, 0.0pp forgetting), +6.1pp on CIFAR-100 (38.3% AA). Validated across both benchmarks.
- **Cosine spawning:** Simple cosine threshold replaces energy-based spawning. +30.1pp on P-MNIST (56.7% → 86.8% with 1-NN alone).
- **Best system:** Competitive learning + cosine spawning + top-k readout. 91.8% P-MNIST AA, 0.0pp forgetting. 30 lines, no backprop, 20 seconds runtime. 8,597 vectors.
- **Resonance dynamics disproved:** Energy-gradient dynamics over dense correlated codebooks converge to centroid blur, not class attractors. More dynamics = worse (Step 103, -34pp at 20 iterations).
- **Centroid floor measured:** Single d×C matrix with outer product accumulation: 30% AA. Codebook's sparse storage is load-bearing — averaging destroys task-specific geometry.

Failed readout mechanisms: differential response (attention × displacement anti-correlate), neighborhood coherence (static vector property penalizes boundary vectors), self-routing gates (sum-all aggregation drowns signal). Each failure extracted a constraint narrowing the search space.

Research continues — searching for the atomic substrate where memory, learning, inference, and perception are one operation.
