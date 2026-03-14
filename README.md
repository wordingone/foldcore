# FoldCore

A codebook-based learning system that processes data in a single pass with no backpropagation, no replay buffer, and no separate training/inference phases.

**This is active research with known structural problems. We are looking for aggressive, constructive review.** If you can identify what this system actually is (or isn't), where the math breaks down, or whether any of this is genuinely new — we want to hear it. See [Known Issues](#known-issues) and [Open Questions](#open-questions).

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

### `src/atomic_fold.py` — Experimental unified kernel (GPU, PyTorch)
- Single codebook with per-prototype confidence weights (kappa)
- Softmax attention for both training updates and classification
- Energy-gated spawning via Hopfield energy
- No matrix layer, no routing

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

- **Matrix layer is dead for classification.** In `foldcore_manytofew.py`, the `classify()` method reads only the codebook. The matrix layer (RK cells, coupling, eigenform dynamics) does not contribute to classification output. Removing all 8 matrix cells produces identical classification results. The matrix layer may still be relevant for generation, but this has not been rigorously validated.
- **Training/inference mismatch.** The codebook trains with hard assignment (nearest-prototype update) but classifies with hard or soft readout. Soft readout (softmax attention) on a hard-trained codebook fails — the energy landscape is too flat because prototypes were shaped for hard matching. This is the core structural limitation.
- **Zero forgetting is trivial in attractive-only mode.** Prototypes are never overwritten or repelled, so old prototypes are preserved by construction. Any nearest-prototype method with append-only storage achieves this. The gradient update (which repels wrong-class prototypes) breaks zero forgetting.
- **Readout is the bottleneck.** The codebook stores information effectively (48K prototypes, full coverage) but reads it weakly. 1-NN over high-dimensional prototypes is a weak classifier compared to learned decision boundaries. This explains most of the accuracy gap vs SOTA.
- **Frozen feature dependence.** All reported benchmarks use frozen feature extractors (random projection for P-MNIST, frozen ResNet-18 for CIFAR-100). The system has not been tested with end-to-end learned features.
- **Merge is untested at scale.** `merge_thresh=0.95` at `d=512` produces 0 merges in practice. The self-compression property (codebook shrinks as redundancy is absorbed) has not been demonstrated.

## Open Questions

1. Is this a rediscovery of dictionary learning, sparse coding, or online vector quantization? The individual components (prototype vectors, nearest-neighbor update, cosine similarity) are well-known. What, if anything, is new in the combination?
2. Can the training rule and inference rule be made literally identical (not just similar)? The `atomic_fold.py` attempts this with softmax attention for both, but results are preliminary.
3. Does iterative self-retrieval (generation via `r_{t+1} = reconstruct(r_t)`) produce meaningful output, or does it collapse to a fixed point?
4. Is per-prototype confidence (`kappa` in `atomic_fold.py`) genuinely novel, or is it a mixture model with online EM?
5. Can the birth/scale/compression lifecycle (spawn→grow→merge) be demonstrated as a formal system property on real data?

## How to Contribute

The most valuable contribution is honest analysis. Run the code, read the math, and tell us what's wrong.

- **Identify prior art.** If this is a known technique under a different name, say so with a citation.
- **Break the benchmarks.** If the reported numbers don't reproduce, or if a simpler baseline matches them, that's important.
- **Answer the open questions.** Especially #1 (novelty) and #4 (kappa vs online EM).
- **Propose experiments.** What test would distinguish this from existing prototype-based methods?

Open an issue or submit a PR. Blunt feedback is preferred over polite encouragement.

## Status

Active research with significant unresolved problems. The architecture is evolving. Accuracy lags gradient-based continual learning methods.
