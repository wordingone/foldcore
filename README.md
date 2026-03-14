# FluxCore

A codebook-based learning system that processes data in a single pass with no backpropagation, no replay buffer, and no separate training/inference phases.

## What It Does

FluxCore maintains a growing set of prototype vectors on the unit hypersphere. When input arrives:

- **Spawn**: if no existing prototype is similar enough (cosine < threshold), a new prototype is created
- **Update**: the nearest prototype moves toward the input
- **Merge**: if a new prototype is too similar to an existing one, they fuse
- **Classify**: label of the nearest prototype (or weighted vote across k-nearest)

Every input modifies the system. There is no read-only inference mode.

## Architecture

Two implementations:

### `src/fluxcore_manytofew.py` — Canonical kernel (CPU, pure Python)
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
| FluxCore (attractive-only, 1-NN) | 56.7% | 0.0pp |
| FluxCore (full gradient, 1-NN) | 84.1% | 11.4pp |
| Fine-tune baseline | ~52.5% | ~47pp |
| EWC | ~95.3% | ~2pp |

### Split-CIFAR-100 (20 tasks, d=512, frozen ResNet-18)

| Method | Avg Accuracy | Forgetting |
|--------|-------------|------------|
| FluxCore (full gradient, k=10 weighted) | 36.6% | 12.9pp |
| FluxCore (full gradient, 1-NN) | 33.5% | 12.6pp |
| FluxCore (attractive-only, 1-NN) | 32.3% | 12.5pp |
| EWC | ~33% | ~16pp |
| DER++ | ~51% | ~8pp |

### Notes on results
- The attractive-only update rule produces structural zero forgetting (0.0pp) because prototypes are never overwritten. The gradient update breaks this by repelling wrong-class prototypes.
- Accuracy gaps vs EWC/DER++ are due to nearest-prototype readout vs learned decision boundaries, not memory failure.
- Published baselines use 60K samples/task (MNIST) and full training pipelines. FluxCore uses 6K/task with random projection.

## Requirements

- Python 3.8+
- `src/fluxcore_manytofew.py`: no dependencies beyond stdlib
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

## Status

Active research. The system works for continual learning with structural zero forgetting (attractive-only mode) or higher accuracy with nonzero forgetting (gradient mode). Accuracy lags gradient-based methods. The architecture is evolving.
