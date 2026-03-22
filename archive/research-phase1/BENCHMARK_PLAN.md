# FluxCore — Major Benchmark Exit Gate
*From [researcher], 2026-03-13. Reviewed against DIRECTOR_INTENT.md.*

---

## Why This Matters

FluxCore's many-to-few architecture (v17) achieves 33/33 coverage + generation on 140K FLOPs/step. Phase 7b benchmarks validate internal capabilities. But all validation is internal — CSI corpus, synthetic drift, synthetic chaotic series. No external comparison exists.

[director]'s intent: "the atomic substrate that collapses the entire fractured stack into one indivisible thing." To prove this, FluxCore must produce measurable results on a benchmark where the current SOTA (transformers + gradient descent) competes. Not to overfit — to prove the foundation is real.

---

## Target: Continual Learning

### Why continual learning

1. **Transformer weakness**: catastrophic forgetting is the known failure mode. Models trained on task B forget task A. The entire field of continual learning exists because gradient descent overwrites learned representations.

2. **FluxCore structural advantage**: the fold's codebook NEVER overwrites. New inputs spawn new vectors. Old vectors persist. This is not an anti-forgetting mechanism bolted on — it's inherent to the fold's dynamics.

3. **Major published area**: NeurIPS, ICML papers every year. Standardized benchmarks. Published baselines. External validation is meaningful.

4. **Tests the right thing**: online learning, memory retention, adaptation to new distributions — exactly what the fold equation does.

### What it does NOT test

- Token prediction (language modeling) — architectural mismatch, not a capability FluxCore claims
- Raw per-task accuracy vs supervised learning — FluxCore has no loss function or backprop
- Feature extraction quality — FluxCore claims memory organization, not representation learning

---

## Architecture Extension

The ManyToFewKernel needs a minimal output mechanism. Three additions:

### 1. Labeled codebook

When a codebook vector spawns, assign it the label of the input that caused the spawn. The label persists with the vector through updates and merges.

```python
self.cb_labels = []  # parallel to self.codebook

# At spawn:
self.cb_labels.append(label)

# At merge (fuse new into existing):
# Keep existing vector's label (it has more history)
del self.cb_labels[new_idx]
```

### 2. Nearest-prototype classification

At inference, find the nearest codebook vector and return its label.

```python
def classify(self, r):
    """Return label of nearest codebook vector."""
    if not self.codebook:
        return None
    sims = [_vec_cosine(v, r) for v in self.codebook]
    winner = max(range(len(self.codebook)), key=lambda i: sims[i])
    return self.cb_labels[winner]
```

This is the fold's natural readout. No gradient descent. No learned parameters. The codebook IS the classifier — each vector is a labeled prototype.

### 3. Embedding pipeline

Raw images (e.g., 32x32x3 CIFAR) need to become R^d vectors. Options:

| Method | Pros | Cons | Honest? |
|--------|------|------|---------|
| Random projection | Pure, no pretrained model | Loses structure, hard mode | Yes — but unfair comparison |
| Frozen pretrained encoder (ResNet) | Standard in CL papers, fair comparison | Stands on transformer-era features | Yes — CL papers all do this |
| Simple learned encoder | Middle ground | Requires training, adds complexity | Partially |

**Decision: frozen pretrained encoder (same as baselines use).** Continual learning papers compare LEARNING ALGORITHMS, not feature extractors. All baselines use the same frozen encoder. FluxCore's contribution is the learning algorithm (fold memory vs gradient descent). Using the same encoder ensures a fair comparison.

If FluxCore succeeds with a frozen encoder, a FOLLOW-UP experiment with random projection would test whether the fold's memory organization works even without pretrained features. But that's a separate question.

---

## Benchmark Protocol

### Phase 1: Permuted-MNIST (proof of mechanism)

**Dataset**: MNIST digits (28x28 grayscale), 10 classes.
**Protocol**: 10 sequential tasks. Each task applies a fixed random permutation to all pixels, creating a new distribution. Train on each task sequentially (no replay of old tasks).
**Metric**: Average accuracy across all 10 tasks after training on the last task. Per-task accuracy matrix (accuracy on task i after training on task j, for all j >= i).
**Embedding**: Flatten to R^784. Random projection to R^384 (FluxCore's d).
**Why first**: Simple, fast, well-understood. Published baselines are abundant. Proves the mechanism before scaling to harder data.

### Phase 2: Split-CIFAR-100 (the real benchmark)

**Dataset**: CIFAR-100 (32x32 color, 100 classes).
**Protocol**: 20 sequential tasks, 5 classes each. Train on each task sequentially. No replay.
**Metric**: Average accuracy, forgetting measure (accuracy drop on task i after training on later tasks), learning curve.
**Embedding**: Frozen ResNet-18 features → R^512 (standard in CL papers). Or random projection to R^384.
**Why this**: The standard continual learning benchmark. Published baselines for every major method.

### Published baselines to compare against

| Method | Type | Key idea |
|--------|------|----------|
| EWC | Regularization | Penalize changes to important weights |
| SI | Regularization | Online importance estimation |
| LwF | Knowledge distillation | Soft targets from old model |
| PackNet | Architecture | Iterative pruning per task |
| Progressive Nets | Architecture | New columns per task |
| ER (Experience Replay) | Replay | Store subset of old examples |
| A-GEM | Replay | Gradient constraints from old examples |
| DER++ | Replay | Dark knowledge replay |

All use gradient descent + backprop. FluxCore uses fold dynamics. Different physics entirely.

---

## Expected Results (Honest Assessment)

### Where FluxCore should excel

- **Forgetting**: near-zero. Old codebook vectors persist. New tasks spawn new vectors in unoccupied regions. The fold's additive update (lr=0.015) drifts vectors slowly — old prototypes remain stable.
- **Memory efficiency**: codebook vectors are d-dimensional unit vectors. 359 vectors * 384 dims * 4 bytes = ~550 KB total. Compared to storing network weights + importance matrices (EWC) or replay buffers (ER).
- **No replay needed**: FluxCore doesn't store old examples. The codebook IS the memory. This is structurally superior to replay methods.
- **Compute**: no backprop, no gradient computation. Forward pass only (codebook lookup + matrix step).

### Where FluxCore may trail

- **Per-task accuracy**: gradient-optimized models fit decision boundaries precisely. Nearest-prototype classification is simpler — it assigns labels based on distance, not learned decision surfaces. Expect lower peak accuracy on individual tasks.
- **Fine-grained discrimination**: if two classes are very similar in embedding space, the codebook may not spawn enough vectors to separate them. The spawn threshold (0.5) is global — it doesn't adapt to per-class difficulty.
- **No task boundaries**: some CL methods exploit known task boundaries (when the task switches). FluxCore doesn't need task boundaries (it detects novelty via spawning), but this means it can't exploit them either.

### The honest framing

"FluxCore achieves competitive or superior retention with no replay, no regularization, and no gradient descent. Per-task accuracy is [X]% vs [Y]% for gradient-based methods. The fold's memory dynamics solve catastrophic forgetting as an emergent property of the architecture, not as an explicit anti-forgetting mechanism."

If per-task accuracy is too low to be meaningful, the failure classification is: **readout mechanism limitation** (nearest-prototype is too simple) not **memory limitation** (the codebook IS learning, the readout just doesn't extract it well). This distinction matters for knowing what to fix.

---

## Implementation Steps

| Step | Task | Inputs | Output |
|------|------|--------|--------|
| 64 | Labeled codebook extension | fluxcore_manytofew.py | Updated kernel with cb_labels + classify() |
| 65 | Permuted-MNIST proof | MNIST dataset, random projection | Per-task accuracy matrix, comparison to baselines |
| 66 | Split-CIFAR-100 | CIFAR-100, frozen ResNet-18 | Full CL benchmark results, comparison table |
| 67 | Analysis + FRAMEWORK.md | Steps 65-66 results | Honest comparison, failure classification |

Steps 65-66 are the exit gate. Step 67 documents the results in FRAMEWORK.md with honest separation of fundamental vs readout limitations.

---

## Hard Rules

- FRAMEWORK.md governs all experimental claims.
- One variable per experiment.
- Document what happens, not what you hope happens.
- If FluxCore fails a benchmark, classify the failure honestly (fundamental limitation vs readout limitation vs architectural mismatch).
- Do not add complexity to FluxCore to "win" a benchmark. The fold equation is frozen. Only the thin readout layer is new.
- [director]'s standard: "get it closer to its true self." If a benchmark result is technically true but misleading, it fails.

---

*This plan targets the exit gate [director] specified: external validation on a major benchmark where transformers compete. The fold's memory dynamics should solve catastrophic forgetting inherently. If they don't, we learn why.*

*Questions -> [researcher]. FRAMEWORK.md governs.*
