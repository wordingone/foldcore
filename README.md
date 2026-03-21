# The Search

**Can a system improve itself by criteria it generates?**

This repository documents a systematic search for a substrate — a minimal computational structure — that satisfies six simultaneous rules for recursive self-improvement. 588+ experiments across 12 architecture families. No solution found yet. The constraint map from those failures, a self-modification hierarchy (ℓ₀ through ℓ_F), and three theorems characterizing the feasible region are the main contributions.

---

## The Problem

Six rules ([CONSTITUTION.md](CONSTITUTION.md)) define the feasible region. A substrate either satisfies all six or it doesn't.

| Rule | Requirement |
|------|------------|
| R1 | Computes without external objectives |
| R2 | Adaptation arises from computation, not beside it |
| R3 | Every modifiable aspect IS modified by the system |
| R4 | Each modification tested against prior state |
| R5 | One fixed ground truth test the system cannot modify |
| R6 | No part deletable without losing all capability |

R3 is the binding constraint. Every substrate tested so far has hardcoded operations the system cannot see or modify.

---

## What 588 Experiments Found

### Architecture families

| Family | Experiments | Best Result | Status |
|--------|------------|-------------|--------|
| Codebook (LVQ) | ~435 | 94.48% P-MNIST, chain 3/3 ARC games | Mapped — banned |
| LSH graph | ~80 | LS20 L1-L3 (5/5), Recode 5/5 (ℓ_π) | Active — primary family |
| L2 k-means graph | ~25 | LS20 9/10 at 120K | Active |
| Reservoir (ESN) | ~20 | Memory contributes nothing | Killed |
| Hebbian | 3 | 5/5 LS20, chain pass | Active |
| Recode (self-refining LSH) | 3 | 5/5 LS20 (ℓ_π, expands reachable set) | Active |
| Graph (cosine) | ~8 | LS20 L1 at 25738 | Superseded |
| SplitTree | 5 | 3/3 LS20 (deterministic) | Chain-incompatible |
| Absorb | 3 | 1/3 (noisy TV dominant) | Killed |
| Connected-component | 1 | 23 states, too slow | Killed |
| Bloom filter | 2 | 1/10 (random walk luck) | Killed |
| CA | 3 | Degenerate mapping | Killed |

### The chain benchmark

The real test is not single-benchmark performance. It is the **chain**: heterogeneous benchmarks run in sequence with one continuous state, no reset.

```
CIFAR-100 → Atari (ARC-AGI-3) → CIFAR-100
```

| Finding | Step | Result |
|---------|------|--------|
| Frozen centroids → negative transfer | 506, 515 | Universal across codebook AND k-means families |
| Dynamic growth → domain separation | 507-508 | Chain 3/3 ARC games, zero CIFAR forgetting |
| LSH chain via action-scope isolation | 516 | WIN@1116 — different mechanism than codebook chain |
| Threshold tension | 509-513 | CIFAR needs threshold≥3.0, ARC needs ≤0.5. Incompatible. |
| K-means cross-game transfer | 522 | Degenerate (centroid collapse). Attract update load-bearing. |

### Key findings

**All 3 ARC-AGI-3 games solved through Level 3 (Steps 572j-576).** LS20 L3=5/5, FT09 L1=5/5, VC33 L1=5/5. Mechanism: graph + edge-count argmin + mode map + isolated connected-component detection. LS20: 4 directional actions. FT09: 69 actions (64 click grid + 5 simple). VC33: 64 click actions.

**Self-modification hierarchy maps speed vs reachability (Proposition 6).** ℓ₁ (death penalty) accelerates exploration but doesn't expand the graph: 13/20 = 13/20 at 50K (Step 584, p=0.63). ℓ_π (Recode, encoding refinement) expands the graph from 440→1267 cells and improves 3/3→5/5 (Step 542). Speed vs reachability corresponds exactly to ℓ₁ vs ℓ_π.

**System Boundary Theorem (Theorem 3).** R3 (full self-modification) and R5 (fixed ground truth) are simultaneously satisfiable iff ground truth is strictly environmental. The feasible region is not provably empty.

**Negative transfer is universal (Steps 506, 515).** Frozen centroids from one domain break navigation in another. LSH avoids it — random projections are domain-agnostic.

### The constraint map

Constraints extracted from experimental failure across 12 families. See [CONSTRAINTS.md](CONSTRAINTS.md) for the full map with cross-family validation status.

---

## Current Direction

1. **ℓ_π self-modification** — Recode (Step 542, 5/5) is the strongest non-codebook result. Encoding self-modification expands the reachable set. Next: head-to-head Recode vs LSH at 20 seeds (Step 589).
2. **Statistical rigor** — 20-seed validation of all claims. Step 584 showed 5-seed results can be noise. Fisher exact tests on all comparisons.
3. **The paper** — [PAPER.md](PAPER.md) compiles to LaTeX. Three theorems, self-modification hierarchy (ℓ₀–ℓ_F), Propositions 4/6. Addressing NeurIPS reviewer feedback.

---

## Running It

```bash
pip install torch torchvision numpy

# Phase 1: the LVQ baseline
python experiments/run_step250_complete_substrate.py       # Complete demo (~30s)

# Chain experiments (Phase 2)
python experiments/run_step508_full_chain.py                # Full chain CIFAR→ARC→CIFAR
python experiments/run_step474_kmeans_l2.py                 # K-means graph on LS20
```

---

## Repository Structure

```
the-search/
  CONSTITUTION.md        -- 5 principles + 6 rules (R1-R6)
  CONSTRAINTS.md         -- Constraint map with cross-family validation
  RESEARCH_STATE.md      -- Full experiment log (Steps 1-588+)
  PAPER.md               -- Publication draft (compiles to LaTeX)
  build-paper.py         -- LaTeX compilation pipeline

  substrates/            -- Substrate implementations (Phase 1)
  experiments/           -- 588+ experiment scripts
  research/              -- Research methodology and frameworks
  archive/               -- Archived Phase 1 infrastructure
  audits/                -- External audit reports
  tests/                 -- Unit tests
```

## Requirements

- Python 3.8+
- `torch`, `torchvision`, `numpy`
- Tests: `pytest`

## License

CC BY-NC 4.0. See [LICENSE](LICENSE).

## Contributing

Run the code. Read [CONSTRAINTS.md](CONSTRAINTS.md). Tell us what's wrong. Open an issue or PR.

---

*The constraints define the region. The substrate is inside it or it doesn't exist.*
