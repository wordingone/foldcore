# The Search

**Can a system improve itself by criteria it generates?**

This repository documents a systematic search for a substrate — a minimal computational structure — that satisfies six simultaneous rules for recursive self-improvement. 612+ experiments across 12 architecture families. No solution found yet. The constraint map from those failures, a self-modification hierarchy (ℓ₀ through ℓ_F), three theorems, and ten propositions characterizing the feasible region are the main contributions.

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

## What 612+ Experiments Found

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

**16 levels across 3 ARC-AGI-3 games (Steps 572j-610).** LS20 L1-L3=5/5 (mgu pipeline). FT09 all 6 levels, 75 clicks deterministic (Step 608b). VC33 all 7 levels, 176 clicks, analytical BFS (Step 610). All via source analysis — prescribed pipelines per game.

**R1 is cheap, R3 is the bottleneck (Proposition 9).** Competition data: CNN+RL (violates R1) achieves 18 levels. Graph exploration (satisfies R1, violates R3) achieves 17 levels. Source analysis IS the R3 violation — the 16 levels are R3's SPECIFICATION (16 concrete test cases for what an R3-compliant substrate must discover autonomously), not evidence of R3 progress.

**Proposition 6 FALSIFIED (Step 589).** Recode(K=16) 18/20 = LSH(K=16) 18/20. The advantage was K, not self-modification. The hierarchy is descriptively useful but not operationally predictive.

**Feature-Ground Truth Coupling (Proposition 10).** The gap from ℓ₁ to ℓ_F is recording vs predicting ground truth events. Pixel statistics navigate to the WRONG cells (Step 577d: 0/5 L1). Death penalties improve navigation retrospectively (Step 581d: 4/5). Prospective prediction requires features that correlate with ground truth — achievable through population-level selection (GRN architecture), not individual self-reference.

**System Boundary Theorem (Theorem 3).** R3 (full self-modification) and R5 (fixed ground truth) are simultaneously satisfiable iff ground truth is strictly environmental. The feasible region is not provably empty.

**Negative transfer is universal (Steps 506, 515).** Frozen centroids from one domain break navigation in another. LSH avoids it — random projections are domain-agnostic.

### The constraint map

Constraints extracted from experimental failure across 12 families. See [CONSTRAINTS.md](CONSTRAINTS.md) for the full map with cross-family validation status.

---

## Current Direction

1. **R3 specification via depth push** — 16 levels solved via source analysis (3 LS20 + 6 FT09 + 7 VC33). Each level is a concrete test case for what an R3-compliant substrate must discover autonomously. LS20 L4+ blocked at L3 transition (bootstrap fix in progress, Step 612).
2. **Population-level R3** — GRN substrate (Step 607) KILLED (encodings not diverse enough for selection). Framework is sound (Price equation, quorum sensing); engineering challenge is generating behaviorally diverse encoding candidates.
3. **The paper** — [PAPER.md](PAPER.md). Three theorems, Propositions 1-10, self-modification hierarchy (ℓ₀–ℓ_F), biological connections (GRN, quorum sensing, stigmergy, Physarum). 612+ experiments.

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
  experiments/           -- 590+ experiment scripts
  experiments/foldcore-steps/  -- Archived Phase 1 scripts (Steps 1-80). Hardcoded paths to local data. Not portable — included for historical record only.
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
