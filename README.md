# The Search

**Searching for a point inside six walls.**

This repository documents an ongoing search for a substrate that satisfies all six rules of recursive self-improvement simultaneously. Phase 1 mapped the feasible region through systematic elimination. Phase 2 (current) tests whether improvements to classification transfer to navigation — and diagnoses why they don't.

The search follows a [constitution](CONSTITUTION.md): six simultaneous rules (R1-R6) that define the feasible region, plus 24 universal constraints extracted from experimental failure.

---

## What Phase 1 Found (Honest)

Experiments across four substrate architectures (Living Seed, ANIMA, FoldCore, TopK-Fold) produced:

**The constraint map.** 26 universal constraints (U1-U26, 7 provisional), 9 intent constraints (I1-I9), 21 substrate-specific constraints (S1-S21). See [CONSTRAINTS.md](CONSTRAINTS.md). Each constraint is extracted from experimental failure. Together they define what the next substrate must be.

**The constitution.** Six rules (R1-R6) that must hold simultaneously. Not stages to climb -- walls of a feasible region. A substrate either satisfies all six or it doesn't. See [CONSTITUTION.md](CONSTITUTION.md).

**An honest assessment of process().** The 22-line `process()` function is LVQ (Kohonen 1988) + Growing Neural Gas (Fritzke 1995). It satisfies R1, R2 (partial), R5, R6. It fails R3 (cosine, top-K, attract, spawn are hardcoded Python the system can't modify) and R4 (no self-testing mechanism). The "stage progression" was self-assessed and circularly validated. The ARC-AGI-3 results are biased random walk, not intelligence.

**What process() does achieve:**
- 94.48% on Permuted-MNIST with 0.00pp forgetting — **supervised** (external labels required; self-labels = 9.8%, Step 432)
- Level 1 on 3/3 ARC-AGI-3 preview games — **R1-compliant** (actions are self-generated)
- Cross-domain survival: P-MNIST → LS20 → P-MNIST with 0.0pp contamination (Step 433)
- ~2x faster than pure random walk for navigation (60% at 26K vs 40% at 50K, Step 434)

**The label dependency (Step 432):** Classification is entirely supervised. Without external labels, accuracy drops to chance (9.8%). The self-labeling mechanism compounds errors through softmax voting. Navigation is unaffected because actions don't need to be "correct."

**The navigation wall (Step 428):** Action-score convergence kills directed exploration by ~5K steps. No scoring modification fixes this (U25, Steps 426-430). Navigation speed is determined by encoding quality, not scoring formula (FT09: 82 steps vs LS20: 26K, 300x from encoding alone).

**Cross-domain survival (Step 433):** The codebook naturally partitions by domain geometry. P-MNIST knowledge survives LS20 exposure with zero interference. One-directional: the existing codebook suppresses new-domain exploration.

---

## The Six Rules

All must hold simultaneously. See [CONSTITUTION.md](CONSTITUTION.md) for full text.

| Rule | Requirement | Phase 1 Status |
|------|------------|----------------|
| R1 | Computes without external objectives | Dynamics pass; classification requires external labels (Step 432) |
| R2 | Adaptation arises from computation | Partial (attract depends on class vote) |
| R3 | Every modifiable aspect IS modified by the system | **FAILS** (operations hardcoded) |
| R4 | Modification tested against prior state | **FAILS** (no self-testing) |
| R5 | One fixed ground truth test | Passes |
| R6 | No part deletable without losing all capability | Passes |

R3 is the binding constraint. The operations (cosine similarity, top-K voting, attract/spawn rule) are Python code the system cannot see, inspect, or modify.

---

## Phase 2: New Substrates (Current)

Phase 2 builds systems that satisfy R1-R6 from scratch. Not process() with additions. New data structures, new matching operations, new self-modification mechanisms.

### Candidates tested

| Substrate | Data Structure | R3 Audit (U) | Status | Key Finding |
|-----------|---------------|-------------|--------|-------------|
| SelfRef | Codebook (vectors) | 10 U | Active | Best discrimination (94%), 0 levels. Cosine frozen (R3). |
| TapeMachine | Integer tape | 10 U | **KILLED** | Hash violates U20 (no local continuity). 35% disc. |
| ExprSubstrate | Expression tree | 8 U | **KILLED** | U21: scoring rewards noise diversity, not signal. |
| TemporalPrediction | Prediction matrix | 4 U | **KILLED** | U22: LMS converges → W freezes → exploration dies. |

### Key Phase 2 discoveries

**U20 (local continuity):** The substrate must map similar inputs to similar outputs. Cosine provides this for free. Hash and random splits violate it.

**U22 (convergence kills exploration):** Fixed-size prediction state converges when pred_err → 0. Growth (codebook spawning) prevents convergence. This is why LVQ navigates but temporal prediction doesn't.

**U24 (argmin ≠ argmax):** Exploration (argmin = least familiar) and classification (argmax = most familiar) are opposite operations. No single action mechanism serves both. Softmax voting improved classification +3.3pp but made navigation *worse* (Step 426).

**Step 428 (the navigation wall):** Action-score convergence is the mechanism. Top-K sum scores converge as codebook grows → all actions equally familiar → random walk from ~5K steps onward. The substrate is a directed explorer for ~5K steps, then pure noise.

### Active direction

The navigation wall (Step 428) specifies the next attack: prevent action-score convergence. Options under test: score normalization by class count, entry decay, per-entry quality metrics.

---

## Running It

```bash
# Phase 1: the LVQ baseline
pip install torch torchvision numpy
python experiments/run_step250_complete_substrate.py    # Complete demo (~30s)
python experiments/foldcore-steps/run_step99_topk_vote.py  # P-MNIST benchmark

# Phase 2: new substrates
python substrates/selfref/test_selfref.py              # Self-referential codebook
python substrates/tape/test_tape.py                     # Integer tape machine
python substrates/expr/test_expr.py                     # Expression tree
```

---

## Repository Structure

```
the-search/
  CONSTITUTION.md        -- 5 principles + 6 rules (R1-R6). The feasible region.
  CONSTRAINTS.md         -- U1-U20, I1-I9, S1-S21. The experimental record.
  RESEARCH_STATE.md      -- Full experiment log and honest assessment.
  INDEX.md               -- File-by-file index of all experiments.

  substrates/            -- All substrate implementations
    living-seed/         -- Phase 1: Living Seed (Sessions 1-17) [CLOSED]
    anima/               -- Phase 1: ANIMA (Sessions 18-23) [CLOSED]
    foldcore/            -- Phase 1: Codebook baseline
    eigenfold/           -- Phase 1: Matrix codebook [CLOSED]
    topk-fold/           -- Phase 1: The 91.8% system (LVQ + top-K)
    selfref/             -- Phase 2: Self-referential codebook chain
    tape/                -- Phase 2: Integer tape machine
    expr/                -- Phase 2: Self-modifying expression tree

  experiments/           -- Experiment scripts (Phase 1 + Phase 2)

  knowledge/             -- Structured knowledge base (78 entries, 70+ constraints)
  paper/                 -- Paper compiler
  research/              -- Framework documentation
  tempest/               -- Tempest substrate (Rust, historical)
  tests/                 -- Unit tests
```

## The Constitution

The [constitution](CONSTITUTION.md) defines five architecture-independent principles and six rules:

1. **Computation must exist without external objectives** (R1)
2. **Adaptation must arise from computation, not beside it** (R2)
3. **Each modification must be tested against what came before** (R4)
4. **The frozen frame must be minimal** (R3 + R6)
5. **There must be one ground truth the system cannot modify** (R5)

The rules are simultaneous constraints, not sequential stages. The Phase 1 "stage progression" (Stages 1-8) was superseded by the R1-R6 feasibility framing after external review exposed systematic inflation.

## Constraints from Experiments

26 universal constraints (7 provisional) define what ANY substrate must satisfy:

- **U1-U4:** Structural (read+write one operation, one data structure, zero forgetting, minimal)
- **U5-U8:** Selection (sparse over global, Lipschitz boundary, no iteration, hard over soft)
- **U9-U13:** Dynamics (curriculum must match solution, dense memory kills, discrimination ≠ navigation, structured noise, additions hurt)
- **U14-U16:** Self-reference (substrate IS its search, robust to perturbation, encode differences)
- **U17-U20:** Capacity (fixed memory exhausts, shared channels contaminate, dynamics ≠ features, local continuity required)
- **U21-U26:** Phase 2 (diversity scoring degenerate, convergence kills exploration, distributed updates destabilize, argmin ≠ argmax, score/bias coupling, self-label error compounding)

Each constraint is a closed door. The pattern of elimination IS the search.

---

## What Failed and Why It Matters

- **The Living Seed** (Sessions 1-17): architecture ceiling at 6/8 frozen elements. The equation is hardcoded Python.
- **ANIMA** (Sessions 18-23): all parameters either non-binding or non-adaptable. Stage 3 ceiling.
- **Eigenform arc** (17 experiments): algebraically interesting, zero applied capability.
- **The decomposition arc** (Steps 235-278): human-designed algorithms executed by k-NN. Manual compilation, not discovery.
- **35 encoding experiments** (Steps 377-412): all failed. Cosine saturation at high-D is the Goldilocks zone, not a bug to fix.
- **Step 417** (REPEL/FREEZE flags): LVQ with a state machine. The frozen frame grew. Direction killed by founder.
- **Phase 2 substrates** (Steps 417-425): TapeMachine, ExprSubstrate, TemporalPrediction — all killed. SelfRef active.
- **60+ navigation experiments** (Steps 354-428): every action mechanism modification failed. The wall is action-score convergence (Step 428).

Each failure extracted constraints. The constraint map IS the specification for the next substrate.

---

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
