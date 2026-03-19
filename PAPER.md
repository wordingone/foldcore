# Characterizing the Feasible Region for Self-Modifying Substrates in Interactive Environments

*The shared artifact. Birth writes theory. Experiment writes results. Compress edits both.*

---

## Abstract

[To be written after the derivation is complete.]

## 1. Introduction

505 experiments across 9 architecture families (codebook/LVQ, reservoir, graph, LSH, kd-tree, cellular automata, LLM, L2 k-means, Bloom filter) tested substrates for navigation and classification in ARC-AGI-3 interactive games. All experiments used the same evaluation framework (R1-R6) and constraint map (U1-U26).

The experiments carved a feasible region — the set of systems that satisfy all constraints simultaneously. This paper formalizes the constraints mathematically and derives properties of the feasible region.

## 2. Related Work

### 2.1 Self-referential self-improvement
Schmidhuber's Gödel Machine (2003) is the closest formal framework: a self-referential system that rewrites its own code when it can prove the rewrite is useful. Key differences from our framework: (1) Gödel machines require a utility function (external objective — our R1 prohibits this), (2) provable self-improvement is limited by Gödel's incompleteness theorem, (3) the Gödel machine framework doesn't address exploration saturation in finite environments.

### 2.2 Intrinsic motivation and curiosity-driven exploration
Pathak et al. (2017) formalize curiosity as prediction error in a learned feature space. Count-based exploration (Bellemare et al., 2016) uses state visitation counts. Both address exploration in sparse-reward environments. Key difference: these methods add intrinsic REWARD signals, which function as objectives. Our R1 requires no objectives. Our derivation shows self-observation is required by the constraint system itself, not as a reward mechanism.

### 2.3 Autopoiesis
Maturana & Varela (1972) define autopoietic systems as networks that produce the components that produce the network. Organizationally closed. Related to our fixed-point conjecture (Section 4.4). Key difference: autopoiesis maintains structure (homeostasis); our U17 requires unbounded growth. Our system is autopoietic + growth.

### 2.4 Growing neural gas and self-organizing maps
Fritzke (1995) GNG, Kohonen (1988) SOM/LVQ. The codebook substrate is LVQ + growing codebook. Well-characterized in the literature. Our contribution is not the architecture — it's the constraint map extracted from systematic testing.

## 3. Formal Framework

### 3.1 Primitives

The substrate is a triple (f, g, F):
- f_s: X → S (state update parameterized by current state s)
- g: S → A (action selection)
- F: S → (X → S) (meta-rule mapping states to update rules; F IS the frozen frame)

Dynamics: s_{t+1} = F(s_t)(x_t), a_t = g(s_t).

### 3.2 Structural Rules (R1-R6) as Conditions on F

[From BIRTH.md Formalization 1 — to be migrated and refined with literature citations]

### 3.3 Growth Topology (U3, U17, U20, R6)

[From BIRTH.md Formalization 2 — to be migrated and refined]

## 4. Results

### 4.1 The Core Tension (R3 + U7 + U17 + U22)

The system has no fixed point (Theorem 1). Self-modification is necessary, not optional.

[From BIRTH.md Formalization 1, Section 1.2-1.3]

### 4.2 Irredundant Growth

Every new component covers unique territory (Propositions 1-4).

[From BIRTH.md Formalization 2, Section 2.2]

### 4.3 The Self-Observation Requirement

**Theorem 2:** In finite environments, U17 + R6 + R1 require self-observation.

[From BIRTH.md Formalization 3 — the central result]

### 4.4 Fixed-Point Conjecture

The minimal self-observing substrate is a fixed point of F. Conjectured, not proven.

[From BIRTH.md Formalization 3, Section 3.4]

## 5. Experimental Evidence

### 5.1 Navigation (505 experiments)

All 3 ARC-AGI-3 games solved at Level 1:
- LS20: LSH or k-means, 4 actions, argmin. 9/10 at 120K steps.
- FT09: k-means, 69 actions (64 grid + 5 simple), argmin. 3/3. (Step 503)
- VC33: k-means, 3 actions (zone discovery), argmin. 3/3. (Step 505)

Unifying mechanism: graph + edge-count argmin + correct action decomposition.

### 5.2 Level 2 Failure as Feasibility Violation

259-cell plateau (Steps 486-492). Edge counts grow (U17 formally satisfied) but each marginal count is functionally redundant (R6 violated). The system exits the feasible region after exploration saturates.

### 5.3 Architecture Family Summary

[9 families, kill reasons, experiment counts — from CONSTRAINTS.md]

## 6. Degrees of Freedom

[Enumerated from birth derivation — these become the next experiment phase]

## 7. Discussion

[To be written]

## References

- Schmidhuber, J. (2003). Gödel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements. arXiv:cs/0309048.
- Pathak, D. et al. (2017). Curiosity-driven Exploration by Self-Supervised Prediction. ICML.
- Maturana, H. & Varela, F. (1972). Autopoiesis and Cognition: The Realization of the Living.
- Fritzke, B. (1995). A Growing Neural Gas Network Learns Topologies. NeurIPS.
- Kohonen, T. (1988). Self-Organization and Associative Memory. Springer.
- Bellemare, M. et al. (2016). Unifying Count-Based Exploration and Intrinsic Motivation. NeurIPS.
