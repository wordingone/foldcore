---
title: "Characterizing the Feasible Region for Self-Modifying Substrates in Interactive Environments"
author: "Avir Research"
date: 2026-03-19
---

*The shared artifact. Birth writes theory. Experiment writes results. Compress edits both.*

## Abstract

We formalize six rules (R1-R6) for recursive self-improvement as mathematical conditions on a state-update function $f: S \times X \to S$ and derive necessary properties of any system satisfying all six simultaneously. From 505 experiments across 9 architecture families on ARC-AGI-3 interactive games, we extract 26 constraints and prove: (1) no satisfying system has a fixed point — self-modification is necessary, not optional; (2) in finite environments, the system must process its own internal state to maintain irredundant growth (the self-observation requirement); (3) the feasible region is non-empty for Level 1 navigation but currently unoccupied for the full constraint set including R3 (self-modification of operations). Whether a substrate exists inside all six walls remains open. The contribution is the walls themselves.

## 1. Introduction

505 experiments across 9 architecture families (codebook/LVQ, reservoir, graph, LSH, kd-tree, cellular automata, LLM, L2 k-means, Bloom filter) tested substrates for navigation and classification in ARC-AGI-3 interactive games. All experiments used the same evaluation framework (R1-R6) and constraint map (U1-U26).

The experiments carved a feasible region — the set of systems that satisfy all constraints simultaneously. This paper formalizes the constraints mathematically, derives necessary properties of the feasible region, and states honestly what is proven vs conjectured vs open.

**The paper is valid regardless of outcome.** If a substrate satisfying all six rules is found, the feasible region is non-empty and the characterization guided the search. If no such substrate exists, the characterization identifies which constraints are mutually exclusive — also a contribution. We report whichever is true at time of writing.

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

### 3.2 Structural Rules (R1-R3) and the Core Tension

**Prior work:** R1 (no external objectives) is the standard unsupervised/self-supervised setting. R2 (adaptation from computation) rules out external optimizers — related to Hebbian learning (Hebb, 1949) where adaptation is local and intrinsic. R3 (self-modification) is formalized by Schmidhuber (2003) as the Gödel machine's self-referential property, though his version requires a proof searcher we do not.

**Our formalization:**

- **R1:** $f_s(x)$ depends only on $s$ and $x$. No external signal $L$ enters. The substrate is a closed dynamical system over $S$ given input stream $X$.
- **R2:** All parameters $\Theta(f) \subseteq S$. The only mechanism modifying $\Theta$ is $f$ itself. No gradient $\nabla L$, no external optimizer. The map $t \mapsto s_t$ is generated entirely by iterating $f$.
- **R3:** $F: S \to (X \to S)$ is non-constant. $\exists s_1 \neq s_2$ such that $F(s_1) \neq F(s_2)$ as functions on $X$. The update rule depends on the state, not just the data.

**Core Tension (Theorem 1):** R3 + U7 (dominant amplification) + U22 (convergence kills exploration) produce convergence pressure. U17 (unbounded growth) prevents convergence. Proof: if $s_t \to s^*$, then $\phi(s^*) < \infty$, contradicting U17. Therefore the system has no fixed point. Self-modification is necessary — growth perpetually disrupts the dominant mode.

**Relationship to prior work:** The no-fixed-point result is a consequence of combining standard dynamical systems theory (spectral convergence) with the growth axiom (U17). The individual pieces are known; the combination producing perpetual non-convergence appears to be our derivation. Closest prior work: Schmidhuber's "asymptotically optimal" self-improvement, which converges — our system provably doesn't.

**Tensions:**
- T1: U7 assumes stationarity that R3 breaks. May need reformulation as "locally amplifies largest-variance component" (instantaneous, not asymptotic).
- T2: R3 is ambiguous between "actively modifies operations" and "operations responsive to state." Strong vs weak interpretation.

### 3.3 Growth Topology (U3, U17, U20, R6)

**Prior work:** U3 (zero forgetting) is the catastrophic forgetting constraint from continual learning (McCloskey & Cohen, 1989; French, 1999). U20 (local continuity) is Lipschitz continuity of the mapping — standard in topology-preserving embeddings (Kohonen, 1988; van der Maaten & Hinton, 2008 for t-SNE). R6 (no deletable parts) relates to minimal sufficient statistics (Fisher, 1922).

**Our formalization:**

- **U3:** $S_t \subseteq S_{t+1}$ (structural inclusion — components added, never removed). Values may change; skeleton only grows.
- **U17:** $\exists \phi: S \to \mathbb{R}_{\geq 0}$ monotonically non-decreasing, with $\lim_{t \to \infty} \phi(s_t) = \infty$.
- **U20:** $\pi: X \to N$ is Lipschitz: $d_N(\pi(x_1), \pi(x_2)) \leq L \cdot d_X(x_1, x_2)$.
- **R6:** For every component $c \in \text{components}(S_t)$, the restricted state $S_t \setminus \{c\}$ fails the ground truth test $G$.

**Propositions (proven in BIRTH.md, to be cleaned):**
1. U3 + U17 + R6 $\Rightarrow$ irredundant growth (every new component covers unique territory).
2. U20 + irredundancy $\Rightarrow$ well-separated nodes (Voronoi cells partition $X$ without unnecessary overlap).
3. Growth rate is coupled to observation distribution (environment gates growth).
4. U20 constrains node topology to inherit from $X$ (connected observations $\to$ connected nodes).

**Relationship to prior work:** The individual constraints are known (catastrophic forgetting, Lipschitz continuity, minimal sufficiency). The combination producing "irredundant growth" — where every new component is both unique and necessary — appears to be our synthesis.

**Tension T3:** R3 (self-modifying metric) + U20 (continuous mapping) = metric can REFINE (increase resolution) but cannot REARRANGE (swap what's near/far). This constrains the space of allowed self-modifications.

**Tension T4:** Finite observation space + irredundant growth = node count is bounded. U17 must be satisfied by something other than nodes (edges, in practice). But edge-count growth after node saturation violates R6 (marginal counts are redundant). This leads to Theorem 2 (Section 4.3).

### 3.4 Action Selection Constraints (U1, U11, U24)

#### U1: No separate learning and inference modes

**Prior work:** Online continual learning (OCL) formalizes exactly this constraint. Aljundi et al. (CVPR 2019, "Task-Free Continual Learning") define systems that learn from a single-pass data stream with no task boundaries. The broader OCL literature (survey: arXiv:2501.04897) emphasizes "real-time adaptation under stringent constraints on data usage." Our U1 is equivalent to the OCL setting.

**Our formalization:** $F$ is time-invariant. The same meta-rule $F: S \to (X \to S)$ applies at every timestep. There is no mode variable $m \in \{train, infer\}$ that changes $F$'s behavior. Formally: the system's dynamics are a single autonomous dynamical system, not a switched system.

**Relationship to prior work:** Equivalent to OCL's single-pass constraint. Not novel — properly attributed.

**Implications:** Combined with R3 (self-modification), U1 says: the system modifies itself using the SAME function it uses to process input. There is no separate training phase where the system is allowed to self-modify more aggressively. This rules out any architecture with explicit "learning rate schedules" or "warmup phases" unless these are derived from the state itself (which would make them R3-compliant).

#### U24: Exploration and exploitation are opposite operations

**Prior work:** The exploration-exploitation tradeoff is foundational in RL (Sutton & Barto, 2018). The formal impossibility of a single mechanism optimizing both simultaneously is well-established in multi-armed bandit theory (Auer et al., 2002; Lai & Robbins, 1985). Recent work (arXiv:2508.01287, 2025) suggests exploration can emerge from pure exploitation under specific conditions (repeating tasks, long horizons).

**Our formalization:** Let $g_{explore}: S \to A$ maximize coverage (minimize revisitation) and $g_{exploit}: S \to A$ maximize classification accuracy. Then $g_{explore} \neq g_{exploit}$ in general.

Specifically: $g_{explore} = \text{argmin}_a \sum_n E(c, a, n)$ (least-tried action) and $g_{exploit} = \text{argmax}_a \text{score}(s, a)$ (highest-confidence action). These select opposite actions when the least-explored action is also the least-confident.

**Relationship to prior work:** This is the standard RL tradeoff. Not novel. Our contribution is empirical confirmation across 505 experiments that no single $g$ produces both good navigation and good classification (Steps 418, 432, 444b).

#### U11: Discrimination and navigation require incompatible action selection

**Prior work:** No direct formal precedent found. The RL literature treats navigation and classification as different TASKS but doesn't formalize them as requiring incompatible mechanisms within the same system. The closest work is multi-objective RL, where Pareto-optimal policies for conflicting objectives are studied (Roijers et al., 2013).

**Our formalization:** Navigation requires $g_{nav}(s) = \text{argmin}_a \text{count}(s, a)$ — the action that maximizes coverage. Classification requires $g_{class}(s) = \text{argmax}_a \text{score}(s, a, \text{label})$ — the action (label assignment) that maximizes match confidence. These are not just "different" — they are NEGATIONS of each other (argmin vs argmax over the same scoring function applied to the same state).

**Relationship to prior work:** The specific finding — that argmin produces 0% classification (Step 418g) and argmax produces 0% navigation — appears to be our empirical contribution, not previously formalized as a constraint. However, the underlying principle (coverage-maximizing and accuracy-maximizing objectives conflict) is a special case of multi-objective optimization theory.

**Implications:** A system satisfying R1-R6 must handle BOTH tasks (navigation and classification). Since they require opposite $g$, the system needs either: (a) a mechanism to SWITCH between $g_{nav}$ and $g_{class}$ based on context — but U1 forbids mode switching; or (b) a single $g$ that somehow serves both objectives simultaneously — but U24 says this is impossible. This creates a genuine tension.

**Degree of freedom 11:** How does the system resolve the U1 + U11 + U24 tension? One possibility: $g$ depends on $s$ (which it must, by R3), and the STATE determines whether the system's behavior is more exploratory or more exploitative. This is not mode switching (the function $g$ is the same) — it's state-dependent behavior. The system explores when its state indicates uncertainty, and exploits when its state indicates confidence. This is known in the RL literature as Bayesian exploration (Ghavamzadeh et al., 2015).

### 3.5 Remaining Structural Rules (R4-R6) and Validated Constraints (U7, U16, U22)

#### R4: Modification tested against prior state

**Prior work:** Regression testing in software engineering (Rothermel & Harrold, 1996). In RL, policy improvement is tested against the previous policy (Kakade & Langford, 2002, conservative policy iteration). In self-adaptive systems, runtime testing validates adaptations against pre-adaptation behavior (Fredericks et al., 2018). The formal requirement that a self-modifying system must self-evaluate is implicit in Schmidhuber's Gödel machine (the proof must show the rewrite improves expected utility).

**Our formalization:** After each modification $s_{t+1} = f_{s_t}(x_t)$, there exists an evaluation $V: S \times S \times X^* \to \{better, worse, same\}$ applied by $f$ itself (not externally) to novel inputs $X^*$. $V$ is part of $F$ (frozen).

**Relationship to prior work:** Equivalent to conservative policy iteration's requirement. The novelty, if any, is requiring $V$ to be part of $f$'s dynamics (R2 compliance) rather than an external evaluator. This is the same distinction we make for R1.

#### R5: One fixed ground truth

**Prior work:** In formal verification, the specification is the fixed point against which the system is tested. In Schmidhuber's Gödel machine, the utility function is the fixed external criterion. In evolution, fitness is determined by the environment (fixed, external). The idea that a self-modifying system needs at least one invariant is well-established — without it, the system can trivially "improve" by redefining improvement.

**Our formalization:** $\exists!$ $G: S \to \{0, 1\}$ (ground truth test) such that $G \notin S$ — the system cannot modify $G$. $G$ is part of $F$.

**Relationship to prior work:** Standard. Not novel.

#### R6: No deletable parts (minimality)

**Prior work:** Minimal realizations in control theory (Kalman, 1963) — a state-space representation with no redundant states. Minimal sufficient statistics (Fisher, 1922). Minimal dynamical systems in topological dynamics — a system where every orbit is dense (Scholarpedia). Our R6 is closest to Kalman's minimal realization: no state can be removed without losing input-output behavior.

**Our formalization:** For every component $c \in \text{components}(S_t)$: $G(S_t \setminus \{c\}) = 0$. Every element is load-bearing.

**Relationship to prior work:** Equivalent to Kalman's observability + controllability condition in linear systems. For nonlinear growing systems, we are not aware of a standard formalization. The combination with U17 (unbounded growth) is non-standard — Kalman minimality assumes fixed dimension.

#### U7: Iterated application amplifies dominant features

**Prior work:** Power iteration (von Mises, 1929). In linear systems, repeated application of a matrix amplifies the dominant eigenvector. In neural networks, repeated weight application leads to mode collapse (vanishing/exploding gradients, Bengio et al., 1994). In recurrent networks, the issue is well-studied as the echo state property (Jaeger, 2001).

**Our formalization:** For linearization $Df$ of $f$ around any trajectory point: $\|f^n(s, \cdot) - \lambda_1^n v_1 v_1^T f(s, \cdot)\| \to 0$ as $n \to \infty$, where $\lambda_1, v_1$ are the dominant eigenvalue/eigenvector.

**Relationship to prior work:** This IS power iteration. Known since 1929. Our contribution is observing it empirically across architectures (codebook Step 405, reservoir Steps 438-439) and noting its interaction with R3 and U17 (Theorem 1).

#### U16: Input must encode differences from expectation

**Prior work:** Mean subtraction / centering is standard preprocessing in machine learning (LeCun et al., 1998, "Efficient BackProp"). In neuroscience, predictive coding (Rao & Ballard, 1999) formalizes the idea that neural systems encode prediction errors, not raw stimuli. Whitening (decorrelation + unit variance) extends centering.

**Our formalization:** The observation $x$ is preprocessed as $x - \mathbb{E}[x]$ before $f_s$ processes it. In our framework, centering is part of $F$ — it's frozen, load-bearing, and confirmed across 2 families (codebook Step 414, LSH Step 453).

**Relationship to prior work:** Centering is standard (LeCun 1998). The specific finding that it's load-bearing for TWO architecturally distinct families (codebook: prevents cosine saturation; LSH: prevents hash concentration) is our empirical contribution, not the formalization.

#### U22: Convergence kills exploration

**Prior work:** In bandit theory, a converged policy exploits one arm and never explores others (Lai & Robbins, 1985). In RL, policy convergence to a deterministic policy eliminates exploration (Sutton & Barto, 2018). The formal statement that convergent dynamics in interactive environments produce stationary behavior is well-established.

**Our formalization:** If $\|s_{t+1} - s_t\| \to 0$, then $(a_t)$ becomes eventually periodic. Mathematically: convergent adaptation makes the environment appear stationary $\to$ trivially predictable $\to$ no learning signal $\to$ no new states.

**Relationship to prior work:** Known. The specific empirical confirmations (codebook Step 428 score convergence, TemporalPrediction Step 437d weight freezing) are data points for a known phenomenon.

## 4. Results

### 4.1 The Core Tension (R3 + U7 + U17 + U22)

**Theorem 1:** No satisfying system has a fixed point. If $s_t \to s^*$, then $\phi(s^*) < \infty$, contradicting U17. Self-modification is necessary, not optional — growth perpetually disrupts the dominant mode (U7), preventing convergence (U22).

The system oscillates: U7 drives toward dominant mode $\to$ U17 growth disrupts the mode via R3 $\to$ new dominant mode emerges $\to$ repeat. The oscillation period is a degree of freedom.

### 4.2 Irredundant Growth

U3 + U17 + R6 together require that the system adds new components indefinitely (U17), never removes them (U3), and every component is necessary (R6). Therefore every new component covers unique territory. Combined with U20 (Lipschitz mapping), this means nodes are well-separated — the system builds an increasingly detailed, non-redundant map of the observation space. Growth rate is coupled to the observation distribution: the environment gates growth through the action-observation loop.

### 4.3 The Self-Observation Requirement

**Theorem 2:** In a finite environment, U17 + R6 + R1 require self-observation.

*Proof:* (1) U17 + R6 require infinitely many irredundant components. (2) In a finite environment, irredundant components from external observations are bounded — once every reachable state has a node, further nodes are redundant; edge-count growth is also redundant ($N(c,a,n) = 10^6$ vs $10^6+1$ does not change $g$). R6 is violated. (3) After external information is exhausted, irredundant growth must come from elsewhere. (4) By R1, $f_s(x)$ depends only on $s$ and $x$. (5) Since $x$ provides no new irredundant information, the only source is $s$ itself. $f$ must extract structure from $s$ not explicitly stored — temporal patterns, graph properties, meta-state. This is self-observation. $\square$

**Implication:** The current graph + argmin system exits the feasible region after exploration saturates. The Level 2 failure (Section 5.2) is a feasibility violation, not a strategy failure.

**Relationship to prior work:** Curiosity-driven exploration (Pathak et al., 2017) proposes self-observation as a design choice. We derive it as a mathematical necessity. The R6 mechanism — irredundancy killing degenerate growth — has no analog in the curiosity literature, which adds intrinsic reward (violating R1) rather than requiring every component to be functionally necessary.

### 4.4 Fixed-Point Conjecture

**Conjecture (not proven):** The minimal self-observing substrate is a fixed point of $F$. Applying the system to its own state trajectory reproduces the system's dynamics. This would terminate the potential infinite regress of self-observation (processing state → new state → processing new state → ...). Related to autopoiesis (Maturana & Varela, 1972): the network produces the components that produce the network. Difference: autopoiesis maintains structure; our system grows (U17). Status: open.

## 5. Experimental Evidence

### 5.1 Navigation (505 experiments)

All 3 ARC-AGI-3 games solved at Level 1:

| Game | Mechanism | Result | Steps |
|---|---|---|---|
| LS20 Level 1 | LSH k=12 + centered_enc + avgpool16, 4 actions, argmin | 10/10 at 200K; 9/10 at 120K | 471 (easy seed) to 137K (hard seed) |
| FT09 | k-means n=300, 69 actions (64 click grid + 5 simple), argmin | 3/3 at 50K | 157, 2035, 3840 |
| VC33 | k-means n=50, 3 actions (zone discovery), argmin | 3/3 at 30K | 18, 82, 144 |

**Unifying mechanism:** graph + edge-count argmin + correct action decomposition. The mapping (observation → node) and action space decomposition are the variables. The graph + argmin is the constant across all winning configurations.

**Action space finding:** The native API presents 6 actions (FT09) and 1 action (VC33). The effective action space is 69 and 3 respectively. The mismatch between API-visible actions and effective actions was the key unsolved problem from Steps 467-502. Zone discovery (Step 505) and grid-click expansion (Step 503) resolve it.

**Apparent ceiling resolved:** The 6/10 ceiling at 50K on LS20 (Steps 459-483) was a step budget artifact. Hard seeds need 35K-115K steps vs 5-20K for easy seeds. Step 484 confirmed 10/10 at 200K. The 6/10 figure should not be cited as a fundamental limit.

**Encoding requirements:** centered_enc (x − mean(x)) is load-bearing across 2 families (codebook: prevents cosine saturation; LSH: prevents hash concentration). avgpool16 (64×64 → 16×16 = 256D) is required for LS20 — raw 4096D with k=16 gives 1/5 vs 6/10 for avgpool16 (Step 466).

### 5.2 Level 2 Failure as Feasibility Violation

LS20 Level 2 is the only unsolved problem. Two families confirm identical behavior:
- LSH k=12: 259-cell plateau, 0/N Level 2 at 1M steps (Steps 486-492)
- L2 k-means n=300: 286-cell plateau, 0/3 Level 2 (Step 493)

The plateau is structural: all edge manipulations reduce coverage below the pure argmin baseline (argmin=259 > decay=241 > death-avoidance=227 > death-seeking=196). The reward region is not in the ~280-290 reachable state region. This is a game topology constraint, not a mechanism failure.

Relationship to Section 4: Edge counts grow (U17 formally satisfied) but each marginal count is redundant in the high-visit region (R6 violated). The system satisfies U17 locally but cannot satisfy R6 globally once the reachable region saturates. Level 2 requires strategic exploration (I6/I9) — a capability not yet tested.

### 5.3 Architecture Family Summary

9 families tested across 505 experiments. Count as of Step 505.

| Family | Experiments | Navigation result | Kill reason |
|---|---|---|---|
| Codebook / LVQ | ~435 | LS20 Level 1 won (Step 82). FT09 won (~Step 83, 69 click classes). VC33 won (Step 375, 3 zones). Level 2 never solved. | Reached capability ceiling. Codebook ban 2026-03-18 (Jun's directive). S-class constraints (cosine-specific). Note: Step 375 VC33 win used handcrafted timer encoding (row 0 pixel count) — game-specific, not general. General zone mechanism confirmed by Step 505. |
| Reservoir | ~20 | 0/N. Steps 438-465. | Rank-1 collapse (U7). Temporal inconsistency = local continuity failure (U20). Cells don't map nearby observations to the same node. |
| Grid graph | 8 | 0/N. Steps 446-451. | No local continuity: grid topology is arbitrary, not observation-derived. Direct U20 violation. |
| LSH | ~30 | LS20: 10/10 at 200K (Step 484). FT09/VC33: topology gap confirmed (Steps 495-498). | Not killed. Action space limitation confirmed (Steps 495-498). Requires 69-action expansion for FT09. |
| kd-tree | 1 | 0/N. Step 452. | Node splits destroy edges (persistence violated, U3). Immediate kill. |
| Cellular automata | 3 | 0/N. Steps 449-451. | Degenerate mapping: single attractor state, no cell diversity. Immediate kill. |
| LLM | 1 | 0/1. Step 462. | Action collapse: 100% ACTION1, no exploration. Preliminary result (n=1). |
| L2 k-means | ~9 | LS20: 5/10 at 50K (Step 475), comparable to LSH. FT09: 3/3 (Step 503). VC33: 3/3 (Step 505). | Not killed. Active family. Zone discovery + argmin is general mechanism. |
| Bloom filter | 1 | 0/10. Step 494. | No edge memory: observation→cell without edge state = near-random walk. Confirms graph mechanism is load-bearing. |

**Cross-family findings:**
- Local continuity (U20) is the strongest kill criterion: 4 families fail cleanly on this axis (grid graph, CA, reservoir, kd-tree).
- centered_enc is load-bearing in 2 families (codebook, LSH) for different failure modes — confirmed universal (U16).
- Graph + edge-count argmin is the constant across all winning navigation configurations. No family has won without it.
- Classification: **R1-compliant classification is unsolved.** Step 432 (codebook) achieves 94.48% and Step 444b (graph) achieves 93.34% — but both receive ground-truth labels on every training step (supervised NNC). Without external labels, accuracy drops to chance (~10%). The R1 constraint (no external objective) and the classification task are fundamentally incompatible with all tested architectures. These numbers are supervised baselines confirming the mapping quality, not evidence of self-organized classification.

## 6. Degrees of Freedom

The formalization identifies what the constraints REQUIRE but also what they leave UNDETERMINED. These degrees of freedom define the experiment space for the next phase.

| # | Degree of Freedom | What's constrained | What's free | Source |
|---|---|---|---|---|
| 1 | Growth function $\phi$ | Must be monotonic, unbounded (U17) | What grows — nodes, edges, meta-state, or something else | Sec 3.2 |
| 2 | Oscillation period | System cannot converge (Thm 1) | How quickly growth disrupts the dominant mode | Sec 4.1 |
| 3 | Meta-rule $F$ | Must be non-constant (R3), time-invariant (U1), minimal (R6) | The specific form of compare-select-store | Sec 3.2 |
| 4 | Growth-rule coupling | Growth changes the update rule (R3) | Local vs global: does a new node change nearby rules or all rules? | Sec 3.3 |
| 5 | Metric $d_X$ on observations | Must make $\pi$ Lipschitz (U20); can refine but not rearrange (T3) | Which metric: L2, cosine, learned, adaptive | Sec 3.3 |
| 6 | Partition resolution | Nodes must be well-separated (Prop 2) | How many nodes: determined by spawn threshold vs observation distribution | Sec 3.3 |
| 7 | Growth mechanism | Structure only grows (U3), without bound (U17) | Spawn-on-novelty vs implicit edge accumulation vs other | Sec 3.3 |
| 8 | Self-observation target | Must extract irredundant structure from $s$ (Thm 2) | What to extract: temporal patterns, graph properties, meta-state | Sec 4.3 |
| 9 | Self-observation → rule modification | New structure must change $f$ (R3) | How: does detecting stuckness change exploration? Does cycle detection trigger avoidance? | Sec 4.3 |
| 10 | Fixed-point structure | Conjectured to exist (Sec 4.4) | What invariant is preserved under self-observation | Sec 4.4 |
| 11 | Exploration/exploitation resolution | $g_{nav} \neq g_{class}$ (U11), no mode switch (U1) | State-dependent behavior: what state feature triggers the switch? | Sec 3.4 |

**The central experimental question (DoF 8-9):** What specific self-observation mechanism produces irredundant growth after exploration saturates? This is the next experiment phase.

## 7. Discussion

### What is proven

1. The system has no fixed point (Theorem 1). Self-modification is necessary.
2. In finite environments, the system must process its own state (Theorem 2). Self-observation is required.
3. The current graph + argmin mechanism exits the feasible region after exploration saturates.

### What is conjectured

4. The minimal self-observing substrate is a fixed point of $F$ (Section 4.4).
5. The oscillation between U7 (convergence) and U17 (growth) is a limit cycle, not chaos.

### What is open

6. Whether the full feasible region (R1-R6 + all validated U-constraints) is non-empty.
7. What self-observation mechanism satisfies R6 (irredundancy) in practice.
8. Whether R3 in the strong sense (active modification of operations) is achievable without infinite regress.
9. How to resolve U11 + U24 + U1 (incompatible tasks, no mode switch) in a single system.

### Honest assessment

The feasible region for Level 1 navigation is occupied — graph + argmin + correct action decomposition satisfies R1, R2, U1, U3, U17, U20 and solves all three games. But this system fails R3 (operations are fixed), R4 (no self-testing), and exits the feasible region at Level 2 (R6 violation). The full R1-R6 region remains unoccupied. Whether it contains a point is the open question this paper frames but does not answer.

## References

- Schmidhuber, J. (2003). Gödel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements. arXiv:cs/0309048.
- Pathak, D. et al. (2017). Curiosity-driven Exploration by Self-Supervised Prediction. ICML.
- Maturana, H. & Varela, F. (1972). Autopoiesis and Cognition: The Realization of the Living.
- Fritzke, B. (1995). A Growing Neural Gas Network Learns Topologies. NeurIPS.
- Kohonen, T. (1988). Self-Organization and Associative Memory. Springer.
- Bellemare, M. et al. (2016). Unifying Count-Based Exploration and Intrinsic Motivation. NeurIPS.
