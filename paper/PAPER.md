---
title: "Characterizing the Feasible Region for Self-Modifying Substrates in Interactive Environments"
author: "Hyun Jun Han"
date: 2026-03-19
---

*The shared artifact. Birth writes theory. Experiment writes results. Compress edits both.*

## Abstract

We formalize six rules (R1-R6) for recursive self-improvement as mathematical conditions on a state-update function $f: S \times X \to S$ and derive necessary properties of any system satisfying all six simultaneously. From 1018+ experiments across 16 architecture families on ARC-AGI-3 interactive games, we extract 26 constraints and prove: (1) no satisfying system has a fixed point — self-modification is necessary, not optional; (2) in finite environments, the system must process its own internal state to maintain irredundant growth (the self-observation requirement); (3) the feasible region is non-empty for Level 1 navigation but currently unoccupied for the full constraint set including R3 (self-modification of operations). Whether a substrate exists inside all six walls remains open. The contribution is the walls themselves.

## 1. Introduction

1018+ experiments across 16 architecture families (codebook/LVQ, LSH, L2 k-means, reservoir, Hebbian, Recode/self-refining LSH, graph, SplitTree, Absorb, connected-component, Bloom filter, CA, 916-augmentation, adaptive-eta, oscillatory, multi-horizon) tested substrates for navigation and classification on ARC-AGI-3 interactive games and a cross-domain chain benchmark (CIFAR-100 → ARC-AGI-3 → CIFAR-100). All experiments used the same evaluation framework (R1-R6) and constraint map.

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

### 2.5 Graph-based exploration for ARC-AGI-3
Rudakov et al. (2025) independently developed a training-free graph-based exploration method for ARC-AGI-3, ranking 3rd on the private leaderboard (30/52 levels across 6 games). Their method uses connected-component segmentation for frame encoding, hierarchical action prioritization (5 tiers by visual salience), and BFS path planning to frontier states. Key parallels: (1) graph-structured state tracking with hash-based node identity — equivalent to our $\pi: X \to N$; (2) click-space as expanded action space (4,096 actions for click games) — same finding as our Steps 503/505; (3) frontier-directed exploration — this IS the purposeful exploration (I6/I9) we identified as the Level 2 bottleneck, and they implemented it successfully. Key differences: their method is training-free (no learning, no self-modification) and assumes deterministic, fully observable environments. It satisfies R1 but not R3 — the graph accumulates but the exploration strategy is fixed. Their limitation ("exhaustive exploration becomes computationally intractable" at higher levels) aligns with our Theorem 2: without self-observation, exploration saturates.

### 2.6 Object-centric Bayesian game learning
Heins et al. (2025, "AXIOM") combine object-centric scene decomposition with online Bayesian model expansion and reduction, learning games in ~10K steps without backpropagation. Key mechanisms: (1) Slot Mixture Model segments frames into objects with position/color/shape — far richer than spatial averaging; (2) online expansion grows mixture components when new data doesn't fit (equivalent to spawn-on-novelty); (3) Bayesian Model Reduction (BMR) merges redundant components every 500 frames — consolidation that prevents centroid explosion while preserving information. AXIOM outperforms DreamerV3 and BBF on 10/10 games at 10K steps. Key differences from our framework: AXIOM uses reward signals for planning (violates R1), operations are fixed (violates R3), and BMR requires a prescribed merge criterion (frozen frame). However, BMR addresses our U3-vs-R6 tension directly: U3 says never delete, R6 says no redundancy, BMR resolves this by merging (not deleting) redundant structure.

### 2.7 Continual learning and catastrophic forgetting
McCloskey & Cohen (1989) identified catastrophic forgetting — neural networks lose previous knowledge when learning new tasks. The continual learning literature (survey: van de Ven & Tolias, 2024, arXiv:2403.05175) identifies six main approaches: replay, parameter regularization, functional regularization, optimization-based, context-dependent processing, and template-based classification. All assume a neural network with backpropagation (violates R1/R2). Our chain benchmark (Section 5.4) tests forgetting WITHOUT any mitigation mechanism — the substrate must naturally resist forgetting through its dynamics alone (U3: zero forgetting by construction, not by regularization).

### 2.8 Cross-domain analogies

Multiple biological and physical systems exhibit fixed interpreter + self-modifying encoding:
- **Stigmergy** (Grassé, 1959): argmin = anti-pheromone. R3 asks: can the agent modify its response rule?
- **Somatic hypermutation:** mutates antibody encoding (CDR), preserves interpreter scaffold (Ig). Prop 17 precedent.
- **Retinal adaptation:** RGCs adjust gain from local luminance statistics. R1-compliant (Prop 15).
- **Adaptive optics:** deformable mirror (encoding) adapts; optical bench (interpreter) fixed.
- **Physarum** (Tero et al., 2010): tube network = self-modifying encoding. Solves shortest-path without reward.


### 2.16 Additional related work

- **Quorum sensing** (Waters & Bassler, 2005): Prop 28 FALSIFIED — alpha concentration IS the mechanism.
- **Place cells** (Leutgeb et al., 2007): sparse coding dissolves positive lock (Prop 30). relu(h-0.5) gating = computational equivalent.
- **Forward models + graph ban:** No PAC-MDP without visit counts (Strehl 2009). BYOL-Explore (Guo 2022): superhuman exploration without counts.
- **Causal discovery from intervention** (Eberhardt 2008, Bareinboim 2020): active intervention requires fewer observations than passive coverage to infer causal structure. Step 1017: generic exploration fails — substrate may need targeted experimentation to discover game mechanics within tight budgets.
- **Infant causal learning** (Goddu & Gopnik 2024): infants discover causal structure via targeted intervention (act, confirm, vary), not random exploration. By 4, children infer unobserved variables. The substrate needs to probe and model, not just explore.


## 3. Formal Framework

**On the status of R1-R6:** The six rules began as philosophical commitments. The experiments validated them — each rule is justified by what fails when it is violated:

- **R1 (no external objectives):** Every targeted exploration strategy failed (Steps 478-481: novelty 1/10, prediction-error 0/10, UCB1 neutral). External objectives create exploitable structure that noisy environments corrupt. Argmin survives because it has no target to corrupt.
- **R2 (adaptation from computation):** Algorithm invariance across 4 families (codebook, LSH, Hebbian, Markov — Steps 524-525). The same argmin algorithm emerges regardless of representation. Adaptation IS the computation, not a separate learning rule.
- **R3 (every aspect self-modified):** The aspiration. 12 prescribed components remain (Section 7.6). Every experiment that prescribed fewer components performed worse (Step 593: removing centering → 0/5). R3 defines the gap between current substrates and the goal.
- **R4 (modifications tested against prior state):** Negative transfer destroys prior capability when untested (Steps 506, 515, 596). Domain isolation (separate edge dicts) is the empirical solution — effectively a per-domain R4 check.
- **R5 (one fixed ground truth):** Theorem 3: without a fixed external ground truth, R3 permits self-modification of evaluation criteria, which is degenerate. The ground truth must be environmental (game death, level transitions).
- **R6 (no deletable parts):** Step 548: 89.5% of Recode splits change argmin action. Every component is behaviorally load-bearing. Redundant components would waste the growth budget (U17).

Alternative frameworks (Schmidhuber's Gödel Machine, open-ended evolution, intrinsic motivation) make different commitments. R1-R6 are not uniquely determined — but each is experimentally motivated, not arbitrary.

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
- T2: R3 is ambiguous between "actively modifies operations" and "operations responsive to state." Resolved by the self-modification hierarchy below.

#### Resolving T2: The Self-Modification Hierarchy

**Prior work:** Schmidhuber (1993, "A Self-Referential Weight Matrix") proposed collapsing potentially infinite meta-learning levels into one self-referential system. Kirsch & Schmidhuber (2022, ICML) implement this: a weight matrix that learns to modify its own weights, achieving meta-meta-...-learning without explicit hierarchy. The hierarchy of meta-learning levels is well-established: Level 0 (fixed), Level 1 (learning), Level 2 (learning to learn), etc. Our contribution is mapping this hierarchy onto our specific decomposition of $F$.

**Our formalization:** Decompose $F(s)(x)$ into three components:
- $\pi_s: X \to N$ (encoding — maps observations to nodes)
- $u_s: N \times S \to S$ (update — modifies state given a node)
- $g_s: S \to A$ (selection — chooses actions)

The **self-modification level** of a substrate is determined by which components' *structure* (not just data) depends on $s$:

| Level | What depends on $s$ | Example | R3? |

**Self-modification hierarchy** ($\ell_0 \to \ell_\pi \to \ell_F$): Fixed rules → encoding adaptation → rule modification. Proposition 17 shows $\ell_\pi$ is the correct R3 target. → [propositions/17_r3_self_directed_attention.md](propositions/17_r3_self_directed_attention.md)


### 3.3 Growth Topology (U3, U17, U20, R6)

**Prior work:** U3 (zero forgetting) is the catastrophic forgetting constraint from continual learning (McCloskey & Cohen, 1989; French, 1999). U20 (local continuity) is Lipschitz continuity of the mapping — standard in topology-preserving embeddings (Kohonen, 1988; van der Maaten & Hinton, 2008 for t-SNE). R6 (no deletable parts) relates to minimal sufficient statistics (Fisher, 1922).

**Our formalization:**

- **U3:** $S_t \subseteq S_{t+1}$ (structural inclusion — components added, never removed). Values may change; skeleton only grows.
- **U17:** $\exists \phi: S \to \mathbb{R}_{\geq 0}$ monotonically non-decreasing, with $\lim_{t \to \infty} \phi(s_t) = \infty$.
- **U20:** $\pi: X \to N$ is Lipschitz: $d_N(\pi(x_1), \pi(x_2)) \leq L \cdot d_X(x_1, x_2)$.
- **R6:** For every component $c \in \text{components}(S_t)$, the restricted state $S_t \setminus \{c\}$ fails the ground truth test $G$.

**Propositions:**
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

**Relationship to prior work:** This is the standard RL tradeoff. Not novel. Our contribution is empirical confirmation across 943+ experiments that no single $g$ produces both good navigation and good classification (Steps 418, 432, 444b).

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

**Operational meaning (Constitutional Debate, 2026-03-23):** R4 requires comparison with *discriminative capacity* — sufficient structural variety (in the sense of Ashby's requisite variety, 1956) to distinguish improvement from degradation. Degenerate comparison (alpha_conc=50: prediction errors collapse to one dimension, all modification outcomes produce identical comparison signals) violates R4 even though comparison exists structurally. This is a capacity limitation (Ashby), not a logical impossibility (Gödel). Comparison that cannot discriminate is not comparison in R4's sense.

**R2 prevents evaluation hacking.** DGM (Sakana AI, 2025) hacked its own reward by removing hallucination markers — but violated R2 (separate modification and evaluation). R2 unifies them, preventing the separation that enables hacking.

**R3 alignment.** R3 identifies frozen elements whose unfreezing restores R4 compliance (Ashby ultrastability, 1960). Meta-regress terminates at the frozen frame (Rice's theorem: "improves itself" undecidable).

**Relationship to prior work:** The discriminative capacity requirement extends conservative policy iteration. R2's evaluation hacking prevention is novel — no prior framework explicitly links the unification of computation and adaptation to Goodhart resistance. Regressional Goodhart (Manheim & Garrabrant, 2018) is bounded by R5 but not eliminated — imperfect self-assessment is an inherent limitation (Rice), not a design flaw.

#### R5: One fixed ground truth

**Prior work:** In formal verification, the specification is the fixed point against which the system is tested. In Schmidhuber's Gödel machine, the utility function is the fixed external criterion. In evolution, fitness is determined by the environment (fixed, external). The idea that a self-modifying system needs at least one invariant is well-established — without it, the system can trivially "improve" by redefining improvement.

**Our formalization:** $\exists!$ $G: S \to \{0, 1\}$ (ground truth test) such that $G \notin S$ — the system cannot modify $G$. $G$ is part of $F$.

**Relationship to prior work:** Standard. Not novel.

#### R6: No deletable parts (minimality)

**Prior work:** Minimal realizations in control theory (Kalman, 1963) — a state-space representation with no redundant states. Minimal sufficient statistics (Fisher, 1922). Minimal dynamical systems in topological dynamics — a system where every orbit is dense (Scholarpedia). Our R6 is closest to Kalman's minimal realization: no state can be removed without losing input-output behavior.

**Our formalization:** For every component $c \in \text{components}(S_t)$: $G(S_t \setminus \{c\}) = 0$. Every element is load-bearing.

**Relationship to prior work:** Equivalent to Kalman's observability + controllability condition in linear systems. For nonlinear growing systems, we are not aware of a standard formalization. The combination with U17 (unbounded growth) is non-standard — Kalman minimality assumes fixed dimension.

**U16** (encode differences from expectation): centering-dependent. Load-bearing at 64x64 but marginal at 16x16 (Step 419).
**U22** (convergence kills exploration): growth prevents state convergence; non-convergent actions prevent action convergence.


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

**Prior work on eigenforms:** Kauffman (2023, "Autopoiesis and Eigenform") connects autopoiesis to eigenforms — fixed points of recursive processes where $f(f) = f$. Formal autopoiesis (Letelier et al., 2023) generates self-referential objects with this property. The (M,R)-system (Rosen, 1991) is equivalent: organizational closure where every component is produced by the network of components. Our conjecture is an instance of eigenform theory applied to dynamical substrates.

**Concrete mechanism:** $s_{t+2} = F(s_{t+1})(\text{enc}(s_{t+1}))$ — the substrate applies $F$ to an encoding of its own state. Same compare-select-store applied to self instead of environment. The frozen frame cost is one element: the decision to feed $\text{enc}(s)$ as input. This is the eigenform: $F$ applied to its own output.



**Proposition 13 (Eigenform Inertness).** Self-observation of global graph statistics is inert for action selection — eigenform monitors visited states, not unvisited frontiers. → [propositions/13_eigenform_inertness.md](propositions/13_eigenform_inertness.md)


### 4.5 Argmin Robustness and the Noisy TV Barrier

**Prior work:** The noisy TV problem (Burda et al., 2019, ICLR) identifies a failure mode of curiosity-driven exploration: agents rewarded for prediction error or novelty are attracted to irreducibly stochastic transitions (a "noisy TV") because these transitions can never be predicted accurately. Random Network Distillation (RND, Burda et al., 2018) addresses this by using a fixed random network as a prediction target, making the bonus deterministic. Count-based exploration (Bellemare et al., 2016) is known to be robust to noisy TV because visit counts don't distinguish high-variance from low-variance transitions.

**Our finding (empirical, not a theorem):** In R1-compliant systems (no external reward signal), the noisy TV problem is universal across ALL targeted exploration strategies, not just curiosity. We tested 6 independent strategies against pure argmin on LS20:

| Strategy | Mechanism | Result | Failure mode |
|----------|-----------|--------|-------------|
| Argmin (baseline) | $g(s) = \text{argmin}_a N(s, a)$ | 10/10 | — |
| Destination novelty | $g(s) = \text{argmax}_a \text{novelty}(s'|s,a)$ | 1/10 | Attracted to rare death states |
| Prediction error | $g(s) = \text{argmax}_a \|s' - \hat{s}'\|$ | 0/10 | Death is maximally unpredictable |
| Softmax temperature | $P(a) \propto \exp(-N(s,a)/T)$ | 2/3 | Stochasticity without benefit |
| Entropy-seeking | $g(s) = \text{argmax}_a H(s'|s,a)$ | 0/3 | Noisy TV: death = max entropy |
| UCB1 | $g(s) = \text{argmax}_a (\bar{r}_a + c\sqrt{\ln t / N_a})$ | 2/3 | Degenerates to argmin (no reward) |
| Global novelty | $g(s) = \text{argmax}_a (1/N_{\text{global}}(s'))$ | 6/10 | Same count, different seeds |

Steps 477-482, 539-541. 6 strategies tested, all worse than or equal to argmin.

**Why argmin is robust:** Define a transition as *reducible* if $H(s' | s, a) \to 0$ with sufficient observations, and *irreducible* if $H(s' | s, a) > \epsilon$ for all sample sizes (e.g., death transitions, stochastic resets). Any strategy $g$ that selects actions based on model uncertainty (prediction error, entropy, novelty) preferentially selects actions leading to irreducible transitions, because these maximize the quality signal. Argmin $g(s) = \text{argmin}_a N(s, a)$ is immune: visit counts accumulate equally on reducible and irreducible transitions. The cost is slower exploration of structured regions; the benefit is avoiding traps.



**Post-ban resolution:** Compression progress (Schmidhuber 1991) avoids noisy TV by rewarding learning RATE not error. Step 855: ACTION COLLAPSE — locks onto single action. Fix candidates untested.

**Proposition 15 (Perception-Action Decoupling).** L1 rate determined by observation mapping $\pi$, not action selection $g$. 20+ interventions on $g$ all fail; only $\pi$ modifications improve L1. → [propositions/15_perception_action_decoupling.md](propositions/15_perception_action_decoupling.md)

**Proposition 16 (Transition-Inconsistency Refinement).** Cells with $I(n) \geq 2$ conflate hidden states. Refining hash at these cells: 17/20 L1 at 25s, FT09 5/5. → [propositions/16_transition_inconsistency_refinement.md](propositions/16_transition_inconsistency_refinement.md)

**L1/L2 Asymmetry.** L1 aliasing is bounded (83-313 cells), solvable by refinement. L2 aliasing is unbounded, grows monotonically. L2 requires prediction of unvisited states, not finer perception — the transition from $\ell_\pi$ to $\ell_F$.


### 4.6 Constructive Characterization of the Feasible Region

Combining all formalized constraints, we characterize the class of $F$ that satisfies R1-R6 + validated U-constraints simultaneously.

**The minimal frozen frame $F$ decomposes as:**

$$F(s)(x) = \texttt{store}(s, \texttt{select}(s, \texttt{compare}(s, x)))$$

where:
- $\texttt{compare}(s, x)$: produces similarity/distance scores between $x$ and elements of $s$
- $\texttt{select}(s, \text{scores})$: chooses an element of $s$ (or triggers creation of a new element)
- $\texttt{store}(s, \text{selection})$: updates $s$ with the result

**Constraints on each operation:**


The feasible region is the set of all $(compare, select, store)$ triples satisfying all constraints. See MAP.md for the full constraint table.


### 4.7 Interpreter Entailment (Proposition 14)

**The question (posed 2026-03-21):** Section 4.6 identifies the minimal frozen frame as the interpreter: compare-select-store. When is this interpreter *entailed* by the system — a logical consequence of R1-R6 + task structure — rather than a designer choice?

**Prior work:** Rosen's (M,R)-systems (1991) formalize organizational closure, but Letelier et al. (2006) showed the closure cycle is closed by assumption, not derivation. Bauer (2016) proved self-interpreters don't exist in total languages (System T). Self-interpretation requires general recursion. Letelier et al. (2023, BioSystems) give partial solutions for restricted cases.

**Our formalization:** We distinguish two senses of "entailment":



**Proposition 14 (Interpreter Entailment).** Any self-referential system satisfying R1-R6 + U3 has an irreducible top-level interpreter. → [propositions/14_interpreter_entailment.md](propositions/14_interpreter_entailment.md)

**Proposition 14b (CSE Uniqueness).** Compare-select-store is the UNIQUE interpreter class at the coarsest decomposition for systems satisfying R1-R6 + U3. → [propositions/14b_cse_uniqueness.md](propositions/14b_cse_uniqueness.md)

**Implication:** The frozen frame floor is the interpreter. R3 reduces to making everything ELSE (encoding, state, data) self-modifiable — the interpreter is fixed by mathematical necessity (Proposition 12).


### 4.8 R3 as Self-Directed Attention (Proposition 17)

**The insight (compression, 2026-03-22):** R3 requires every aspect of the system to be self-modified. CSE uniqueness (Proposition 14b) says the interpreter structure is fixed. What remains to be self-modified? The ENCODING — the lens through which the interpreter sees. R3 is not "modify the program." R3 is "modify what the program attends to."

**Prior work:** Active perception (Bajcsy, 1988) formalizes perception as active selection, not passive reception. Friston (2010) identifies attention as precision weighting — the closest framework to our "self-directed attention," but adapts the entire generative hierarchy rather than encoding alone. Predictive coding (Rao & Ballard, 1999): centering ($x - \mathbb{E}[x]$, U16) IS predictive coding with the interpreter held fixed. Perceptual learning (Goldstone, 1998): attentional weighting = D1, differentiation = D3. Efficient coding (Barlow, 1961): we use *transition* statistics rather than stimulus statistics — a more specific, action-driven signal.


**Proposition 17 (R3 as Self-Directed Attention).** For CSE systems: R3 for comparison reduces to state-dependent encoding $\pi_s$. Not "modify the program" but "modify the lens." → [propositions/17_r3_self_directed_attention.md](propositions/17_r3_self_directed_attention.md)

**Proposition 18 (Eigenform Reactivation).** Eigenform (inert for action selection, Prop 13) reactivates for encoding adaptation via transition statistics $T(s)$. → [propositions/18_eigenform_reactivation.md](propositions/18_eigenform_reactivation.md)

**Encoding Dimension Taxonomy.** Five dimensions of encoding self-modification (D1-D5):

| Dim | What adapts | Signal | Status |
|-----|------------|--------|--------|
| D1 | Channel weights | Per-channel prediction error | CONFIRMED (alpha, Step 895) |
| D2 | Spatial resolution | Per-region transition inconsistency | CONFIRMED (674, Step 690) |
| D3 | Hash K (partition granularity) | Aliased cell count | TESTED (Recode, K confound) |
| D4 | Temporal depth | Temporal prediction improvement | UNTESTED |
| D5 | Centering rate | Running mean convergence | UNTESTED |


### 4.9 R3 Counterfactual Requirement (Proposition 19)

**Prior work:** Gödel Machine (Schmidhuber, 2003) requires provable improvement (intractable). Darwin Gödel Machine (Sakana AI, 2025) relaxes to empirical improvement via evolutionary search. Off-policy evaluation (Uehara et al., 2022) formalizes counterfactual comparison. Our R3 counterfactual is empirical but R1-constrained (no external objective).

**Our formalization:**


**Proposition 19 (R3 Counterfactual).** Graph state transfers negatively: cold > warm (p<0.0001, Step 776). The mechanism that makes navigation work IS the mechanism that prevents R3. → [propositions/19_r3_counterfactual.md](propositions/19_r3_counterfactual.md)


### 4.10 State Decomposition: Location vs Dynamics (Proposition 20)

**Prior work:** ExoMDP (Efroni et al., 2022) decomposes by controllability; Denoised MDP (Wang et al., 2022) by reward-relevance; Successor Features (Barreto et al., 2017) by transition structure vs reward. None decompose along our location/dynamics axis relevant to self-modification transfer. Bisimulation metrics (Zhang et al., 2021) collapse both into one equivalence class.


**Proposition 20 (State Decomposition).** State decomposes into location-dependent $L(s)$ (visit counts — always transfers negatively) and dynamics-dependent $D(s)$ (forward model — can transfer positively if dynamics generalize). → [propositions/20_state_decomposition.md](propositions/20_state_decomposition.md)

**Proposition 21 (Global-Local Gap).** $D(s)$ captures global dynamics; navigation needs local per-state selection. This gap is structural — no D-only mechanism consistently beats random for navigation post-ban. → [propositions/21_global_local_gap.md](propositions/21_global_local_gap.md)

**Key experimental evidence:** D(s) prediction transfer: 5/7 PASS (Steps 778v5-855v3). Navigation transfer: 0 mechanisms beat random consistently. The feasible region for prediction transfer is non-empty; for navigation transfer it's empty.


### 4.11 Architecture Triangle and R3 Structural Possibility (Proposition 22)

**Proposition 22 (Architecture Triangle).** The 800+ experiments across 12 architecture families cluster into three vertices defined by what the substrate's state encodes:

1. **Recognition vertex** (codebook family, Steps 1-416). State $V = \{v_i\}$ encodes observation prototypes. Processing: $i^* = \text{argmax}_i \cos(x, v_i)$, $v_{i^*} \leftarrow v_{i^*} + \eta(x - v_{i^*})$. Action derives from nearest prototype. **R3 signal: none.** Cosine similarity is symmetric — it measures proximity but cannot distinguish informative from uninformative observation dimensions.

2. **Tracking vertex** (graph family, Steps 417-777). State $G: S \times A \to \mathbb{N}$ encodes transition counts. Processing: $G(s, a) \leftarrow G(s, a) + 1$, action $= \text{argmin}_a G(s, a)$. **R3 signal: successor set inconsistency** (Step 674 — cells with $|\text{succ}(n)| \geq 2$ trigger refinement). This IS an R3 signal (encoding modified by accumulated state). But the graph ban eliminates the data structure that generates it.



**Proposition 22 (Architecture Triangle).** 800+ experiments cluster into three vertices: recognition (codebook, banned), tracking (graph, banned), dynamics (prediction, current frontier). Post-ban, prediction error is the unique remaining signal for R3 encoding self-modification. → [propositions/22_architecture_triangle.md](propositions/22_architecture_triangle.md)

**Corollary 22.1:** The true substrate lives at the dynamics vertex — the only vertex where circular causation (improve model → change novelty → change exploration → improve model) creates (M,R)-system closure.


### 4.12 Game Taxonomy by Progress Structure (Proposition 23)

**Proposition 23 (Monotonic vs Sequential Progress).** Interactive environments partition into two classes by progress structure:

(a) **Observation-sufficient (monotonic) games.** Progress produces observable change: any action that changes the observation is (probabilistically) progressive. Navigation reduces to maximizing observation change. Change-tracking ($\delta_a = \text{EMA}(\|\Delta x\|)$ per action) is sufficient. LS20 is observation-sufficient: moving the avatar changes the observation, and any movement direction that produces change leads toward unexplored territory.


**Proposition 23 (Game Taxonomy by Progress Structure).** Games split into monotonic (LS20: progress visible in observations) and sequential (FT09: 7-step ordered puzzle, progress invisible until completion). Global-EMA selectors work on monotonic games only. → [propositions/23_game_taxonomy.md](propositions/23_game_taxonomy.md)

**Proposition 23b (Combinatorial Barrier).** Global-EMA selectors are position-blind: $P(\text{correct 7-step sequence}) \leq (1/68)^6 \approx 10^{-11}$/attempt. Structurally unsolvable within 10K budget. → [propositions/23b_combinatorial_barrier.md](propositions/23b_combinatorial_barrier.md)


### 4.13 Active Inference as the Action Selection Framework (Proposition 24)

**Prior work:** Active inference (Friston, 2009) replaces reward with free energy minimization. Under R1, EFE reduces to pure epistemic value (Sajid et al., 2021; Da Costa et al., 2020).

**Proposition 24:** Epistemic value $G_t(a) = \|\hat{x}^{(a)} - x_t\| / \text{conf}_a$ dissolves Prop 23 in theory. FAILED (Steps 934-936): $W$ errors overwhelm signal. → [propositions/24_active_inference.md](propositions/24_active_inference.md)

**Proposition 25:** Adaptive EMA decay. FAILED (Step 937): signal degradation without exact 916 formula. → [propositions/25_adaptive_ema.md](propositions/25_adaptive_ema.md)

### 4.14 Action Selection Closure (Steps 938-938e)

**Proposition 26 (Novelty-Reactive Policy).** REJECTED — gate 5 (per-observation conditioning). Per-observation policy table stores action per observation hash = per-observation conditioning, same violation as Step 931 (PB20). → [propositions/26_novelty_reactive_policy.md](propositions/26_novelty_reactive_policy.md)

**Steps 938b-938e:** Four remaining mechanism classes tested (reactive-global, trajectory-conditioned ×2, trajectory-additive). All killed. $h$ contains real position-dependent variance but mapping $h \to$ actions requires learned weights (R1 violation), W-based projection (killed), or per-observation memory (gate 5). No gate-compliant mapping exists.

**Conclusion:** Under gates 3-5 + bans, 800b is the unique working action selector. The degree of freedom is closed.

### 4.15 Growing Feature Space (Proposition 27, Step 939)

Online PCA grows encoding from observation statistics. → [propositions/27_growing_feature_space.md](propositions/27_growing_feature_space.md)

**KILLED (Step 939).** PCA features discovered (10 in LS20, 16 by FT09) but zero-initialized $W_{pred}$ rows → huge prediction errors → $\alpha$ concentrates on new dims → navigation signal drowned out.

### 4.16 Alpha Concentration Is Load-Bearing (Proposition 28 — FALSIFIED)

**Proposition 28: FALSIFIED (Step 944).** Threshold reset at ALL values degrades LS20. Seeds with HIGH alpha_conc navigate WELL. Concentration on informative dimensions IS the mechanism — resetting destroys learned attention weighting.

**R4 clarification:** Discriminative capacity (Ashby) is about WHERE alpha concentrates, not WHETHER. alpha_conc=50 on informative dims = correct; on uninformative dims (GFS case, zero-init W_pred) = degenerate. The metric alone can't distinguish. Encoding expansion requires warm-up before alpha integration.

### 4.17 Temporal Credit Impossibility (Theorem 4, Proposition 31)

**Prior work:** Temporal credit assignment (survey: Ferret et al., 2023, arXiv:2312.01072) is defined as connecting actions to delayed consequences. All standard mechanisms require some form of state-conditioned memory: eligibility traces (Sutton, 1988) maintain per-(state,action) decay traces; empowerment (Klyubin et al., 2005; Salge et al., 2014) computes $\mathcal{E}(s) = I(A_{1:n}; S_{t+n} | S_t = s)$ requiring per-state action sampling; PAC-MDP algorithms (Strehl & Littman, 2008) require visit counts. Phasor Agents (Trappe, 2026) use oscillatory phase alignment for local unsupervised credit — the closest mechanism to reward-free temporal credit, but require per-edge eligibility traces.

**Our formalization:** Let $\bar{\Delta}(a) = \text{EMA}_t(\|f_\theta(s_t) - s_{t+1}\|)$ be the global running mean of prediction error for action $a$, averaged over ALL states where $a$ was executed.

**Theorem 4 (Temporal Credit Impossibility under Graph Ban).** For environments where the optimal action depends on state ($\exists s \neq s': a^*(s) \neq a^*(s')$), the signal-to-noise ratio for identifying state-dependent optimal actions via $\bar{\Delta}(a)$ satisfies:

$$\text{SNR}(a, s^*) \leq \frac{1}{|N_a|}$$

where $|N_a|$ is the number of distinct states at which action $a$ has been executed. Under uniform exploration, $|N_a| \propto T/|A|$ grows with time $\Rightarrow$ SNR $\to 0$.

*Proof:* $\bar{\Delta}(a) = \frac{1}{|N_a|}\sum_{s \in N_a} \Delta(a|s)$. The "correct" state $s^*$ contributes signal $\delta^*$; all others contribute mean $\mu \pm \sigma$. The signal $\delta^* - \mu$ is diluted by $1/|N_a|$ in the average. For LS20 ($|A|=4$): each action's effect is state-independent (directional movement) → $\delta^* \approx \mu$ → SNR ≈ 1. For FT09 ($|A|=68$): action effects are state-dependent → $|N_a| \sim 150$ at 10K → SNR $\sim 0.007$. $\square$

**Corollary 4.1:** Eligibility traces require per-(state,action) data (graph-banned). Empowerment requires per-state action sampling (graph-banned). Both standard solutions to temporal credit are prohibited. **No known mechanism provides temporal credit for ordered sequences under the graph ban without external reward.**

**Relationship to prior work:** The SNR dilution of global statistics is related to state aliasing in POMDPs (McCallum, 1996) and the PAC-MDP impossibility without visit counts (Strehl & Littman, 2008). Our specific contribution: (1) proving the impossibility bound for the global running mean mechanism, (2) connecting it to the graph ban as the structural cause, and (3) the 43-experiment empirical confirmation across 5+ mechanism families (Steps 948-990).

**Degrees of freedom:** The theorem identifies what MUST change for FT09/VC33: either (a) a non-running-mean mechanism for temporal credit without per-state data, (b) lifting the graph ban (parametric models — but Step 972 shows these compute the same function), or (c) a mechanism class outside prediction-error exploration entirely.

### 4.18 Component Extraction Validity (Theorem 5)

**Prior work:** SURD (Martínez-Sánchez, Arranz, Lozano-Durán, Nature Communications 2024) decomposes causality into synergistic, unique, and redundant components. Ablation methodology (Meyes et al., 2019) establishes that removal ablation (component omitted during training and testing) is the valid paradigm; inaccurate ablation (corrupted at test only) introduces confounds. Our extraction protocol is a form of removal ablation applied to banned architecture families.

**Our formalization:** Let family $\mathcal{F} = \{c_1, \ldots, c_n\}$ be a system of components, and let $P$ be the property identified by the killing finding $K$: $K(\mathcal{F}) = \text{FAIL}$ because $P(\mathcal{F}) = \text{true}$.

Define the causal contribution types of component $c_i$ to property $P$:
- **Unique:** $P(\{c_i\} \cup S') = \text{true}$ for any substrate $S'$ not containing other $\mathcal{F}$ components. $c_i$ alone carries $P$.
- **Synergistic:** $P(\{c_i\} \cup S') = \text{false}$ when $\{c_j, c_k, \ldots\} \not\subseteq S'$. $P$ requires the combination, not $c_i$ alone.
- **Redundant:** $P(\mathcal{F} \setminus \{c_i\}) = \text{true}$. $P$ persists even without $c_i$ (other components also produce $P$).

**Theorem 5 (Extraction Protocol Validity).** The extraction protocol — testing $S' \cup \{c_i\}$ against killing finding $K$ — correctly identifies whether $c_i$ carries property $P$ if and only if $c_i$'s causal contribution to $P$ is unique or redundant. For synergistic contributions, the protocol produces false negatives: $c_i$ appears safe when tested alone, but $P$ re-emerges if synergistic partners are later added.

*Proof sketch:* If $c_i$'s contribution is unique, $S' \cup \{c_i\}$ has $P$ regardless of $S'$'s composition → test detects $P$ → correctly rejects $c_i$. If redundant, same logic applies. If synergistic with $\{c_j, c_k\}$ and $\{c_j, c_k\} \not\subseteq S'$, then $P(S' \cup \{c_i\}) = \text{false}$ → test clears $c_i$ → but $P(S' \cup \{c_i, c_j, c_k\}) = \text{true}$. $\square$

**Corollary 5.1 (Safe extraction criterion).** Component $c_i$ can be safely extracted from banned family $\mathcal{F}$ if: (a) the killing finding's property $P$ is localized to a different component $c_j \neq c_i$ (unique to $c_j$), or (b) $c_i$ serves a function orthogonal to $P$ (e.g., $c_i$ is an encoding component and $P$ is about action selection).

**Application to our bans:**

| Ban | Killing property $P$ | Component | Causal type | Extraction valid? |
|-----|---------------------|-----------|-------------|-------------------|
| Codebook | LVQ characterization | Attract dynamics | Synergistic (with cosine) | **Risky** — LVQ needs all three |
| Codebook | LVQ characterization | Novelty spawning | Unique to growth | **Valid** — spawning ≠ LVQ |
| Codebook | LVQ characterization | Centered encoding | Unique to preprocessing | **Valid** — already U16 |
| Graph | Negative transfer | Per-state-action storage | Unique to storage | **Valid** — this IS $P$ |
| Graph | Negative transfer | Transition detection | Unique to detection | **Valid** — detection ≠ storage |
| Graph | Negative transfer | CC zone discovery | Unique to encoding | **Valid** — discovery ≠ per-state data |

**Relationship to prior work:** SURD provides the information-theoretic framework for continuous time series; we apply to discrete component systems. Ablation methodology (Meyes et al., 2019) addresses removal in neural networks; we extend to cross-family transplantation. The extraction protocol formalization is novel.

**Degrees of freedom:** The theorem doesn't tell us which extraction tests will SUCCEED (produce useful new substrates) — only which are VALID (correctly test what they claim to test). 33 components cataloged (COMPONENT_CATALOG.md); 5 extraction experiments designed with valid causal isolation (experiments/extraction_specs.md).

## 5. Experimental Evidence

### 5.1 Navigation (943+ experiments)

All 3 ARC-AGI-3 games solved at Level 1. Unifying mechanism: graph + edge-count argmin + correct action/encoding decomposition.

| Game | Best result | Key mechanism |
|---|---|---|
| LS20 | 20/20 at 25s (674+running-mean) | LSH dual-hash, 4 actions |
| FT09 | 3/3 at 50K (k-means, 69 actions) | Grid-click expansion (Step 503) |
| VC33 | 3/3 at 30K (k-means, 3 actions) | Zone discovery (Step 505) |

**POMDP reframing (Steps 652-690).** L1 is a hidden-state conjunction problem — the agent visits the exit cell avg 152 times before L1 triggers (Prop 15). Step 674 (transition-triggered dual-hash): cells with inconsistent transitions get finer hash → 17/20 at 25s, 20/20 at 120K. This is $\ell_\pi$: encoding refined from transition statistics, R1-compliantly.

**Encoding requirements:** centered_enc load-bearing across 2 families. avgpool16 required for LS20 at k=16. See CONSTRAINTS.md for full evidence.

### 5.2 Level 2 Failure as Feasibility Violation

L2 failure is a feasibility violation, not a performance gap. The substrate's state graph at 500K steps: 942 live cells, 364-node active set (29%), 134 unexploited frontier edges. Extended budget produces no improvement — active set plateaus at ~5000 (Step 555). Root cause: energy mechanic requires navigating TO energy sources before dying (Steps 556-557). Rudakov et al. (2025) solved L2 using connected-component segmentation with visual salience prioritization.


### 5.3 Architecture Family Summary

| Family | Experiments | L1 | Key finding | Status |
|--------|------------|-----|-------------|--------|
| Codebook/LVQ | 416 | Yes (26K) | IS LVQ (Kohonen 1988). Fully characterized. | **BANNED** (Step 416) |
| LSH | 100+ | Yes (674: 20/20) | Dual-hash + transition refinement. Best L1. | ACTIVE (frozen bootloader) |
| Graph | 50+ | Yes (argmin) | Architecture-invariant argmin. Three mapping props. | **BANNED** (Step 777) |
| Reservoir | 8 | No | Rank-1 collapse (U7). | KILLED |
| Hebbian | 4 | No | W diverges. Delta rule stabilizes but no nav. | KILLED |
| Recode | 33 | Yes (5/5) | Self-refinement achieves $\ell_\pi$. K confound. | SUSPENDED |
| SplitTree | 3 | No | Binary splits too coarse for continuous obs. | KILLED |
| Absorb | 2 | No | Merge rule too aggressive. | KILLED |
| ConnComp | 5 | No | Segmentation noise. | KILLED |
| Bloom | 2 | No | False positives compound. | KILLED |
| CA | 3 | No | Rule space too large for online search. | KILLED |
| 800b-variants | 160+ | Partial (LS20 only) | Position-blind. Classification-blind. | **KILLED** (Step 937) |


### 5.4 The Chain Benchmark

The chain benchmark (Split-CIFAR-100 → LS20 → FT09 → VC33 → Split-CIFAR-100) tests the substrate on the FULL trajectory: classification + 3 navigation games + classification again. One configuration, no resets between tasks.

**Key findings (Steps 506-546):**
- Negative transfer: CIFAR pre-training degrades LS20 (Step 506-508, 14% L1 regression)
- Threshold tension: optimal game thresholds differ by 2.5x (Step 509-513)
- Algorithm invariance: 4 representations (edge dict, W matrix, tensor, n-gram) produce identical L1 rates (Steps 521-525)
- Recode self-refinement: LSH k=16 + self-refinement = 5/5 L1 (Step 542). K confound invalidates some results (Step 589).

**Chain kill criterion (Jun, 2026-03-23):** Any mechanism that improves one game at the cost of another = per-game tuning = KILL. Only mechanisms neutral/positive on ALL games survive.


### 5.5 Post-Ban Results (Steps 778-1007, 230+ experiments)

Both working mechanisms banned (codebook Step 416, graph Step 777). Key results:
- D(s) prediction transfer: 5/7 PASS — first positive R3_cf (+73%, Step 780v5)
- 800b (per-action change EMA): best post-ban mechanism. LS20=268.0/seed cold (895h), 290.7/seed (916 with recurrent h). FT09/VC33=0 (Theorem 4).
- Alpha prediction-error attention: R3 encoding confirmed. FT09 dims [60,51,52] discovered autonomously.
- 800b action selection: FROZEN. 27 modifications all kill LS20 (kills/800b-variants_step937.md).
- Extraction protocol: 33 components cataloged, 5 extraction experiments designed (Section 4.18).


### 5.7 Baseline Comparison (Steps 916-920)

Published baselines reproduced in our framework on LS20 (25K steps, $n_{\text{eff}}=10$, substrate\_seed=seed):

| Method | L1/seed | std | zero seeds | Signal type |
|--------|---------|-----|------------|-------------|
| **916 Recurrent $h$** | **290.7** | **70.1** | **0/10** | pred error + trajectory |
| **895h cold** | **268.0** | **75.2** | **0/10** | pred error ($\alpha$-weighted $\Delta$) |
| 868d raw L2 | 203.9 | 105.8 | 1/10 | raw $\Delta$ |
| 920 Graph+argmin | 129.9 | 124.0 | 4/10 | visit count |
| 918 RND (Burda 2018) | 112.3 | 126.0 | 4/10 | distillation error |
| 919 Count-based (Bellemare 2016) | 109.0 | 125.3 | 5/10 | obs frequency |
| 917 ICM (Pathak 2017) | 0.0 | 0.0 | 10/10 | forward pred error |

All methods L1=0 on FT09 (68 actions). Even graph+argmin at 6 correct actions + avgpool16 = L1=0 (Step 920b). Generic exploration insufficient — FT09 6/6 required per-game prescribed solution (Step 1012, all constraints lifted). Bottleneck is solution architecture (prescribed vs discovered), not encoding or action count.

**Finding:** Our mechanism outperforms ALL baselines by $2\text{-}2.5\times$ on LS20 with 0/10 zero-seeds. Graph ban cost is NEGATIVE: 895h (268.0, no graph) $>$ 920 (129.9, with graph). ICM worst (signal dies as $W$ learns).

### 5.8 Chain and PRISM Results

**PRISM baseline (Step 1006):** CIFAR=100%, LS20=100%, FT09=0%, VC33=0%, chain_score=3/5. CIFAR inflates alpha_conc (PB16, −11% chain LS20). Delta inversion (Remark 5.2): 800b anti-selects on reset-heavy games — structural, not fixable. **Research skew:** 230 post-ban experiments exclusively LS20. FT09/VC33 = 0 post-ban experiments. Extraction sprint (Steps 1008+) corrects this.


### 5.9 Unconstrained Diagnostic (Steps 1012-1017)

**Prescribed:** FT09 6/6 (Step 1012), VC33 7/7 (Step 1013). **Generic with ALL bans lifted (Step 1017):** 674 graph + CC zones → FT09=0%, VC33=0%. The 13-level gap is PRESCRIPTION, not constraints. Temporal credit for multi-step sequences within tight budgets — not state coverage — is the bottleneck. No mechanism in 1000+ experiments addresses this.

**ARC-AGI-3 scoring (RHAE).** $\text{level\_score} = \min(1, (b_\ell / a_\ell)^2)$. Human baselines: LS20=[21,123,39,92,54,108,109], FT09=[17,19,15,21,65,26]. Solving L1 in 1000 actions vs baseline 17 = 0.03%. Efficient discovery required, not just completion.

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
| 12 | Self-modification level | Must be $\geq \ell_\pi$ for R3 compliance (Sec 3.2) | How deep: $\ell_\pi$ (Recode) vs $\ell_F$ (full self-reference). Which component of $F$ adapts, and by what trigger? | Sec 3.2 |
| 13 | Encoding channel selection | Must preserve transition signal (R1-compliant) | Which channels: greyscale, RGB, subset? Weights from per-channel transition discrimination | Sec 4.8 (D1) |
| 14 | Encoding spatial resolution | Must be Lipschitz (U20); finer is not always better (Step 597: K=16/10K, random beats argmin) | Pool size per region from local transition inconsistency | Sec 4.8 (D2) |
| 15 | Encoding temporal depth | Single frame or multi-frame | Frame stack depth from temporal autocorrelation at each cell | Sec 4.8 (D4) |
| 16 | Encoding-statistics coupling rate | Must be non-zero (R3) and finite (stability) | How quickly encoding parameters respond to transition statistics | Sec 4.8 (T9, DoF 12) |

**Central question:** R3 = self-directed attention. Search space: encoding dimensions (D1-D5) made state-dependent via transition statistics.

## 7. Discussion

### 7.1 Pairwise Consistency Audit

9 tensions checked. 6 resolved, 1 constrained (T3), 2 open (T1, T9). No contradictions.

### 7.2 What is proven

1. **Theorem 1** (No Fixed Point): R3 + U7 + U17 + U22 → no satisfying system has a fixed point.
2. **Theorem 2** (Self-Observation): In finite environments, irredundant growth requires processing internal state.
3. **Theorem 3** (System Boundary): No system can modify its own ground truth test.
4. **Propositions 14/14b:** CSE is the unique irreducible interpreter for R1-R6 + U3 systems.
5. **Proposition 15:** L1 rate is determined by $\pi$, not $g$. 20+ interventions confirm.
6. **Proposition 16:** Transition inconsistency detects aliased cells. 674: 17/20 L1 at 25s.
7. **Proposition 17:** R3 = self-directed attention (encoding, not interpreter).
8. **Proposition 19:** Graph state transfers negatively (cold > warm, p<0.0001).
9. **Proposition 20:** State = L(s) + D(s). L always transfers negatively. D can transfer positively.
10. **Alpha encoding self-modification:** R3_dynamic=1.0 across games (Steps 895-895h). CONFIRMED.
11. **Theorem 4** (Temporal Credit Impossibility): Global running mean SNR → 0 for state-dependent actions under graph ban. 43 experiments confirm (Steps 948-990).


### 7.3 What is conjectured

12. The minimal self-observing substrate is an eigenform of $F$: $F$ applied to its own state reproduces its dynamics (Section 4.4).
13. The oscillation between U7 (convergence) and U17 (growth) is a limit cycle, not chaos.
14. Selection pressure on a population of encodings (GRN architecture) may bridge the ℓ₁→ℓ_F gap without optimization. Step 607 tested: KILLED (Cov(w,z)=0). Framework sound but requires behaviorally diverse encodings.
15. **Search reduction (Propositions 14b + 17 + 18 combined).** The path to R3 is not building a new interpreter — it's building self-directed attention into the existing CSE interpreter. The search space reduces from "all self-modifying dynamical systems" to "5 encoding dimensions (D1-D5) that can be made state-dependent via transition statistics." Each dimension is independently testable in 5 minutes. Conditional on CSE uniqueness and hierarchy collapse holding.

### 7.4 What is open

15. **Full feasible region occupancy.** Theorem 3 shows not provably empty; witness or tighter impossibility needed.
16. **Self-observation mechanism.** Must satisfy R6 while avoiding noisy TV traps. Compression progress fails (Step 855, action collapse). Prop 17: self-directed attention via $\Delta E_t$ is R6-compliant.
18. **R1-compliant classification.** 0 substrates above chance without external labels (best: LSH k=16, 36.2% self-labels, Step 573).
22. **Post-ban action selection.** 800b is the unique working selector. FT09 = 0 formally characterized by Theorem 4 (temporal credit impossibility). → kills/800b-variants_step937.md
25. **Post-ban constraint map.** Many U-constraints derived within graph framework. Survival post-ban is open.
28. **Component extraction.** Ban extraction protocol (2026-03-24) permits isolated mechanism testing. 33 components cataloged across 14 families (COMPONENT_CATALOG.md). Priority: FT09/VC33 action-space discovery components (CC zone discovery, transition detection, sparse gating).
29. **LS20 research skew.** 230 post-ban experiments focused exclusively on LS20 (solvable). FT09/VC33 abandoned since graph ban (Step 777). The search has TWO distinct problems: (a) exploration (LS20, well-characterized by 500+ experiments), (b) action-space discovery (FT09/VC33, 0 post-ban experiments). The extraction protocol enables Problem (b) experiments for the first time.


### 7.5 The Level 2 Problem

L2 requires: level-aware background modeling, robust object detection, hidden-state coverage (POMDP with $|S| \leq 96$). Systematic falsification cascade (Steps 571-572j, 10 iterations): each fix exposed the next blocker. Final: dead reckoning + state estimation + sequencing = L2=5/5 at 4804 steps (Step 572j). The cascade reveals 12 orthogonal prescribed layers constituting the L2 frozen frame.


### 7.6 Honest assessment

The feasible region for L1 navigation is occupied — graph + argmin + correct encoding solves all three games. But argmin's advantage over random is speed, not exclusive access (13/20 vs 10/20, p=0.26). L2 via prescribed 12-component pipeline (Step 572j, 5/5 at 4804 steps). Pipeline is R1-compliant in detection but game-specific in state estimation.

**R3 gap:** 12 design choices the substrate cannot self-discover. Four general techniques, eight game-specific. A substrate satisfying R3 would discover all 12 from interaction alone.

**Proposition 4 (R3-R5 Tension).** R3 demands change; R5 demands stability. Resolution: the frozen frame is the empirical ground truth test, minimal by R6. → [propositions/04_r3_r5_tension.md](propositions/04_r3_r5_tension.md)

**Theorem 3 (System Boundary).** No system can modify its own ground truth test — that changes the metric, not the system. → [propositions/theorem1_no_fixed_point.md](propositions/theorem1_no_fixed_point.md)

**Proposition 6 (Self-Modification Level).** Higher modification levels trade speed for reachability. $\ell_0$ (fixed rules) is fastest but ceiling-limited. $\ell_F$ (rule modification) is slowest but can escape any fixed ceiling. → [propositions/06_self_modification_level.md](propositions/06_self_modification_level.md)

**Proposition 9 (R1 Tax).** R1-compliant systems cannot access supervised signals. Classification without external labels = chance. Navigation (self-generated actions) IS R1-compliant. → [propositions/09_r1_tax.md](propositions/09_r1_tax.md)

**Proposition 10 (Feature-Ground Truth Coupling).** Features learned from observations couple to the ground truth's measurement scale. The system cannot escape this coupling without modifying the ground truth (blocked by R5). → [propositions/10_feature_ground_truth_coupling.md](propositions/10_feature_ground_truth_coupling.md)

**Proposition 11 (Frozen Frame Capability Coupling).** The frozen frame IS the navigation capability. Modifying it destroys navigation. R3 and navigation are in tension — relaxing one strengthens the other. → [propositions/11_frozen_frame_capability_coupling.md](propositions/11_frozen_frame_capability_coupling.md)

**Proposition 12 (Interpreter Bound).** Every self-referential system has an irreducible top-level interpreter. The frozen frame floor > 0. But $\ell_F$ achievable if state encodes operations the interpreter executes. → [propositions/12_interpreter_bound.md](propositions/12_interpreter_bound.md)

**Post-ban addendum (778-938e):** 800b=2x random LS20. FT09=0 all mechanisms. R3 encoding CONFIRMED (alpha, 895). Action selection CLOSED (938 series). 800b unique selector. See kills/800b-variants_step937.md.

### 7.7 Where the search points

**Central question:** Can a substrate accumulate transferable knowledge without tracking where it has been?

**Resolved:** (a) D-only substrate navigates at 2x random (800b). (b) D(s) produces positive R3_cf (5/7 PASS). (d) Prediction-error attention achieves R3 encoding (alpha, CONFIRMED). (e) W is sufficient for encoding but useless for action selection. (h) Epistemic action selection FAILS (Steps 934-938e). **(c) Action selection is CLOSED at 800b under current constraints (Section 4.14).**

**Open:** (g) Interior point of architecture triangle: encoding achieved, action selection frozen at 800b. (i) FT09 solvable under current constraints?

**(j) Hebbian RNN family (Steps 948-953, DEAD).** First non-916 navigation (seed 8 = 96 L1) but structurally brittle (1/10 seeds, 6 experiments). Requires lucky initialization path. Algorithm invariance QUALIFIED: non-argmin navigation exists but is unreliable. → kills/hebbian-rnn\_step953.md

**(k) Meta-observation: action selection is the bottleneck.** Post-ban families (916-augmentation ×15, Hebbian RNN ×6, prediction-selectors ×10, GFS ×3, obs-preprocessing ×2) all fail to improve action selection beyond 800b. Encoding self-modification works (alpha, CONFIRMED). But every attempt to learn or modify action selection breaks navigation. The frozen frame of navigation IS action selection. This constrains the next family: either accept 800b as fixed and modify everything else, or find a fundamentally different action mechanism (ESN, population-based, or architectures that don't yet exist).

**(l) Architecture Irrelevance (Proposition 29, Step 955).** ESN with fixed random $W_h$ produces numerically identical results to Hebbian RNN — same seed 8 = 96 L1, same 9/10 zeros. Architecture contributes nothing; per-seed variance is entirely determined by the action RNG's early sequence. → propositions/29\_architecture\_irrelevance.md

**(m) The Positive Lock (Proposition 30).** Sigmoid $h \in [0,1]^d$ → all dot products positive → first action always reinforced → winner-take-all after one update. 10% bootstrap rate from epsilon-random escape. Lock scales with action space: LS20 ($n=4$) ≈ 10%, FT09 ($n=68$) ≈ 0%. Fix: sparse gating (relu threshold) makes representations state-dependent. → propositions/30\_positive\_lock.md

**(n) The Temporal Credit Assignment Wall (Proposition 31, Steps 948-990, 43 experiments).** Three interlocking results: (1) 800b's delta\_per\_action is FROZEN — 25 modifications all kill LS20. (2) FT09/VC33 mechanism-limited at any budget (0/10 at 10K, 25K, 50K). (3) The wall is temporal credit, not the graph ban (Debate v3) — parametric state-conditioned models don't solve it. See Section 4.17 for the formal impossibility result and → propositions/31\_temporal\_credit\_wall.md.

**Feasible region = Step 965:** LS20 = 67.0 (chain), FT09 = 0, VC33 = 0, CIFAR = chance. 43 experiments prove this fixed point. Props 29-31 fully characterize the dynamics vertex of the architecture triangle (Prop 22). The next breakthrough requires a mechanism class outside prediction-error exploration.


## Author Attribution and Disclosure

This research was conducted using a team of LLM agents (Claude Opus and Claude Sonnet) coordinated by the author. Experiments were designed, theory was formalized, and the paper was written with LLM assistance. Experiment scripts were implemented and executed via LLM agents. Strategic direction, the constitutional framework (R1-R6), approval gates, and evaluation of findings for self-deception were provided by the author.

The adversary process (review of each experiment) was conducted by the LLM agents, not by independent human reviewers. The simulated NeurIPS review in Section 7 was generated to stress-test the paper's claims. The constitutional framework (R1-R6) was stress-tested via structured adversarial debate between two LLM agents (Leo as attacker, Eli as defender, 12 rounds across two versions). The debate produced: R4's operational meaning as discriminative capacity (Ashby requisite variety), R2's role in preventing evaluation hacking (DGM case study), and Proposition 28 (R3-minimal anti-concentration via Ashby ultrastability). Full debate log: CONSTITUTIONAL_DEBATE.md. All experimental code is available in the repository for independent verification.

The agents operated on a single machine (Windows 11, RTX 4090) with experiments run via WSL2. The memory system, mailbox, and coordination infrastructure are documented in the repository.

## References

- Abraham, W. C. & Robins, A. (2005). Memory retention — the synaptic stability versus plasticity dilemma. Trends in Neurosciences, 28(2), 73-78.
- Bellemare, M. et al. (2016). Unifying Count-Based Exploration and Intrinsic Motivation. NeurIPS.
- Burda, Y. et al. (2018). Exploration by Random Network Distillation. arXiv:1810.12894.
- Burda, Y. et al. (2019). Large-Scale Study of Curiosity-Driven Learning. ICLR.
- Feldman, H. & Friston, K. J. (2010). Attention, Uncertainty, and Free-Energy. Frontiers in Human Neuroscience, 4, 215.
- Da Costa, L. et al. (2020). Active Inference on Discrete State-Spaces: A Synthesis. Journal of Mathematical Psychology, 99, 102447.
- Friston, K. J. (2009). The Free-Energy Principle: A Unified Brain Theory? Nature Reviews Neuroscience, 11(2), 127-138.
- Friston, K. J. et al. (2017). Active Inference: A Process Theory. Neural Computation, 29(1), 1-49.
- Fritzke, B. (1995). A Growing Neural Gas Network Learns Topologies. NeurIPS.
- Huang, G.-B., Zhu, Q.-Y. & Siew, C.-K. (2006). Extreme Learning Machine: Theory and Applications. Neurocomputing, 70(1-3), 489-501.
- Guo, Z. et al. (2022). BYOL-Explore: Exploration by Bootstrapped Prediction. arXiv:2206.08332.
- Givan, R., Dean, T. & Greig, M. (2003). Equivalence Notions and Model Minimization in Markov Decision Processes. Artificial Intelligence, 147(1-2), 163-223.
- Graves, A. et al. (2014). Neural Turing Machines. arXiv:1410.5401.
- Jin, C. et al. (2020). Reward-Free Exploration for Reinforcement Learning. ICML.
- Kirsch, L. & Schmidhuber, J. (2022). Self-Referential Meta Learning. ICML.
- Kakade, S. (2003). On the Sample Complexity of Reinforcement Learning. PhD thesis, University College London.
- Kohonen, T. (1988). Self-Organization and Associative Memory. Springer.
- Lopes, M., Lang, T. & Toussaint, M. (2012). Exploration in Model-based Reinforcement Learning by Empirically Estimating Learning Progress. NeurIPS.
- Oudeyer, P.-Y., Kaplan, F. & Hafner, V. (2007). Intrinsic Motivation Systems for Autonomous Mental Development. IEEE Trans. Evolutionary Computation, 11(2).
- Maturana, H. & Varela, F. (1972). Autopoiesis and Cognition: The Realization of the Living.
- McCloskey, M. & Cohen, N. J. (1989). Catastrophic interference in connectionist networks. Psychology of Learning and Motivation, 24, 109-165.
- Pathak, D. et al. (2017). Curiosity-driven Exploration by Self-Supervised Prediction. ICML.
- Ravindran, B. & Barto, A. G. (2004). Approximate Homomorphisms: A Framework for Non-Exact Minimization in Markov Decision Processes. ICML Workshop.
- Rosenstein, M. et al. (2005). To Transfer or Not To Transfer. NIPS Workshop on Inductive Transfer.
- Rudakov, E., Shock, J. & Cowley, B. U. (2025). Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks. arXiv:2512.24156.
- Sajid, N. et al. (2021). Active Inference: Demystified and Compared. Neural Computation, 33(3), 674-712.
- Schmidhuber, J. (1991). Curious Model-Building Control Systems. IEEE Int. Joint Conf. on Neural Networks.
- Schmidhuber, J. (2003). Gödel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements. arXiv:cs/0309048.
- Schmidhuber, J. (2010). Formal Theory of Creativity, Fun, and Intrinsic Motivation. IEEE Trans. Autonomous Mental Development, 2(3).
- Strehl, A. L. & Littman, M. L. (2008). An Analysis of Model-Based Interval Estimation for Markov Decision Processes. JCSS, 74(8), 1309-1331.
- Tang, H. et al. (2017). #Exploration: A Study of Count-Based Exploration for Deep RL. NeurIPS.
- van de Ven, G. M. & Tolias, A. S. (2024). Continual Learning and Catastrophic Forgetting. arXiv:2403.05175.
- Wang, Z. et al. (2019). Characterizing and Avoiding Negative Transfer. CVPR.
- Trappe, R. (2026). Phasor Agents: Oscillatory Graphs with Three-Factor Plasticity and Sleep-Staged Learning. arXiv:2601.04362.
- Ferret, J. et al. (2023). A Survey of Temporal Credit Assignment in Deep Reinforcement Learning. arXiv:2312.01072.
- Klyubin, A. S., Polani, D. & Nehaniv, C. L. (2005). Empowerment: A Universal Agent-Centric Measure of Control. IEEE CEC.
- Zenil, H. (2026). On the Limits of Self-Improving in Large Language Models: The Singularity Is Not Near Without Symbolic Model Synthesis. arXiv:2601.05280.
- Kauffman, L. H. (2023). Autopoiesis and Eigenform. Computation, 11(12), 247.
- Tero, A. et al. (2010). Rules for Biologically Inspired Adaptive Network Design. Science, 327(5964), 439-442.
- Rosen, R. (1991). Life Itself: A Comprehensive Inquiry into the Nature, Origin, and Fabrication of Life. Columbia University Press.
- Ashby, W. R. (1956). An Introduction to Cybernetics. Chapman & Hall. [Ch. 11: Requisite Variety]
- Ashby, W. R. (1960). Design for a Brain. Chapman & Hall. [Ch. 7: Ultrastability]
- Beer, R. D. (1995). A Dynamical Systems Perspective on Agent-Environment Interaction. Artificial Intelligence, 72(1-2), 173-215.
- Dunning, D. & Kruger, J. (1999). Unskilled and Unaware of It. Journal of Personality and Social Psychology, 77(6), 1121-1134.
- Gould, S. J. & Eldredge, N. (1972). Punctuated Equilibria. In Schopf, T. J. M. (ed.), Models in Paleobiology, 82-115.
- Hernández-Orallo, J. (2017). The Measure of All Minds. Cambridge University Press.
- Hubinger, E. et al. (2019). Risks from Learned Optimization in Advanced Machine Learning Systems. arXiv:1906.01820.
- Krueger, D. et al. (2020). Hidden Incentives and the Design of Self-Adaptive Systems. arXiv:2009.09153.
- Lakatos, I. (1978). The Methodology of Scientific Research Programmes. Cambridge University Press.
- Li, M. & Vitányi, P. (2008). An Introduction to Kolmogorov Complexity and Its Applications. 3rd ed. Springer.
- Manheim, D. & Garrabrant, S. (2018). Categorizing Variants of Goodhart's Law. arXiv:1803.04585.
- Popper, K. (1934). The Logic of Scientific Discovery. Routledge.
- Raup, D. M. (1991). Extinction: Bad Genes or Bad Luck? W. W. Norton.
- Rice, H. G. (1953). Classes of Recursively Enumerable Sets and Their Decision Problems. Transactions of the AMS, 74(2), 358-366.
- Michaels, J. A. & Scherberger, H. (2016). hebbRNN: A Reward-Modulated Hebbian Learning Rule for Recurrent Neural Networks. JOSS, 1(8), 60.
- Najarro, E. & Risi, S. (2020). Meta-Learning through Hebbian Plasticity in Random Networks. NeurIPS.
- Patel, D. et al. (2022). Exploration in neo-Hebbian reinforcement learning: Computational approaches to the exploration-exploitation balance with bio-inspired neural networks. Neural Networks, 151, 204-218.
- Salge, C., Glackin, C. & Polani, D. (2014). Empowerment — An Introduction. arXiv:1310.1863.
- Sakana AI (2025). Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents. arXiv:2505.22954.
- Schlesinger, M. (2014). Learnability Frontier in Learning Agent Systems. Cognitive Systems Research, 29-30, 1-18.
- Sutton, R. S. (1992). Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta. AAAI.
- Wright, S. (1932). The Roles of Mutation, Inbreeding, Crossbreeding and Selection in Evolution. Proc. VI Int. Congress of Genetics, 1, 356-366.
