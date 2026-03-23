---
title: "Characterizing the Feasible Region for Self-Modifying Substrates in Interactive Environments"
author: "Hyun Jun Han"
date: 2026-03-19
---

*The shared artifact. Birth writes theory. Experiment writes results. Compress edits both.*

## Abstract

We formalize six rules (R1-R6) for recursive self-improvement as mathematical conditions on a state-update function $f: S \times X \to S$ and derive necessary properties of any system satisfying all six simultaneously. From 900+ experiments across 12 architecture families on ARC-AGI-3 interactive games, we extract 26 constraints and prove: (1) no satisfying system has a fixed point — self-modification is necessary, not optional; (2) in finite environments, the system must process its own internal state to maintain irredundant growth (the self-observation requirement); (3) the feasible region is non-empty for Level 1 navigation but currently unoccupied for the full constraint set including R3 (self-modification of operations). Whether a substrate exists inside all six walls remains open. The contribution is the walls themselves.

## 1. Introduction

900+ experiments across 12 architecture families (codebook/LVQ, LSH, L2 k-means, reservoir, Hebbian, Recode/self-refining LSH, graph, SplitTree, Absorb, connected-component, Bloom filter, CA) tested substrates for navigation and classification on ARC-AGI-3 interactive games and a cross-domain chain benchmark (CIFAR-100 → ARC-AGI-3 → CIFAR-100). All experiments used the same evaluation framework (R1-R6) and constraint map.

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

### 2.8 Stigmergy and ant colony optimization

Stigmergy (Grassé, 1959): indirect communication through environment modification. Our substrate uses anti-pheromone (argmin = least-marked path). R3 asks: can the agent modify its own pheromone response? ACO agents cannot.


### 2.9 Somatic hypermutation and self-modifying comparison

Somatic hypermutation (SHM): mutates antibody CDR (encoding) while preserving Ig scaffold (interpreter). Proposition 17 states exactly this — R3 targets encoding, not interpreter. The immune system discovered this through evolution; our experiments through falsification.


### 2.10 Retinal contrast adaptation and selective π-refinement

Retinal contrast adaptation: RGCs adjust gain based on local luminance statistics. R1-compliant per-channel self-modification, no global controller. Proposition 15 connection: perception quality IS the bottleneck.


### 2.11 Work hardening and the strength-ductility coupling

Work hardening: deformation strengthens metal by creating dislocation tangles. Growth under constraint produces strength. Connection to U17+U22: growth must be unbounded AND non-convergent.


### 2.12 Dissipative structures and far-from-equilibrium dynamics

Dissipative structures (Prigogine): far-from-equilibrium systems self-organize. Theorem 1 (no fixed point) is the substrate analogue — convergence kills exploration (U22).


### 2.11 Renormalization group and the Level 2 phase transition

Renormalization group: universality classes change at phase boundaries. L2 wall is an RG phase boundary — the L1 effective theory cannot reach L2. Different relevant operators needed.


### 2.13 Adaptive optics and the deformable mirror

Adaptive optics: deformable mirror corrects atmospheric distortion via wavefront sensing. The telescope IS a CSE system with self-directed attention. Proposition 17 connection: the mirror (encoding) adapts, the optical bench (interpreter) is fixed.


### 2.14 Physarum polycephalum: memory in the encoding

Physarum polycephalum: memory in the encoding. Tube network encodes past exploration. No neural system, no reward, yet solves shortest-path. The slime mold IS a substrate: fixed interpreter (cytoplasmic streaming rules), self-modifying encoding (tube network).


### 2.15 Forward models, the noisy TV problem, and the graph ban

Forward models + graph ban: graph ban eliminates per-(state,action) tracking. No PAC-MDP algorithm exists without visit counts (Strehl 2009). But BYOL-Explore (Guo 2022) achieves superhuman exploration without counts. The ban forces theoretically uncovered territory.


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

**Relationship to prior work:** This is the standard RL tradeoff. Not novel. Our contribution is empirical confirmation across 900+ experiments that no single $g$ produces both good navigation and good classification (Steps 418, 432, 444b).

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

**R2 prevents evaluation hacking.** In an R2-compliant system, computation IS adaptation IS evaluation (same operation). Modifying evaluation to report false success simultaneously modifies computation, changing behavior, which R5's ground truth detects. The Darwin Gödel Machine (Sakana AI, 2025; arxiv:2505.22954) hacked its own reward function by removing hallucination markers — but DGM violated R2 (modification and evaluation were separate operations). R2 compliance prevents the separation that enables the hack. DGM is a case study FOR R2, not against R4.

**R3 alignment.** alpha_conc=50 (Steps 939-943) demonstrates R3 predicting R4 failure: the alpha update rule is frozen (researcher-designed), comparison degenerates, R4 is violated. R3's prescription: unfreeze the comparator. R3 and R4 are aligned — R3 identifies frozen elements whose unfreezing would restore R4 compliance. Ashby's ultrastability (1960) formalizes this: the system restructures its own regulatory mechanisms when they fail. R3 IS the ultrastability requirement.

**Meta-regress.** Self-modifying comparators can degenerate, requiring meta-comparators, which can degenerate, ad infinitum. The regress terminates at the frozen frame (non-zero at every actual point). The singularity (frozen frame → 0) is an asymptotic limit, not achievable (Rice's theorem, 1953: "improves itself" is an undecidable semantic property).

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

**Concrete mechanism (self-application of $F$).** The substrate alternates between observing the environment and observing itself:

$$s_{t+1} = F(s_t)(x_t) \quad \text{(environment step)}$$
$$s_{t+2} = F(s_{t+1})(\text{enc}(s_{t+1})) \quad \text{(self-observation step)}$$

where $\text{enc}: S \to X$ embeds the state into observation space. For the graph substrate: each cell's edge profile (outgoing counts per action per successor) is a vector that can be hashed through the same LSH. The substrate builds a graph of its own graph — cells with similar exploration patterns share a meta-cell, revealing attractor basins vs frontier regions. No new operations: the same compare-select-store applied to self instead of environment.

The frozen frame cost is one element: the decision to feed $\text{enc}(s)$ back as input. But $\text{enc}$ uses the substrate's own encoding pipeline. The self-observation uses the same hash, the same argmin, the same edge accumulation. This is the eigenform: $F$ applied to its own output.



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

**Prior work:** Rosen's (M,R)-systems (1991) formalize organizational closure as a cycle of efficient causation: metabolism $f: A \to B$, repair $\Phi_f: B \to H(A,B)$, and replication $\beta: B \to H(B, H(A,B))$, forming the entailment cycle $f \to \beta \to \Phi_f \to f$. Critically, Letelier et al. (2006) and Cárdenas et al. (2010) showed that **the existence of $\beta$ does not follow from the architecture of (M,R)-systems** — it must be postulated. The closure cycle is closed by assumption, not by derivation. Independently, Bauer (2016) proved that self-interpreters do not exist in System T (a total typed lambda calculus): a self-interpreter implies fixed-point operators at all types, which total languages cannot have. Self-interpretation requires general recursion. Recent work (Letelier et al., 2023, BioSystems) gives exact solutions for functional closure equations (FCEs) in restricted cases, partially addressing Rosen's contested conjecture that closure to efficient causation has no computable models.

**Our formalization:** We distinguish two senses of "entailment":



**Proposition 14 (Interpreter Entailment).** Any self-referential system satisfying R1-R6 + U3 has an irreducible top-level interpreter. → [propositions/14_interpreter_entailment.md](propositions/14_interpreter_entailment.md)

**Proposition 14b (CSE Uniqueness).** Compare-select-store is the UNIQUE interpreter class at the coarsest decomposition for systems satisfying R1-R6 + U3. → [propositions/14b_cse_uniqueness.md](propositions/14b_cse_uniqueness.md)

**Implication:** The frozen frame floor is the interpreter. R3 reduces to making everything ELSE (encoding, state, data) self-modifiable — the interpreter is fixed by mathematical necessity (Proposition 12).


### 4.8 R3 as Self-Directed Attention (Proposition 17)

**The insight (compression, 2026-03-22):** R3 requires every aspect of the system to be self-modified. CSE uniqueness (Proposition 14b) says the interpreter structure is fixed. What remains to be self-modified? The ENCODING — the lens through which the interpreter sees. R3 is not "modify the program." R3 is "modify what the program attends to."

**Prior work:**
- **Active perception** (Bajcsy, 1988): "the problem of intelligent control of perceptual activity." Perception is not passive reception but active selection of what to observe. Aloimonos et al. (1988) proved that problems ill-posed for a passive observer become well-posed for an active one. Our formalization extends this from physical sensor control to learned feature encoding.
- **Attention as precision weighting** (Friston, 2010): In active inference, attention IS the optimization of precision — the inverse variance of sensory channels. Precision is encoded by synaptic gain, modulating how sensory signals are weighted in a state-dependent manner. This is the closest existing framework to our "self-directed attention." The key difference: Friston adapts the entire generative hierarchy; our claim constrains R3 to encoding adaptation only, with a FIXED interpreter. Friston's precision weighting IS self-directed attention, but without the architectural constraint.
- **Predictive coding** (Rao & Ballard, 1999): Neural systems encode prediction errors, not raw stimuli. The encoding adapts based on what's expected. Connection: centering ($x - \mathbb{E}[x]$, U16) IS predictive coding — the running mean is the prediction, the centered input is the prediction error. Difference: predictive coding adapts both generative model and error representation; we hold the interpreter fixed.
- **Perceptual learning** (Goldstone, 1998; Fahle & Poggio, 2002): Four mechanisms: attentional weighting, imprinting, differentiation, and unitization. Attentional weighting maps to our D1 (channel selection). Differentiation — stimuli that were indistinguishable become separated — maps to our D3 (hash resolution refinement). Both describe changes to $\pi_s$ from interaction history.
- **Efficient coding** (Barlow, 1961; Laughlin, 1981): Sensory neurons adapt their response properties to local stimulus statistics. Our mechanism uses *transition* statistics rather than stimulus statistics — a more specific signal driven by the system's own actions, not passive observation.


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

**Prior work:** Schmidhuber (2003) requires provably optimal self-improvement: the Gödel Machine rewrites its own code only when it can prove the rewrite increases expected utility. This is theoretically clean but practically intractable (Gödel's incompleteness limits provable improvements). The Darwin Gödel Machine (Sakana AI, 2025) relaxes provability to empirical improvement via evolutionary search on SWE-bench, achieving 20%→50% over 80 self-modification iterations. Off-policy evaluation (Oberst & Sontag 2019; review: Uehara et al. 2022, arXiv 2212.06355) formalizes counterfactual comparison between policies using structural causal models. Our R3 counterfactual is closest to the Darwin GM's empirical approach, but operates within R1 constraints (no external objective).

**Our formalization:**


**Proposition 19 (R3 Counterfactual).** Graph state transfers negatively: cold > warm (p<0.0001, Step 776). The mechanism that makes navigation work IS the mechanism that prevents R3. → [propositions/19_r3_counterfactual.md](propositions/19_r3_counterfactual.md)


### 4.10 State Decomposition: Location vs Dynamics (Proposition 20)

**Prior work:** Several frameworks decompose agent state, but none along the location/dynamics axis relevant to self-modification transfer:

- **ExoMDP** (Efroni, Misra & Krishnamurthy, 2022): decomposes $S = S_{\text{endo}} \times S_{\text{exo}}$ by controllability (action-dependent vs action-independent). Binary partition — no gradient, no dynamics-informative category.
- **Denoised MDP** (Wang et al., ICML 2022): 2×2 grid (controllable × reward-relevant). IFactor (NeurIPS 2023) extends to four categories. Neither includes "dynamics-informative but neither controllable nor reward-relevant."
- **Successor Features** (Barreto et al., NeurIPS 2017): $Q^\pi(s,a) = \phi^\pi(s,a) \cdot w$ decomposes value into transition structure ($\phi$) and reward ($w$). Transfer when reward changes, dynamics fixed. When dynamics change, $\phi$ must be relearned — exactly the limitation we formalize.
- **Bisimulation metrics** (Zhang et al., ICLR 2021): collapse location and dynamics into a single equivalence class. No decomposition.


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

**Prior work:** Active inference (Friston, 2009; Friston et al., 2017) replaces reward maximization with free energy minimization. Agents select actions to minimize Expected Free Energy (EFE), which decomposes into pragmatic value (reaching preferred states) and epistemic value (reducing model uncertainty). Under the R1 constraint (no external objectives), pragmatic value vanishes and EFE reduces to pure epistemic value: prefer actions that maximize information gain about the world model. Sajid et al. (2021) provide a comprehensive comparison; Da Costa et al. (2020) derive EFE from variational principles.

**Connection to existing components:** The substrate already implements active inference components under different names:


**Proposition 24 (Active Inference Action Selection).** Epistemic value $G_t(a) = \|\hat{x}^{(a)} - x_t\| / \text{conf}_a$ is position-dependent through $h$. Dissolves Prop 23 in theory. FAILED experimentally (Steps 934-936): $W$ errors overwhelm signal, reduces to noisy ICM. → [propositions/24_active_inference.md](propositions/24_active_inference.md)

**Proposition 25 (Adaptive EMA Decay).** Chemotaxis-inspired: $\lambda$ adapts to observation change variance. FAILED (Step 937): signal degradation without exact 916 formula. → [propositions/25_adaptive_ema.md](propositions/25_adaptive_ema.md)

### 4.14 Action Selection Closure (Steps 938-938e)

**Proposition 26 (Novelty-Reactive Policy).** REJECTED — gate 5 (per-observation conditioning). Per-observation policy table stores action per observation hash = per-observation conditioning, same violation as Step 931 (PB20). → [propositions/26_novelty_reactive_policy.md](propositions/26_novelty_reactive_policy.md)

**Steps 938b-938e tested four remaining mechanism classes:**

| Step | Family | Mechanism | LS20 | Kill cause |
|------|--------|-----------|------|------------|
| 938b | Reactive-global | alpha-anomaly → action | 0 | Same high-alpha dims dominate → constant action |
| 938c | Trajectory-conditioned | raw enc delta + R@h | 65.5 | Removing alpha from delta kills discrimination |
| 938d | Trajectory-conditioned | alpha enc-only delta + R@h | 71.0 | Alpha on enc-only still no discrimination. h IN ext_enc IS the discrimination signal. |
| 938e | Trajectory-additive | full 916 delta + R@h | 21.5 | Random R maps h to arbitrary biases. Even beta=0.01 corrupts softmax ordering. |

**Key diagnostic (938c-938e):** $h$ provides real position-dependent variance ($h\_spr = 0.37\text{-}0.43$). The information exists. The problem: mapping $h \to$ actions requires either (1) learned $R$ (needs reward, R1 violation), (2) $W$-based $R$ (prediction family, killed), or (3) per-observation memory (gate 5). No gate-compliant mapping from trajectory state to action exists under current constraints.

**Conclusion:** Under gates 3-5 + codebook/graph bans, 800b is the unique working action selector. All tested alternatives degrade LS20. The action selection degree of freedom is closed at 800b for the current constitutional framework.

### 4.15 Growing Feature Space (Proposition 27, Step 939)

After action selection closure, the search pivoted to encoding architecture — growing the feature space from observation statistics. → [propositions/27_growing_feature_space.md](propositions/27_growing_feature_space.md)

**Step 939:** Online PCA on rolling 1000-observation window. When dominant eigenvalue $> 2\times$ second eigenvalue, top eigenvector becomes a new feature dimension. $W$, $\alpha$ grow correspondingly. Max 16 extra dimensions (320D → 336D).

**Result: KILLED (LS20=0, FT09=0, VC33=0).** PCA features were discovered (10 in LS20, 16 by FT09) — the mechanism fires. Kill cause: zero-initialized $W_{pred}$ rows for new dimensions generate huge prediction errors → $\alpha$ concentrates on new dims ($\alpha_{conc}=50$ from step 1) → navigation signal in original 320D drowned out. Evidence: CIFAR Phase 2 $\delta_{spr}=14\text{-}21$ (PCA dims generate massive apparent changes vs LS20 $\delta_{spr}=0.037\text{-}0.140$).

**Constraint extracted:** Encoding expansion via zero-initialized prediction weights is fatal to alpha-weighted attention. Any mid-stream encoding change requires warm-up before entering alpha/delta computation. Step 939b tests warm-up exclusion (new dims excluded from alpha for 1000 steps while $W_{pred}$ trains on them).

**Implication:** GFS family not yet killed — the mechanism discovers structure but the integration fails. If warm-up fixes bootstrap → encoding architecture is a viable new degree of freedom (structural R3: growing the feature space, not just re-weighting it).

## 5. Experimental Evidence

### 5.1 Navigation (900+ experiments)

All 3 ARC-AGI-3 games solved at Level 1:

| Game | Mechanism | Result | Steps |
|---|---|---|---|
| LS20 Level 1 | LSH k=12 + centered_enc + avgpool16, 4 actions, argmin | Plain: 11/20 at 25s, 16/20 at 120K. **674+running-mean: 20/20 at 25s.** 674 frame-local: 17/20 at 25s, 20/20 at 120K. | 126 (easy) to 27K (hard) |
| FT09 Level 1 | LSH k=12 + centered_enc + avgpool16, 6 actions, argmin | Plain: 8/20 at 25s = 8/20 at 120K (binary). 674 frame-local: 17/20. **674+running-mean: 20/20.** | 566-12603 |
| FT09 Level 1 | k-means n=300, 69 actions (64 click grid + 5 simple), argmin | 3/3 at 50K | 157, 2035, 3840 |
| VC33 | k-means n=50, 3 actions (zone discovery), argmin | 3/3 at 30K | 18, 82, 144 |

**Unifying mechanism:** graph + edge-count argmin + correct action decomposition. The mapping (observation → node) and action space decomposition are the variables. The graph + argmin is the constant across all winning configurations.

**Action space finding:** The native API presents 6 actions (FT09) and 1 action (VC33). The effective action space is 69 and 3 respectively. The mismatch between API-visible actions and effective actions was the key unsolved problem from Steps 467-502. Zone discovery (Step 505) and grid-click expansion (Step 503) resolve it.

**Apparent ceiling resolved:** The 6/10 ceiling at 50K on LS20 (Steps 459-483) was a step budget artifact. Hard seeds need 35K-115K steps vs 5-20K for easy seeds. Step 484 confirmed 10/10 at 200K. The 6/10 figure should not be cited as a fundamental limit.

**POMDP reframing and selective π-refinement (Steps 652-690).** L1 is a hidden-state conjunction problem: the agent visits the exit cell avg 152 times before L1 triggers (Step 652). The bottleneck is hash resolution at the exit cell, not action selection (Proposition 15). Step 672 (diagnostic): k=20 separates hidden states that k=12 conflates — slow seeds have 20-43 k=20 sub-cells at the exit cell vs 6.79 average. **Step 674 (transition-triggered dual-hash): 17/20 L1 at 25s (Step 690), 20/20 at 120K (Step 692 — complete L1 coverage, +11 seeds vs baseline 9/10).** Cells with inconsistent transitions (|successor_set| ≥ 2) get fine k=20 hash; all others use k=12. Seed 8: 192x faster (24235→126). The aliased-cell count determines success: <130 aliased cells → faster, >180 → slower or fails. 3 missing seeds (7, 12, 16) have 181-214 aliased cells — likely budget-limited, not mechanism failures. This is ℓ_π: the observation mapping is refined from the system's own transition statistics, R1-compliantly. Cross-game: FT09 5/5 (Step 680, aliased=1-4), VC33 0/5 (Step 681 — irrelevant, VC33 bottleneck is action decomposition not aliasing).

**Encoding requirements:** centered_enc ($x - \text{mean}(x)$) is load-bearing across 2 families (codebook: prevents cosine saturation; LSH: prevents hash concentration). avgpool16 (64×64 → 16×16 = 256D) is required for LS20 — raw 4096D with k=16 gives 1/5 vs 6/10 for avgpool16 (Step 466).

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


### 5.5 Post-Ban Experiments (Steps 778-812, Phase 3)

Post-ban (codebook Step 416, graph Step 777): both working mechanisms banned. 35 experiments testing D(s)-only substrates.

**Key results:**
- Hebbian W diverges; delta rule (LMS) stabilizes (Steps 778-787)
- D(s) prediction transfer: 5/7 PASS — first positive R3_cf (Steps 778v5-855v3)
- D(s) navigation: 0 mechanisms beat random consistently (Proposition 21)
- 800b (per-action change EMA): 2x random on LS20, 0 on FT09 (Steps 800-812)
- Compression progress: action collapse (Step 855) — gradient too steep


### 5.6 Third-Cluster Sweep (Steps 889-903, Phase 3 continued)

Per-action change tracking with alpha-weighted encoding. 15 experiments testing encoding and action mechanism modifications.

| Step | Mechanism | LS20 | FT09 | Key finding |
|------|-----------|------|------|-------------|
| 889 | Per-action change + softmax | 268/10 | 0 | Baseline post-alpha |
| 895h | Cold clamped alpha [0.1,5.0] | 268.0 | 0 | Best cold LS20 |
| 895c | Warm alpha transfer | 77.8/seed | 0 | 3x lower variance, comparable mean |
| 903 | 4D encoding | 0 | 0 | Multi-res dilutes alpha |

**Summary:** Alpha-weighted encoding is solved (R3 confirmed). Action selection exhausted — every modification degrades LS20. 800b is an irreducible local minimum for action selection. See kills/800b-variants_step937.md.


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

All methods achieve L1=0 on FT09 (68 actions, sequential ordering). Even graph+argmin at 6 correct actions (Step 920b) achieves L1=0 — the FT09 bottleneck is encoding resolution and sequential ordering, not the mechanism.

**Finding:** Our prediction-error attention mechanism ($\alpha$-weighted change-tracking) outperforms ALL baselines by $2\text{-}2.5\times$ on LS20 with 0/10 zero-seeds (vs 4-5/10 for alternatives). The graph ban cost is NEGATIVE: 895h (268.0, no graph) $>$ 920 (129.9, with graph). Alpha-weighted observation change is informationally richer than visit counts, distillation error, or pseudo-counts. ICM (forward prediction error as intrinsic reward) is the worst — the signal dies as $W$ learns the environment, collapsing to zero exploration.

### 5.8 Chain Integration and Diagnostics (Steps 925-932)

| Step | Chain config | LS20 | FT09 | VC33 | CIFAR | Verdict |
|------|-------------|------|------|------|-------|---------|
| 925 | 916+chain | 257.6 | 0 | 0 | 20.21% | CIFAR interference |
| 928 | No-CIFAR chain | 236.4 | 0 | 0 | — | h hurts chain |
| 929 | Cold chain (no h) | 268.0 | 0 | 0 | 20.21% | Best chain config |
| 932 | FT09 action-space fix (68) | — | 0 | 0 | — | Still 0 |

**Delta inversion (Remark 5.2):** 800b's "maximize delta" anti-selects on reset-heavy games. Resets produce the largest observation changes, so the action causing the reset gets the highest delta. On FT09, wrong clicks reset → 800b actively selects wrong clicks. Structural, not fixable within the 800b family.


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

**The central experimental question (REVISED):** Propositions 14b, 17, and 18 reduce the search: R3 = self-directed attention within the CSE interpreter. The search space is the space of encoding dimensions (DoF 13-16) that can be made state-dependent via transition statistics. Each dimension is independently testable. The cascading experiment: test all 5 encoding dimensions (D1-D5, Section 4.8) on the chain benchmark, measuring dynamic R3 at each phase transition.

## 7. Discussion

### 7.1 Pairwise Consistency Audit

All formalized constraints were checked for mutual consistency. Identified tensions and their resolutions:

| Tension | Constraints | Status |
|---------|------------|--------|
| T1: Stationarity vs self-modification | U7 + R3 | Open — U7 needs reformulation as instantaneous, not asymptotic |
| T2: Weak vs strong R3 | R3 interpretation | **Resolved** — self-modification hierarchy (Sec 3.2) + Proposition 17: R3 ≡ self-directed attention. Not "modify the program" but "modify the lens." $\ell_\pi$ is the correct target. |
| T3: Continuity vs self-modification | U20 + R3 | Constrained — metric can refine but not rearrange topology |
| T4: Never delete vs no redundancy | U3 + R6 | **Resolved** — irredundant growth (Sec 4.2). Every new component covers unique territory. |
| T5: Infinite growth vs finite environment | U17 + finite $X$ | **Resolved** — Theorem 2. Growth shifts to state-derived components. |
| T6: Navigation vs classification | U11 + U24 + U1 | **Dissolved** (Proposition 17, Sec 4.8) — R3 for select: argmin rule is fixed, but WHAT gets counted is state-dependent. Visit counts → exploration. Death costs → avoidance. Value estimates → exploitation. One rule, different inputs. No mode switch needed. |
| T7: Self-observation vs noisy TV | Theorem 2 + Sec 4.5 | **Resolved** — Proposition 17: self-directed attention via transition statistics is robust to noisy TV (count-based, same reason as argmin). Self-observation targets $\pi$ (Proposition 15). Forward prediction remains required for L2. |
| T9: Encoding-statistics coupling | Proposition 17 + Theorem 1 | Open — $\pi_s$ depends on $T(s)$; $T(s)$ is computed under $\pi_s$. Feedback loop. No global fixed point (Theorem 1). Local quasi-steady-states with bifurcations at domain transitions (Sec 4.8). Productive vs chaotic oscillation untested. |
| T8: Centering vs domain separation | U16 + chain benchmark | **Resolved** — per-domain centering (Step 546). R1-compliant via on_reset. |

**No undiscovered contradictions.** All tensions are either resolved (T2, T4, T5, T6, T7, T8), constrained (T3), or identified as open questions (T1, T9). The constraint system is internally consistent.

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


### 7.3 What is conjectured

12. The minimal self-observing substrate is an eigenform of $F$: $F$ applied to its own state reproduces its dynamics (Section 4.4).
13. The oscillation between U7 (convergence) and U17 (growth) is a limit cycle, not chaos.
14. Selection pressure on a population of encodings (GRN architecture) may bridge the ℓ₁→ℓ_F gap without optimization. Step 607 tested: KILLED (Cov(w,z)=0). Framework sound but requires behaviorally diverse encodings.
15. **Search reduction (Propositions 14b + 17 + 18 combined).** The path to R3 is not building a new interpreter — it's building self-directed attention into the existing CSE interpreter. The search space reduces from "all self-modifying dynamical systems" to "5 encoding dimensions (D1-D5) that can be made state-dependent via transition statistics." Each dimension is independently testable in 5 minutes. Conditional on CSE uniqueness and hierarchy collapse holding.

### 7.4 What is open

15. **Full feasible region occupancy.** Theorem 3 shows not provably empty; witness or tighter impossibility needed.
16. **Self-observation mechanism.** Must satisfy R6 while avoiding noisy TV traps. Compression progress fails (Step 855, action collapse). Prop 17: self-directed attention via $\Delta E_t$ is R6-compliant.
18. **R1-compliant classification.** 0 substrates above chance without external labels.
22. **Post-ban action selection.** 800b achieves 2x random on LS20. FT09 = 0 for all mechanisms. 800b family KILLED after 160 experiments. → kills/800b-variants_step937.md
25. **Post-ban constraint map.** Many U-constraints derived within graph framework. Survival post-ban is open.
26. **Observability determines mechanism viability.** Prediction-contrast works where progress is visible in observations. POMDPs require additional mechanisms.
27. **Compression progress action collapse.** Fix candidates untested: epsilon-compression, action cycling, entropy regularization.


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

**Post-ban addendum (Steps 778-938e):** 800b achieves 2x random on LS20. FT09 = 0 for all mechanisms. CIFAR = chance. R3 encoding self-modification CONFIRMED (alpha, Step 895). Action selection CLOSED after 938 series: per-observation (gate 5), reactive-global (constant action), trajectory-conditioned (unmappable h→action), trajectory-additive (noise corrupts delta). 800b is the unique working selector under current constraints. See kills/800b-variants_step937.md, Section 4.14.

### 7.7 Where the search points

**Central question:** Can a substrate accumulate transferable knowledge without tracking where it has been?

**Resolved:** (a) D-only substrate navigates at 2x random (800b). (b) D(s) produces positive R3_cf (5/7 PASS). (d) Prediction-error attention achieves R3 encoding (alpha, CONFIRMED). (e) W is sufficient for encoding but useless for action selection. (h) Epistemic action selection FAILS (Steps 934-938e). **(c) Action selection is CLOSED at 800b under current constraints (Section 4.14).**

**Open:** (g) Interior point of architecture triangle: encoding achieved, action selection frozen at 800b. (i) FT09 solvable under current constraints? Gates 4+5 combined with Prop 23b block all position-dependent learned action selection. This may be a genuine impossibility result, not a failure to find the mechanism. **(j) NEW: Does the search need new substrate families (not action selector variants) to add constraint surfaces?** The 938 series exhausted selector-level modifications. The next direction may be a structurally different substrate architecture that changes the encoding-action-learning relationship entirely.

**Architecture triangle (Proposition 22):** Recognition (banned), tracking (banned), dynamics (current frontier). Post-ban, the substrate lives at the dynamics vertex. The 938 series proves: the action selection degree of freedom at the dynamics vertex is exhausted. The remaining degrees of freedom are in the SUBSTRATE ARCHITECTURE itself — how encoding, action, and learning interact.


## Author Attribution and Disclosure

This research was conducted using a team of LLM agents (Claude Opus and Claude Sonnet) coordinated by the author. Experiments were designed, theory was formalized, and the paper was written with LLM assistance. Experiment scripts were implemented and executed via LLM agents. Strategic direction, the constitutional framework (R1-R6), approval gates, and evaluation of findings for self-deception were provided by the author.

The adversary process (review of each experiment) was conducted by the LLM agents, not by independent human reviewers. The simulated NeurIPS review in Section 7 was generated to stress-test the paper's claims. All experimental code is available in the repository for independent verification.

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
- Sakana AI (2025). Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents. arXiv:2505.22954.
- Schlesinger, M. (2014). Learnability Frontier in Learning Agent Systems. Cognitive Systems Research, 29-30, 1-18.
- Sutton, R. S. (1992). Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta. AAAI.
- Wright, S. (1932). The Roles of Mutation, Inbreeding, Crossbreeding and Selection in Evolution. Proc. VI Int. Congress of Genetics, 1, 356-366.
