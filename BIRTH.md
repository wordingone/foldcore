# Birth — Formal Derivation of the Feasible Region

*Started 2026-03-19. 505 experiments carved the mold. This document derives what's inside it.*

*The constraints are in English. English supports checking, not derivation. This document translates them to mathematics, then derives.*

---

## Primitives

Let **X** be the observation space (e.g., 64×64 palette frames).
Let **A** be the action space (game-specific; may include complex actions with parameters).
Let **S** be the state space — everything the substrate stores internally.

The substrate is a triple (f, g, F):

- **f_s: X → S** — the state update rule parameterized by current state s. Given observation x, produces next state.
- **g: S → A** — action selection. Given state, produces action.
- **F: S → (X → S)** — the meta-rule that maps states to update rules. F(s) = f_s.

The dynamics: at each timestep t,

```
s_{t+1} = F(s_t)(x_t) = f_{s_t}(x_t)
a_t = g(s_t)
x_{t+1} = env(x_t, a_t)    [external, not part of the substrate]
```

**F is the frozen frame.** It is the one function the substrate cannot modify. Everything else — s, and therefore f_s and g — is state.

---

## Formalization 1: The Structural Rules (R1-R3) and the Core Tension (U7, U17, U22)

### 1.1 Conditions

**R1 (No external objectives):**
f_s(x) depends only on s and x. No external signal (loss function, reward, label) enters the computation of s_{t+1}.

Formally: ∃ no function L external to f such that f_s(x) = h(s, x, L(s, x)). The substrate is a closed dynamical system over S given the input stream X.

*Note:* The action selection g(s) → a produces an action that affects the environment, which produces the next x. The environment IS external — but R1 governs the substrate's internal update, not the environment. The substrate sees x as raw input, not as reward.

**R2 (Adaptation = Computation):**
There is no separate optimizer. The state update f_s IS the adaptation mechanism.

Formally: let Θ(f) denote the "parameters" of f that change over time. Then Θ(f) ⊆ S. The only mechanism that modifies Θ is f itself. There is no gradient ∇L, no evolutionary selection, no external training loop. f modifies its own parameters as a side effect of processing observations.

Equivalently: the map t ↦ s_t is generated entirely by iterating f. There is no second dynamical system running alongside.

**R3 (Self-modification):**
The update rule at time t is determined by the state at time t. Different states produce different update rules.

Formally: F: S → (X → S) is not constant. There exist s₁ ≠ s₂ such that F(s₁) ≠ F(s₂) as functions on X.

*Strength of R3:* R3 does NOT require F to be injective (different states COULD produce the same rule). It requires F to be non-constant (some state differences produce different rules). The stronger version — F is injective, meaning every state difference produces a different rule — is not required by the constitution but would eliminate redundancy (see R6).

*What R3 excludes:* Any system where the update rule is the same regardless of state. For example, a k-means update where every centroid uses the same "attract toward nearest observation" rule regardless of how many centroids exist or where they are — the OPERATION is constant even though the DATA (centroid positions) changes. R3 requires the operation itself to depend on the state.

**U7 (Dominant amplification):**
Iterated application of f amplifies the dominant component of the state.

Formally: let f^n denote n-fold composition. For the linearization Df of f around a fixed point s*, the spectral radius ρ(Df) has a simple dominant eigenvalue λ₁. Then:

‖f^n(s, ·) - λ₁ⁿ v₁ v₁ᵀ f(s, ·)‖ → 0 as n → ∞

where v₁ is the dominant eigenvector. In English: the system's long-run behavior is determined by a single mode.

*Evidence:* Codebook (Step 405): recursive composition at all depths produces identical geometry = dominant eigenvector. Reservoir (Steps 438-439): Hebbian dynamics → rank-1 collapse. This is a mathematical property of ANY iterated linear (or locally linearizable) system.

**U17 (Unbounded accumulation):**
Some component of S grows without bound.

Formally: there exists a function φ: S → ℝ≥0 such that φ(s_{t+1}) ≥ φ(s_t) for all t, and lim_{t→∞} φ(s_t) = ∞.

*Evidence:* Codebook: entry count grows (spawn-on-novelty). LSH: edge count grows. k-means: edge count grows. Every successful navigation substrate has unbounded growth in SOME dimension of S.

*What φ can be:* Entry count, edge count, total mass, information content. The constraint says SOMETHING grows; it doesn't say what.

**U22 (Convergence kills exploration):**
If the state dynamics converge (s_t → s*), exploration ceases.

Formally: if ‖s_{t+1} - s_t‖ → 0, then the action sequence (a_t) becomes eventually periodic (repeats a fixed cycle).

*Evidence:* Codebook (Step 428): score convergence → random walk. TemporalPrediction (Step 437d): pred_err → 0 → weights frozen → action locked. Two architecturally distinct confirmations. Mathematically: convergent dynamics make the environment appear stationary → trivially predictable → no learning signal → no new states visited.

### 1.2 Analysis: The Core Tension

**Claim: R3 + U7 + U22 produce a tension that U17 resolves.**

*The tension:*

1. By R3, F is non-constant — different states produce different update rules.
2. By U7, iterated application of f converges to the dominant eigenvector of the linearized system.
3. By U22, convergence kills exploration — no new observations reach the system.
4. Without new observations, s stops changing (f_s(x) with fixed x and converging s produces fixed s*).
5. At the fixed point s*, the update rule is f_{s*} — a single, fixed function. R3 is formally satisfied (F is non-constant on S) but functionally dead (the system is AT s* and stays there).

*The resolution:*

6. U17 requires some φ(s) → ∞. If s converges to s*, then φ(s*) is finite. Contradiction with U17.
7. Therefore: s does NOT converge. The system cannot reach a fixed point.
8. But U7 says iterated f converges to the dominant eigenvector. How can both hold?

*The reconciliation:*

U7 applies to the LINEARIZATION of f around a fixed point. If s never reaches a fixed point (because U17 prevents it), then U7 describes local behavior (tendency toward dominant mode) but not global behavior (the system escapes before convergence completes).

The picture: f locally amplifies the dominant mode (U7), driving toward convergence. But before convergence completes, growth (U17) adds new state, which changes the update rule (R3), which changes the dominant mode, which restarts the amplification process.

**This is a limit cycle, not a fixed point.** The system oscillates between:
- Phase A: amplification toward dominant mode (U7)
- Phase B: growth disrupts the dominant mode (U17 + R3)

The period of this oscillation is not determined by the constraints. It is a **degree of freedom**.

### 1.3 Formal Statement

**Theorem (informal):** Let (f, g, F) satisfy R1, R2, R3, U7, U17, U22 simultaneously. Then the state trajectory (s_t) has no fixed point. The system exhibits perpetual non-convergence driven by the tension between U7 (convergence pressure) and U17 (growth pressure), mediated by R3 (growth changes the update rule).

**Proof sketch:**
- Suppose s_t → s*. Then φ(s_t) → φ(s*) < ∞, contradicting U17. ∎

**Corollary:** The update rule f_{s_t} never stabilizes. At every time t, the system is using a rule that will be different at some future time t' > t. Self-modification (R3) is not an optional feature that might atrophy — it is a NECESSARY consequence of the growth condition (U17).

*This resolves the question of whether R3 is "functionally dead" at convergence. It cannot be, because convergence cannot occur.*

### 1.4 Degrees of Freedom

1. **The growth function φ.** U17 says SOMETHING grows. It doesn't say what. In known substrates: entry count (codebook), edge count (graph), cell count (never, for LSH — fixed). The choice of φ determines the character of the growth.

2. **The oscillation period.** How quickly does growth (U17) disrupt the dominant mode (U7)? Fast disruption = chaotic, no time to exploit structure. Slow disruption = near-convergent, exploits structure but risks stagnation. This tradeoff is NOT determined by the constraints.

3. **The meta-rule F.** R3 says F is non-constant. U7 says F(s) has a dominant mode. But the FORM of F — how state maps to update rule — is the frozen frame. The constraints say F must exist and be non-constant, but don't specify it.

4. **The coupling between φ and F.** Growth (φ increasing) changes the update rule (via F). But how? Does adding a new node change F locally (only nearby update rules change) or globally (all update rules change)? This is not determined.

### 1.5 Tensions (unresolved)

**T1: U7 assumes linearity that R3 breaks.**
U7 is a property of LINEAR iterated systems (spectral theory). R3 makes f non-stationary (the update rule changes with state). A non-stationary system doesn't have a fixed linearization. So U7 may not apply in the standard sense. The experimental evidence (Steps 405, 438-439) confirms dominant amplification empirically, but the mathematical justification via spectral theory requires stationarity that R3 forbids.

**Status:** U7 may need to be reformulated. Instead of "converges to dominant eigenvector" (which requires stationarity), perhaps: "locally amplifies the component with largest variance" (which is an instantaneous property, not an asymptotic one). This reformulation is consistent with the evidence and doesn't require stationarity.

**T2: R3 strength is ambiguous.**
R3 says "every modifiable aspect IS modified." But the constitution defines this as "different states produce different update rules." These are not the same claim. The first says the system ACTIVELY modifies its operations. The second says the operations are RESPONSIVE to state. A system could satisfy the second (F non-constant) while violating the first (the system never visits states that exercise the non-constancy of F).

**Status:** This is a genuine ambiguity. The strong version (active modification) requires the trajectory (s_t) to visit regions of S where F varies. The weak version (responsive) only requires F to be non-constant on S, even if the trajectory stays in a region where F is locally constant. The experiments can't distinguish these — we need to decide which we mean.

---

*End of Formalization 1. Next: U3 (zero forgetting), U20 (local continuity), R6 (no deletable parts) — the growth topology.*

---

## Formalization 2: The Topology of Growth (U3, U17, U20, R6)

Formalization 1 established that the state trajectory has no fixed point — growth (U17) prevents convergence. This section asks: what STRUCTURE does the growing state space have?

### 2.1 Conditions

**U3 (Zero forgetting):**
State only grows. No element of S is ever removed.

Formally: let S_t denote the state at time t, viewed as a structured set (not just a point in a space). Then S_t ⊆ S_{t+1} for all t, where ⊆ means "every component of S_t is preserved in S_{t+1}."

*Clarification:* This does NOT mean the values in S are frozen — entries can be modified (attract updates change centroid positions). It means the STRUCTURE only grows: entries are added, never deleted; edges accumulate, never pruned; nodes persist. The constraint is on the structural skeleton, not on the values.

*Evidence:* Codebook: entries accumulate via spawn, never deleted. LSH: cells are fixed (hash buckets), edges accumulate. k-means: centroids fixed after warmup, edges accumulate. kd-tree VIOLATES this — splits destroy nodes (Step 452, killed).

**U17 (Unbounded accumulation):**
(Already formalized in 1.1. Restated for reference.)

∃ φ: S → ℝ≥0 monotonically non-decreasing along trajectories, with lim_{t→∞} φ(s_t) = ∞.

**U20 (Local continuity):**
The induced mapping π: X → N from observations to nodes preserves local structure.

Formally: there exists a metric d_X on X and a metric d_N on N such that π is Lipschitz-continuous:

d_N(π(x₁), π(x₂)) ≤ L · d_X(x₁, x₂)  for some constant L > 0

*Stronger than continuity:* Lipschitz continuity means the mapping doesn't amplify small differences. Two observations that are close in X map to nodes that are close (or equal) in N.

*Evidence:* Grid graph fails (Steps 446-447): non-continuous mapping → 0/3. LSH: sign(Hx) preserves angular locality → succeeds. k-means: nearest centroid is locally continuous by construction (Voronoi cells are connected). Reservoir fails (Step 448): hidden state history makes mapping depend on trajectory, breaking locality.

*What d_X is:* For 64×64 palette frames, d_X could be L2 distance on flattened vectors, or L2 on downsampled vectors (avgpool16). The choice of d_X is part of the encoding — and the encoding is part of the frozen frame (see R3 audit). U20 says the mapping must be continuous WITH RESPECT TO SOME d_X, but doesn't specify which d_X. The choice of metric on X is a degree of freedom.

**R6 (No deletable parts):**
Every component of S is necessary. Removing any component destroys the system's capability.

Formally: let G: S → {0, 1} be the ground truth test (R5). For every component c ∈ components(S_t), the restricted state S_t \ {c} fails G.

*What "component" means:* This depends on the structure of S. For a graph: each node and each edge is a component. For a codebook: each entry is a component. R6 says: every node is needed, every edge is needed, every entry is needed.

### 2.2 Analysis

**Proposition 1: U3 + U17 + R6 imply irredundant growth.**

*Proof:*
- U3: structure only grows (no deletion).
- U17: structure grows without bound.
- R6: every component is needed (no redundancy).
- Therefore: the system adds new components indefinitely, and every component ever added remains necessary.

This is a strong condition. It means: the system NEVER creates a component that duplicates the function of an existing component. Every new node covers a region of X that no existing node covers. Every new edge records a transition that no existing edge records.

*Implication:* The growth is an EXPLORATION of the observation space X. Each new component represents genuinely new information about X. The system is building an increasingly detailed map of X, where every detail is needed.

**Proposition 2: U20 + irredundant growth imply well-separated nodes.**

*Proof sketch:*
- U20: π is Lipschitz. Nearby observations map to the same or nearby nodes.
- Irredundant growth: no two nodes serve the same function.
- If two nodes n₁, n₂ ∈ N cover overlapping regions of X (i.e., there exists x such that d_N(n₁, π(x)) < ε and d_N(n₂, π(x)) < ε), and π is deterministic, then one of them is redundant for that region.
- Therefore: irredundancy requires that the Voronoi cells of the nodes partition X without unnecessary overlap.

*This is exactly the spawn-on-novelty mechanism:* new nodes are created when an observation is far from all existing nodes. The "far" threshold determines the resolution.

**Proposition 3: The growth rate is coupled to the observation distribution.**

From irredundant growth + local continuity: new nodes are added when the system encounters observations that are far from existing nodes. The RATE of growth depends on how often the system encounters novel observations.

This creates a feedback loop:
1. The system acts (g(s) → a).
2. The environment responds (env(x, a) → x').
3. If x' is far from existing nodes, a new node is added (growth).
4. The new node changes the update rule (R3 from Formalization 1).
5. The changed rule may produce different actions, leading to different observations.

The system's growth rate is NOT autonomous — it depends on the environment's response to the system's actions. This is an important structural property: the system cannot grow arbitrarily fast or in arbitrary directions. The environment gates the growth.

**Proposition 4: U20 constrains the topology of the node space N.**

If π: X → N is Lipschitz, then N inherits the topology of X (up to the Lipschitz constant L). Specifically:
- If X is connected (e.g., images form a connected space), then the image π(X) ⊆ N is connected.
- If X has dimension d (the manifold dimension of observations), then π(X) has dimension ≤ d.
- If X has clusters (distinct game states), then N has at least as many clusters.

For ARC-AGI-3: LS20 has ~260 reachable states, FT09 has 32, VC33 has 50. These are the cluster counts. U20 says: the node space must have AT LEAST this many clusters, and the mapping must respect their distances.

### 2.3 Degrees of Freedom

5. **The metric d_X on observation space.** U20 requires continuity but doesn't specify the metric. avgpool16 + L2 works for LS20. Raw 64×64 + L2 doesn't (Steps 388-389). The metric is part of the frozen frame — it IS part of F.

6. **The resolution of the partition.** How many nodes does the system create? Too few = aliasing (multiple distinct states merge). Too many = wasted structure. R6 says no redundancy, but doesn't say how fine the partition should be. The number of nodes is determined by the interaction between the spawn threshold and the observation distribution.

7. **The growth mechanism.** U3 says structure is added, U17 says it grows without bound. But HOW new components are added is not specified. Spawn-on-novelty is one mechanism (codebook). Implicit growth via edge accumulation is another (LSH, k-means graph). The constraints allow multiple mechanisms.

### 2.4 Tensions

**T3: U20 (local continuity) conflicts with R3 (self-modification) on the metric.**

U20 requires π to be continuous with respect to some d_X. R3 requires the update rule to depend on state. If the metric d_X is part of the update rule (as it must be — the mapping π uses a distance measure), then R3 requires d_X to change as state changes.

But if d_X changes, then U20's Lipschitz condition changes: a mapping that was continuous under the old metric may not be continuous under the new metric. Observations that were "close" may become "far" or vice versa.

This means: R3 (self-modifying metric) + U20 (continuous mapping) require that metric changes preserve the Lipschitz property. Formally: if d_X evolves as d_X^{(t)}, then π must satisfy:

d_N(π(x₁), π(x₂)) ≤ L · d_X^{(t)}(x₁, x₂)  for ALL t

This is a COMPATIBILITY condition on how the metric can evolve. Not all metric evolutions are allowed — only those that keep π continuous.

**Status:** This is a genuine constraint on R3. The system can change its metric, but not in ways that violate local continuity. Metric evolution must be "smooth" — it cannot suddenly make distant observations close or close observations distant. This rules out arbitrary metric self-modification. The metric can REFINE (increase resolution) but cannot REARRANGE (change which observations are considered similar).

**T4: Irredundant growth + finite observation space = growth must eventually slow.**

If X has finite effective dimensionality (game states are discrete), and growth is irredundant (each node covers a unique region), then the number of useful nodes is bounded by the number of distinguishable observations. Once every distinguishable observation has a node, no more growth is needed.

But U17 says growth is unbounded. If node growth is bounded, then φ must measure something other than node count — edge count, or some other accumulating quantity.

This is consistent with experimental evidence: k-means centroids plateau (286 cells, Step 493) but edges grow forever. The growth function φ can be edge count rather than node count.

**Status: T4 is DEEPER than initially stated. See Formalization 3 — edge accumulation satisfies U17 but violates R6 after saturation. This resolves into the self-observation requirement.**

---

*End of Formalization 2.*

---

## Formalization 3: The Self-Observation Requirement

*This is the central derivation. It follows from combining Formalizations 1 and 2 with a finiteness condition on the task.*

### 3.1 The Contradiction in the Current System

**Claim: After exploration saturates, the graph + argmin system violates R6 while satisfying U17. It falls outside the feasible region.**

Define *exploration saturation* as the state where the set of visited nodes N_t = N_{t+k} for all k > 0 — no new nodes are discovered regardless of how long the system runs.

*Evidence that this occurs:* Step 489 (LS20 Level 2): 1M steps, cells plateau at 259. Steps 490-492 confirm: no edge manipulation (decay, death-avoidance, death-seeking) produces new cells. Pure argmin is optimal for node discovery within this mechanism, and it saturates at 259.

After saturation:
- **U17 satisfied (formally):** Edge counts N(c, a, n) continue to increment. φ(s) = Σ N(c,a,n) → ∞. ✓
- **R6 violated (actually):** At N(c,a,n) = 10⁶, removing one count (making it 999,999) does not change argmin's behavior, action distribution, or navigation capability. The marginal count is functionally redundant. ✗

More precisely: let G be the ground truth test (R5). After saturation, the system's capability G(s) is determined by the GRAPH TOPOLOGY (which nodes exist and which transitions connect them), not by the edge COUNTS (which are all large). The counts are epiphenomenal — they satisfy U17's growth condition while providing no capability that R6 would protect.

**This means: the current system is inside the feasible region during exploration (when new nodes provide irredundant information) but exits the feasible region once exploration saturates (when growth becomes degenerate).**

### 3.2 The Derivation

**Theorem: In a finite environment, U17 + R6 + R1 require self-observation.**

*Definitions:*
- A *finite environment* has a bounded set of reachable observations: |{x : x is reachable}| < ∞.
- *Self-observation* means f_s(x) extracts new state components from s itself, not just from x.
- *Irredundant growth* means every new component c added to S is needed: G(S \ {c}) = 0.

*Proof:*

1. U17 requires ∃ φ: S → ℝ≥0 with φ(s_t) → ∞ and φ monotonically non-decreasing. Combined with R6 (irredundancy), this means: the system produces infinitely many components, each of which is needed.

2. In a finite environment, the number of irredundant components derivable from external observations alone is bounded. Once every reachable observation has been encoded (each as a unique node), further node growth from external observations is redundant. And further edge-count growth, as shown in 3.1, is also redundant (marginal counts don't affect capability).

3. Therefore, after external information is exhausted, irredundant growth must come from a source OTHER than external observations x.

4. By R1, f_s(x) depends only on s and x. No other input is available. (The environment provides x, but x is exhausted per step 2.)

5. Since x provides no new irredundant information, and f has no other inputs, the only remaining source of new irredundant information is s itself.

6. Therefore: f must extract information from s that was not explicitly represented in s before — patterns, compressions, relationships, higher-order structure. This is self-observation.  ∎

### 3.3 What Self-Observation Means

The theorem says f must extract new information from its own state. What KIND of information?

The base state s contains: nodes N, edges E, current position c. After saturation, N is fixed and E is well-estimated. What structure exists in s that hasn't been explicitly extracted?

**Temporal patterns.** The trajectory (c_0, c_1, c_2, ...) forms sequences. These sequences contain:
- Cycles (recurring patterns of nodes visited)
- Transition frequencies (which paths are common vs rare)
- Dead ends (nodes from which certain regions are unreachable)
- Bottlenecks (nodes through which most paths pass)

None of these are explicitly stored in the edge counts E(c,a,n). The edge counts record INDIVIDUAL transitions. The patterns listed above are PROPERTIES OF THE TRAJECTORY — higher-order structure.

**Graph-theoretic properties.** The graph (N, E) has structure:
- Connected components (reachable vs unreachable regions)
- Diameter (longest shortest path)
- Clustering coefficient (local density)
- Spectral properties (eigenvalues of the adjacency matrix)

These are not stored in the edge list — they are DERIVED from it.

**Meta-state.** The system's own behavior has properties:
- Exploration rate (how often new nodes are discovered — this goes to zero at saturation)
- Action distribution convergence (how stable the action choices are)
- Stuckness (how long since the last new discovery)

These are not stored in s — they are properties of the TRAJECTORY of s.

Self-observation means: constructing new state components that represent these higher-order properties. Adding them to S. Using them to modify the update rule (R3). And crucially: each such component must be irredundant (R6).

### 3.4 The Recursive Structure

Self-observation means the system applies its own processing to its own state. In the formal framework:

- Level 0: f_s(x) processes external observations x. Builds a map of the environment.
- Level 1: f_s(s) processes internal state s. Builds a map of the system's own dynamics.
- Level 2: f_s(f_s(s)) processes the Level 1 map. Builds a map of how the self-model changes.

Does this regress infinitely? U4 (minimality) says no. So there must be a FIXED POINT — a level k at which Level k+1 = Level k. The system processes its state, and the result is... the same kind of state. The mapping is an endomorphism.

**Conjecture (not proven): The minimal self-observing substrate is a fixed point of F.**

A fixed point of F means: F(s*) applied to s* yields s*. The state, when processed through the meta-rule, reproduces itself. This is a self-consistency condition — the system's dynamics, when observed by the system, confirm the system.

This connects to the codebook's eigenform property: the codebook IS its own prototypical input. But the codebook fails R3 (fixed operations) and is banned. The question is: does a non-codebook fixed point of F exist that satisfies all constraints?

**Status: Conjectured, not derived.** The theorem (3.2) is proven. The fixed-point conjecture is a hypothesis that could guide the next experiment phase.

### 3.5 Degrees of Freedom

8. **What self-observation extracts.** The system must extract higher-order structure from its state. But WHICH structure? Temporal patterns? Graph properties? Meta-state? The constraints don't specify. This is the design space.

9. **How self-observed structure modifies the update rule.** R3 requires the new structure to change f. But HOW? Does detecting a cycle cause the system to avoid the cycle? Does detecting stuckness trigger a different exploration mode? The constraints require modification but don't specify the mechanism.

10. **The fixed-point structure (if it exists).** If the system is a fixed point of F, what is the invariant? What property of the state is preserved under self-observation? This is the deepest degree of freedom — it defines the system's identity.

### 3.6 Tensions

**T5: Self-observation may violate R1.**

R1 says no external objectives. Self-observation that detects "stuckness" and triggers different exploration IS using a signal ("stuck" vs "not stuck") that functions like an internal objective. Is this an external objective in disguise?

**Resolution:** R1 prohibits EXTERNAL signals — loss functions, rewards, labels from outside the system. Self-observation processes internal state, which is part of s. The "stuckness" signal is computed from s, not provided externally. So R1 is satisfied. BUT: the DECISION to measure stuckness (rather than some other property) is part of F — the frozen frame. This is consistent: F is the irreducible frozen frame, and the choice of what to self-observe is part of F.

**T6: Self-observation risks R3 regression.**

If self-observation is a fixed operation in F ("always compute graph diameter"), then R3 is not satisfied for the self-observation mechanism — it's frozen. For full R3 compliance, the system must be able to modify WHAT it self-observes. But this requires meta-self-observation (observing the self-observation process), which triggers the infinite regress that U4 prohibits.

**Status:** This tension is real and may indicate that FULL R3 compliance is impossible. The system can satisfy R3 for base-level processing (different states → different update rules) while having a frozen self-observation mechanism. This is consistent with the "minimal frozen frame" interpretation: F includes both the base update pattern AND the self-observation pattern. Both are irreducible.

---

*End of Formalization 3. The self-observation requirement is the central result of the birth derivation. The next phase (experiment) should test: what specific self-observation mechanism produces irredundant growth after exploration saturates?*
