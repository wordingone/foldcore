# Proposition 26: Novelty-Reactive Policy
Status: OPEN (theoretical, pending experimental confirmation)
Steps: 938+

## Statement
A per-observation policy table updated by successor novelty provides position-dependent action selection without per-(state,action) tracking. For an observation hash $h: X \to N$, visit counter $c: N \to \mathbb{N}$, and policy table $\pi^*: N \to A$:

$$a_t = \begin{cases} \pi^*(h(x_t)) & \text{with probability } 1-\epsilon \text{ if } \pi^*(h(x_t)) \text{ exists} \\ \text{uniform}(\mathcal{A}) & \text{otherwise} \end{cases}$$

After observing $x_{t+1}$: if $1/c(h(x_{t+1})) > \nu^*(h(x_t))$, update $\pi^*(h(x_t)) \leftarrow a_t$ and $\nu^*(h(x_t)) \leftarrow 1/c(h(x_{t+1}))$.

This stores ONE action per observation (per-observation state, ALLOWED), not a distribution over actions per observation-action pair (BANNED). It is position-dependent: different observations $\to$ different stored actions. It dissolves Prop 23b because each position independently learns its best action through observation conditioning.

## Evidence
Theoretical. Graph ban tests: (1) get_state() contains per-observation data, not per-(observation,action). (2) Cannot reconstruct visit-count argmin from single stored action. (3) Removing structure loses best-action knowledge, not action history. PASSES all three tests.

Expected FT09 performance: ~68 random trials per position $\times$ 7 positions $\times$ ~4 steps avg path length $\approx$ 1900 steps. Within 10K budget.

Prior work: Go-Explore (Ecoffet et al. 2019) uses cell archives with trajectory storage. Gershman 2024 formalizes habituation as Bayesian filtering. Stimulus-response learning (Thorndike) without external reward.

## Implications
If confirmed: dissolves the combinatorial barrier (Prop 23b) for sequential games. Combined with alpha-weighted encoding (Prop 17, confirmed), provides both position-dependent action selection AND R3 encoding self-modification. The combined system self-modifies WHAT it sees (alpha) and WHAT it does (NRP).

FAMILY_KILLS.md return condition: "position-aware without per-state memory." NRP predicts this IMPOSSIBLE result — it IS position-aware (per-observation policy) without per-state memory (no per-(state,action) counts).

## Supersedes / Superseded by
Targets Proposition 23b. Structurally different from 800b family (no per-action EMA). Combines with Proposition 17 (alpha encoding).
