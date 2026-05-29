# Formal Definition of the Search's Completed Form, and the Completion Theorem

> **STATUS: NATURALIZED THEORY DOC — 2026-05-29.**
> Naturalized from `incoming/arc-agi1-visa/` per §4 visa gate (all three tracks: PROMOTE on theory-alignment,
> Track A/C; REVISE on substrate C1). Authorized for canonical citation per scoped verdicts (VISA.md §4 checklist).
>
> **Case (a) — PROVEN.** Orthogonal surfaces identifiable; spectral Ψ + empirics ID=1.0 (analytical result,
> RESEARCH_STATE Finding 8, separate from any script output).
> **Case (b) — CITED.** Static nonlinear non-identifiability per Hyvärinen–Pajunen 1999.
> **Case (c) — CONJECTURE.** Temporal+interventional identifiability is NOT proven here. Partial empirics
> (0.44 on a toy instance) are suggestive but insufficient. Do NOT cite case (c) as a theorem. Label it
> conjecture in all downstream references.

---

## 1. The substrate

A substrate is a tuple M = (S, f, σ) operating in an environment E.

- S : internal state space (the substrate's full mutable contents — weights, structure, and any meta-state).
- X : observation space of E.  A : action space of E.
- f : S × X → S      the *sole* operator. It updates state from observation. There is no second mechanism.
- σ : S → A          action read out from current state.

Environment E = (Z, T, Φ, g):
- Z : hidden environment state.   T : Z × A → Z hidden transition (the "rule").
- Φ : Z → X  the *surface*: how hidden state is rendered into observation. Unknown to M.
- g : Z → {0,1} environmental ground truth (level/progress). Strictly external; g ∉ M.

A *task instance* is a pair (T, Φ): a rule rendered through a surface.
A *family* is a set of instances sharing T but varying Φ.

## 2. Constraints as predicates on (M, E)

Let s_t = f(s_{t-1}, x_t),  a_t = σ(s_t),  x_t = Φ(z_t),  z_t = T(z_{t-1}, a_{t-1}).

- R0  (dynamics dominate init):  ∀ s_0, s_0' :  lim_t dist(behavior(s_t), behavior(s_t')) = 0.
- R1  (no external objective):  f and σ are not functions of g. Formally g does not appear in the
      definitions of f, σ; the only signal driving f is its own prediction error on x.
- R2  (adaptation is computation):  removing the state-change part of f changes the forward map.
      ¬∃ decomposition f = (forward ∘ update) with forward computable without update.
- R3  (full self-modification):  every coordinate of S changes under f's own dynamics. wdrift(S)>0 on all coords,
      AND structure (dim S, connectivity) is itself modified by f. [R3w weights ∪ R3s structure]
- R4  (transfer / tested against prior):  performance on a NOVEL instance after self-modification on others
      exceeds fresh.  E[P(s_N, novel)] > E[P(s_0, novel)].
- R5  (fixed ground truth):  g is environmental, not in M. [Theorem 3: R3∧R5 ⟺ g ∉ F.]
- R6  (irreducibility):  ∀ component c ∈ M : removing c strictly drops capability.
- BRIDGE (I1>0):  σ(s_t) provably depends on the self-modified content of s_t (mask test moves >5%).

## 3. Where the goal lives: the identifiability condition

The goal — "generalize across anything" — means: M satisfies R0–R6+BRIDGE on EVERY family.
For a family with rule T and surfaces {Φ}, R4 (transfer) requires M to act correctly on a new Φ
having self-modified on others. Acting correctly requires recovering, from observations alone,
the part of dynamics that is INVARIANT across Φ (the rule T) separated from Φ.

DEFINITION (identifiable family). A family is *identifiable* if there exists a measurable functional
Ψ of an interaction trajectory (x_0,a_0,x_1,...,x_n) such that Ψ depends on T and is invariant to Φ:
   Ψ(traj under (T,Φ)) = Ψ(traj under (T,Φ'))  for all Φ,Φ' in the family, and
   Ψ(traj under T) ≠ Ψ(traj under T') for T≠T'.

## 4. The Completion Theorem (statement)

THEOREM (Completion). A substrate M satisfying R0–R6+BRIDGE exists for a family F if and only if
F is identifiable. Moreover:
  (a) If the surface class is the orthogonal group O(d), F is identifiable (spectral Ψ).         [PROVEN below + empirics: ID=1.0]
  (b) If the surface class is arbitrary static nonlinear maps, F is NOT identifiable from static
      statistics alone (nonlinear ICA non-identifiability, Hyvärinen-Pajunen 1999).               [CITED]
  (c) If the environment has TEMPORAL dynamics (T nontrivial) and M may intervene, F is identifiable
      via the temporal functional even through nonlinear Φ, PROVIDED Φ is bijective and the
      action-conditioned dynamics are distinguishable.                                            [CONJECTURE — partial empirics 0.44; not proven]

COMPLETION CRITERION. The search is solved-as-stated iff (c)'s proviso can be DROPPED — i.e. iff
every environment of interest is temporally-identifiable. The search is solved-conditionally iff
we exhibit M that completes on all temporally-identifiable families and PROVE the rest are
non-identifiable (hence no M exists for them — not M's failure but the environment's).

## 5. The reduction that decides it

The goal "across anything" splits, exhaustively, by surface class:
  (i)   structured/invertible-known (orthogonal, affine):  identifiable  → M exists (proven a).
  (ii)  arbitrary static nonlinear:                        NOT identifiable (b) → NO M exists, for ANYONE.
  (iii) temporal + interventional:                         identifiable under proviso (c).  [CONJECTURE]

Case (ii) is the crux. If "anything" includes (ii), the goal is IMPOSSIBLE — and not by weakness of M
but by an information-theoretic impossibility: the invariant the goal requires does not exist in the data.
A human does NOT solve (ii) either: a human facing a truly arbitrary static nonlinear scramble of a
one-shot observation, with no temporal structure and no intervention, cannot identify the rule.
Humans succeed because real environments are case (iii): they have time and permit intervention.
