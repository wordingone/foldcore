# Stage Viability Audit: Living Seed Against Stages 5–8

**Date:** 2026-03-02
**Method:** Reverse-engineer necessary conditions from each stage's exit criterion. Cross-reference against constraints and known architecture.

---

## Stage 5: Topology Becomes Adaptive

**Exit criterion (constitution):** Evolved topology differs from birth. Different initializations converge to similar topological features. Ground truth still passes.

**Necessary condition:** The system must have a self-generated signal that reflects the quality of the current connectivity pattern — so topology can be driven toward better configurations by internal dynamics.

**Living Seed status:** The connectivity is a fixed 1D ring, k=1 nearest neighbors. The update equation uses `x[k+1]` and `x[k-1]` — hardcoded neighbor offsets. There is no signal in the current dynamics that varies with connectivity choice. The only available signals are resp_z (local, per-cell) and the global mean field s — neither of which encodes information about whether a different neighbor structure would improve computation.

**The c003/c007 problem recurs:** A topology adaptation signal requires knowing how MI or distinguishability would change under a different connectivity graph. That is a global, counterfactual quantity — not computable from local dynamics. This is the same information-theoretic wall that blocked beta/gamma.

**Verdict:** Stage 5 will likely vacuously pass for the same structural reason Stage 3 did. The mechanism (make topology adaptive) can probably be implemented, but a self-generated signal that actually navigates the topology landscape does not exist within current dynamics. The exit criterion "different initializations converge to similar topological features" requires a selective pressure that the computation itself does not generate under Principle II.

---

## Stage 6: Functional Form Becomes Adaptive

**Exit criterion:** System discovers non-trivial functional forms not equivalent to the birth form. Ground truth still passes.

**Necessary condition:** The system must represent its own update rule in a form that can be varied, and must have a signal that evaluates functional form quality — allowing gradient or selection pressure toward better forms.

**Living Seed status:** Delta=1.0 (pure replacement) means the system already runs optimally as a memoryless processor. The current functional form includes a mixing term (delta * prior state) that the system optimized away. This suggests the functional form has overhead the system doesn't use. A simpler form — one designed for memoryless operation from the start — might perform equally or better without the mixing term.

**The interesting case:** Stage 6 is the first stage where the Living Seed's optimal parameterization gives a concrete target. If pure replacement is optimal, a functional form without the mixing term is not a discovery — it's already known. A genuinely non-trivial form would need to do something the current equation cannot. That requires a representation space of functional forms, and a signal to navigate it.

**The same wall, again:** Evaluating functional form quality requires knowing how MI changes under alternative forms. That's external measurement under Principle II unless a local signal happens to correlate — and we already know local signals don't correlate with global MI effects (c003).

**Verdict:** Stage 6 has a harder version of the same problem. It might not vacuously pass (the mechanism is harder to implement), but the exit criterion — "discovers non-trivial forms" — requires a navigation signal the architecture can't generate internally.

---

## Stage 7: Representation Becomes Adaptive

**Exit criterion:** System produces a modified update rule as data that passes ground truth and outperforms original. Modification is non-trivial.

**Necessary condition:** The system must represent its own update rule as first-class data and have a mechanism to modify that representation. This requires something like: (a) a program representation, (b) a mutation/recombination operation, (c) an evaluation step.

**Living Seed status:** The update rule is code, not data. There is no self-representation mechanism. This is not a parameter ceiling — it is a categorical gap. The Living Seed cannot satisfy Stage 7's necessary condition without a complete architectural overhaul that adds a self-model.

**Verdict:** Stage 7 is impossible for the Living Seed as currently defined. Not vacuously passable — the mechanism literally does not exist in the substrate. This is the first clean architectural impossibility (as opposed to a performance-landscape impossibility).

---

## Stage 8: Ground Truth Is the Only Frozen Element

**Necessary condition:** Everything except the ground truth test is adaptive state. This requires Stages 2–7 to have been genuinely passed — not vacuously.

**Living Seed status:** If Stage 7 is impossible, Stage 8 is unreachable. But more fundamentally: Stage 8 requires a system that can modify what it computes (Stage 6), how it represents computation (Stage 7), and potentially redesign its own evaluation (up to but not including ground truth). The Living Seed's fixed equation form, fixed topology, and no self-representation make this categorically out of reach.

**Verdict:** Impossible under current architecture.

---

## The Constitutional Flaw

[director]'s concern is well-founded. The constitution's vacuous passage mechanism (Amendment 1) was designed for stages where the frozen element is non-binding. It works correctly there. But it creates a path where a substrate-limited architecture can accumulate vacuous passages — Stage 3 (done), Stage 4 (proposed), Stage 5 (likely), Stage 6 (possible) — without ever hitting the Stage 7 wall until it's too late to course-correct.

The flaw: **the constitution has no early substrate ceiling test.** It was designed to catch methodological errors (skipping stages without validation), not architectural errors (the substrate cannot satisfy later stages regardless of methodology).

A missing principle: **Stage N viability should be assessed before Stage N-1 exit.** Before declaring Stage 3 vacuously passed, the question should be: "Does the substrate have a mechanism capable of passing Stage 7?" If the answer is no, vacuous passage is accumulating debt, not resolving it.

**Proposed constitutional addition:** Before each stage transition, conduct a forward viability check: enumerate the necessary conditions for Stage 7 and verify the substrate can in principle satisfy them. If it cannot, the result is not vacuous passage — it is an architecture ceiling finding. The constitution should call this out explicitly rather than letting it surface via accumulated vacuous passes.

---

## Summary

| Stage | Necessary Condition | Living Seed | Verdict |
|-------|-------------------|-------------|---------|
| 5 | Internal signal reflecting topology quality | No local signal encodes connectivity effects | Likely vacuous pass |
| 6 | Representation of functional form + navigation signal | MI navigation signal blocked by Principle II | Possibly vacuous or stalled |
| 7 | Self-representation of update rule as modifiable data | Categorically absent | Impossible |
| 8 | All elements adaptive except ground truth | Requires Stage 7 | Unreachable |

**The Living Seed's architectural ceiling is Stage 7.** Not because of a parameter ceiling, but because it has no self-representation mechanism. Stages 5 and 6 will consume sessions. The wall is at 7.

**The constitutional fix:** Add a forward viability check at each stage transition. Vacuous passage should require demonstrating the substrate can reach Stage 7, not just that the current frozen element is non-binding.
