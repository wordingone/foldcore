# Proposition 23b: Combinatorial Barrier for Global-EMA Selectors
Status: CONFIRMED (mathematical proof)
Steps: 800, 868b, 895f

## Statement
FT09 modeled as path graph: P_0 -> P_1 -> ... -> P_7 = GOAL. A global-EMA action selector (delta_a = alpha * Delta_a + (1-alpha) * delta_a per action, independent of position) is position-blind. At P_1, selector still prefers a_1 (globally highest delta), triggering reset. Average residence at P_i (i>0): ~1 step (prob 67/68 of immediate reset). EMA requires ~10 samples to adapt. Lower bound: P(correct 7-step sequence) <= (1/68)^6 ~ 10^{-11} per attempt. At ~1400 attempts/10K steps: expected successes ~ 10^{-8}.

## Evidence
FT09 L1=0 for all tested mechanisms (800b, ICM, RND, count, graph+argmin at 68 actions). The barrier is structural: global-EMA selectors are equivalent to memoryless automata on path graphs, which cannot navigate directed paths requiring position-dependent actions. Biological analogue: DNA polymerase proofreading solves sequential problems via LOCAL correction with a TEMPLATE. Substrate has no template.

## Implications
FT09 is structurally unsolvable by any global-EMA action selector within 10K budget. Not a mechanism weakness but a mathematical barrier. Resolution requires position-dependent action selection, which the graph ban constrains. Whether Prop 24 (epistemic action selection) dissolves this is open (Step 934+ tested: KILLED).

## Supersedes / Superseded by
Extends Proposition 23. Targeted by Proposition 24.
