# Proposition 20: Location-Dynamics Decomposition and Negative Transfer
Status: CONFIRMED (20a); PARTIALLY CONFIRMED (20b)
Steps: 776, 778v5, 780v5, 803, 809b, 855b

## Statement
Decompose s = (L(s), D(s), I(s)): location-dependent, dynamics-dependent, interpreter.
(a) Location state L(s) transfers negatively. Visit counts from training bias argmin on new tasks.
(b) Dynamics state D(s) could transfer positively if dynamics generalize across environments.

## Evidence
20(a) confirmed:
- Step 776: Visit counts OR=0.713, p<0.0001 negative transfer
- Step 803: Per-observation cycling counters -> cold=226/seed, warm=0/seed
- Step 788: Global round-robin -> 0 L1, worse than random

20(b) partially confirmed:
- 5/7 D-only substrates show positive prediction transfer (+9% to +73%)
- Forward model W trained on seeds 1-5 predicts better on unseen seeds 6-10
- Transfer robust across action mechanisms
- Navigation transfer = 0 (Proposition 21 gap)

Corollary 20.1: Negative transfer extends to ANY accumulated component coupling to action selection.

## Implications
- Graph ban is direct consequence of 20(a)
- D-only substrates are minimal viable candidate for R3_cf
- Forward model is unique candidate: accumulates dynamics (general) without coupling to action selection during training

## Supersedes / Superseded by
Motivated by Proposition 19. Extended by Proposition 21 (global-local gap).
