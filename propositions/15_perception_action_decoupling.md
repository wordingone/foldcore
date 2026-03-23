# Proposition 15: Perception-Action Decoupling
Status: CONFIRMED
Steps: 652-671 (20+ experiments)

## Statement
For hidden-state conjunction problems, R(pi, g) is primarily determined by pi, not g. The L1 rate depends on whether pi resolves hidden states at the exit cell, not on the action selection strategy. No perturbation of g can reduce k_{n*} (number of aliased hidden states at exit node). Only modifications to pi can improve resolution.

## Evidence
Beyond the 6 strategies in Section 4.5, 14 additional interventions tested:
- Edge-state enrichment (4 variants, Steps 640-648): All KILL or MARGINAL
- Path conditioning (Step 649): 1.09x speedup = noise
- Outcome-conditioned selection (Step 667): KILL 0/10
- Visit-count behavior (Step 668): MARGINAL 5/10, lost 3 baseline seeds
- Gaussian variance (Step 669): MARGINAL 5/10, 145x on successes
- Alternating argmin/random (Step 670): 5/20, found 3 new but lost 5
- 1-step world model (Step 671): KILL 0/10, noisy TV
Per-visit L1 probability at exit cell ~ 1/k_{n*}. Fast seeds: small k_{n*}. Slow: large k_{n*}.

## Implications
Self-observation (Theorem 2) must target pi, not g. Step 674 (transition-inconsistency refinement) achieves this: 17/20 L1 at 25s. Convergence: Theorem 2 -> Prop 15 -> Recode demonstrates l_pi achievable.

## Supersedes / Superseded by
Resolved by Proposition 16 (transition-inconsistency refinement).
