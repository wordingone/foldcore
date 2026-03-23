# Proposition 17: R3 as Self-Directed Attention
Status: CONFIRMED
Steps: 674, 720-749, 895

## Statement
For any system satisfying CSE uniqueness (Prop 14b): R3 for the comparison operation reduces to self-directed attention -- state-dependent encoding pi_s. The interpreter is fixed; the lens adapts. R3 is not "modify the program" but "modify what the program attends to."

## Evidence
1. R3 applied to compare requires compare(s1,.) != compare(s2,.) for some s1 != s2.
2. This holds iff d depends on s (l_F) or pi depends on s (l_pi), or both.
3. Hierarchy collapse: l_F = recursive l_pi for CSE interpreters.
4. Therefore R3 for comparison reduces to state-dependent pi_s.

Concrete mechanism (Step 674): TransitionTriggered674 implements self-directed attention via channel selection (I(n)), spatial resolution (k=12/k=20 based on aliasing), region weighting (budget at aliased cells). All driven by transition statistics T(s).

Step 895: alpha_d proportional to sqrt(mean_error_d) concentrates on informative dims. FT09: 10.87x concentration on puzzle tile locations. LS20: 5.73x. Confirmed across multiple seeds and games.

## Implications
R3 is the well-studied phenomenon of adaptive perception formalized within CSE. The 720 experiments that failed R3 kept encoding fixed while modifying action selection -- the WRONG target (Prop 15). Search reduces to 5 encoding dimensions (D1-D5) that can be made state-dependent via transition statistics.

## Supersedes / Superseded by
Depends on Proposition 14b (CSE uniqueness). Enables Proposition 18.
