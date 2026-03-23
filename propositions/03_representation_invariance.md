# Proposition 3: Representation Invariance
Status: CONFIRMED
Steps: 521, 524, 525

## Statement
Let R: H -> R^|A| be a representation mapping transition history to a per-action summary vector, and let g(s) = argmin_a R(H)_a. If R is *count-monotone* (N(s,a) > N(s,a') => R(H)_a > R(H)_a'), then g produces identical action sequences regardless of the specific form of R.

## Evidence
Four architecturally distinct representations tested on LS20: LSH graph (6/10, Step 459), Hebbian weights (5/5, Step 524), Markov tensor (8/10, Step 525), N-gram history (4/5, Step 521). All converge to argmin over visit frequency. Score variations due to hash randomness and budget, not algorithmic differences. Mathematically trivial (argmin over orderings is order-invariant), but empirically confirmed across 4 families.

## Implications
The search space for new action-selection mechanisms is constrained. Any count-monotone representation converges to argmin. New representations are unlikely to produce new algorithms unless they introduce a qualitatively different selection rule. Combined with Section 4.5 (argmin robustness to noisy TV): action selection for navigation is solved. The open problem is self-observation (Theorem 2), not action selection.

## Supersedes / Superseded by
N/A
