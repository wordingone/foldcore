# Proposition 16: Transition-Inconsistency Refinement (Q20 Resolved)
Status: CONFIRMED
Steps: 672-690

## Statement
For a POMDP with deterministic hidden-state transitions, if observed inconsistency I_tau(n) >= 2 with sufficient visits, then cell n conflates >= 2 distinct hidden states. Transition inconsistency is a sufficient, R1-compliant signal for targeted pi-refinement.

## Evidence
Proof: If pi^{-1}(n) contains a single observation, then under deterministic T_h, each action produces a unique successor. If I_tau(n) >= 2 for the SAME action, then n must contain >= 2 hidden states. QED.

Mechanism (Step 674): Mark cells with I(n)>=2 and visits>=3 as "aliased." Fine hash (k=20) at aliased cells; coarse hash (k=12) elsewhere. Result: **17/20 L1 at 25s** (Step 690, definitive 20-seed sweep). Cross-game: FT09 5/5 (Step 680). Binary criterion outperforms ranked/capped variants (Steps 674b/c/d: all 8/10).

Exit cell naturally has highest I(n) because hidden-state conjunctions create transition variability there. Mechanism discovers task-critical cells without foreknowledge.

## Implications
R1-compliant targeted pi-refinement IS achievable via transition inconsistency. Works for L1 (bounded aliasing) but diverges for L2 (unbounded aliasing, 439+ cells).

## Supersedes / Superseded by
Resolves Proposition 15's open question (targeted pi-refinement without foreknowledge).
