# Proposition 9: The R1 Tax
Status: CONFIRMED
Steps: Competition data analysis

## Statement
R1 (no external objectives) costs approximately one level (~6%) at the ARC-AGI-3 competition frontier. The cost is bounded and non-fatal.

## Evidence
- StochasticGoose (1st place): CNN+RL, 18 levels. Violates R1 (reward) and R3 (fixed architecture).
- Rudakov et al. (3rd place): Graph exploration, 17 levels. Satisfies R1. Violates R3.
- Our system: Graph+argmin+source analysis, 16 levels (3 LS20 + 6 FT09 + 7 VC33). Satisfies R1. Violates R3.
- Gap between R1-violating (18) and R1-satisfying (17): ~1 level.

The 16 levels solved via source analysis are R3's SPECIFICATION -- 16 concrete test cases -- not evidence R3 is nearly satisfied. Source analysis IS the R3 violation. Each additional level solved via analysis makes the R3 spec more precise, not the gap smaller. Rudakov et al.'s 17 levels with fixed strategy confirms: feasible region for R1 alone is large; R1+R3 is the real constraint.

## Implications
R1 is cheap. R3 is where the constraint lies. Competition data: N=3 teams, informative not definitive.

## Supersedes / Superseded by
N/A
