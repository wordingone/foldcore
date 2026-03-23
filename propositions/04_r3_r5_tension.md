# Proposition 4: The R3-R5 Tension
Status: RESOLVED (by Theorem 3)
Steps: 581d, 584

## Statement
R5 requires one fixed ground truth. R3 requires every aspect of computation to be self-modifiable. If evaluation criteria are fixed (R5), then at minimum the system's notion of "good" is prescribed -- violating R3.

## Evidence
Resolution: R5's ground truth is *environmental feedback* (game score, death events, level transitions) -- input the substrate reads, not computation it performs. R3 applies to the substrate's *response* to ground truth, not the signal itself. Holds iff: (a) ground truth is strictly environmental, (b) substrate's interpretation is itself self-modifiable. Step 581d (permanent soft death penalty, 4/5 vs argmin 3/5) is closest empirical approach: system self-selects WHICH edges to penalize (data-driven), but penalty VALUE and DURATION are prescribed.

## Implications
The boundary of self-modification is the system-environment interface (Theorem 3). Everything inside must be self-modifiable (R3). Everything outside must be fixed (R5). Ground truth MUST be environmental feedback, never an internal evaluation function. Any substrate with an internal fitness function violates R3 or R5.

## Supersedes / Superseded by
Formalized by Theorem 3 (System Boundary).
