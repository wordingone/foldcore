# Theorem 2: Self-Observation Requirement
Status: PROVEN
Steps: Theoretical derivation, confirmed by Steps 486-492, 528-529

## Statement
In a finite environment, U17 + R6 + R1 require self-observation: the system must extract structure from its own state to maintain irredundant growth after external information is exhausted.

## Proof
(1) U17 + R6 require infinitely many irredundant components. (2) In finite environment, irredundant components from external observations are bounded. (3) Edge-count growth is also redundant (N(c,a,n)=10^6 vs 10^6+1 doesn't change g). R6 violated. (4) After external info exhausted, irredundant growth must come from s itself. (5) By R1, f_s(x) depends only on s and x. Since x provides no new irredundant info, the only source is s. f must extract structure from s not explicitly stored -- temporal patterns, graph properties, meta-state. QED.

## Evidence
The current graph+argmin system exits the feasible region after exploration saturates. L2 failure (Section 5.2) is a feasibility violation, not a strategy failure. Growth rate ~2 cells/100K at 740K. Edge counts in active set are redundant.

## Implications
The system needs to process its own internal state. Self-observation must target pi (Prop 15). Curiosity literature proposes self-observation as design choice; we derive it as mathematical necessity.

## Supersedes / Superseded by
Built on Theorem 1. Motivates Propositions 13 (eigenform), 15 (target pi), 17 (self-directed attention).
