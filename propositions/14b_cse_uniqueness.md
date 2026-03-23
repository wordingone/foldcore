# Proposition 14b: CSE Uniqueness
Status: CONFIRMED (informal; no counterexample found)
Steps: 12 families, 900+ experiments

## Statement
Compare-select-store is the unique coarsest decomposition of F satisfying R1-R6 + U3. No fourth independent operation exists; any candidate (modify, predict, transform) reduces to CSE applied at different abstraction levels.

## Evidence
1. Individual necessity: compare (R1), store (U3), select (R2+R6) each entailed by distinct rules.
2. Joint sufficiency: graph+argmin satisfies R1,R2,U1-U3,U17,U20 (non-empty feasible region).
3. No fourth operation: modify = recursive CSE (hierarchy collapse). Predict = CSE on different data. Transform without CSE violates corresponding rules.
4. Uniqueness: any alternative either contains CSE as sub-decomposition or omits one of {compare,select,store}.
Literature survey (8 domains): Von Neumann (1966), Schmidhuber (2003), Graves NTM (2014), SUBLEQ/OISC, Rosen (1991) -- all independently arrive at structures containing CSE. None derives it axiomatically.

Every tested family (12, 900+ experiments) decomposes into CSE with different implementations.

## Implications
R3 cannot modify the STRUCTURE of the interpreter -- only its IMPLEMENTATIONS. Search space reduces from all dynamical systems to all (compare,select,store) triples satisfying the constraint table.

## Supersedes / Superseded by
Extends Proposition 14. Enables Proposition 17.
