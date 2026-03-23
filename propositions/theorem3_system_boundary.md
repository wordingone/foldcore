# Theorem 3: The System Boundary Theorem
Status: PROVEN
Steps: Theoretical derivation

## Statement
R3 (full self-modification) and R5 (fixed ground truth) are simultaneously satisfiable if and only if the ground truth g is strictly environmental -- i.e., g not in F.

## Proof
(=>) Suppose g is a component of F. R3 requires every F component to be self-modifiable. If g is self-modifiable, system can alter evaluation criterion, violating R5. If g is protected, that protection is a non-self-modifiable F component, violating R3. Contradiction.
(<=) Suppose g is strictly environmental. R3's requirement doesn't apply to g since g not in F. R5 holds because g is fixed by environment. No contradiction.

## Evidence
All tested substrates use environmental ground truth (game death, level transitions) as R5 signal. Any substrate with internal fitness function violates R3 or R5.

## Implications
The boundary of self-modification is precisely the system-environment interface. Everything inside the system boundary must be self-modifiable (R3). Everything outside (ground truth) must be fixed (R5). The feasible region is not provably empty -- the question is empirical.

## Supersedes / Superseded by
Resolves Proposition 4 (R3-R5 tension).
