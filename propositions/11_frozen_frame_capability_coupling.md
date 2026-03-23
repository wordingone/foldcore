# Proposition 11: The Frozen Frame -- Capability Coupling
Status: CONFIRMED (dissolved by Proposition 14 corollary)
Steps: R3_AUDIT.md Rounds A-B

## Statement
Let U(F) = {u in F : u is neither Modified nor Irreducible} be the unjustified frozen elements. For the navigating substrate, U(F) is load-bearing: removing any u reduces navigation to 0. R3 requires U(F) = empty. But modifying any element is lethal.

## Evidence
- P6 (top-K scoring): Remove -> 0 levels
- P7 (argmin class selection): Remove -> 0 levels
- P8 (class-restricted spawn): Remove -> cb=164 vs 20K, 0 levels
- P9 (class-restricted attract): Remove -> 0 levels
- P15 (seeding protocol): Remove -> degenerate argmin, 0 levels
Coupling C(F) = U(F) intersect L(F) is total. Not an engineering problem but a structural tension.

## Implications
R3 cannot be achieved by incremental modification. The R3-compliant substrate must achieve navigation through a DIFFERENT mechanism where load-bearing elements are Irreducible (forced by constraints) rather than Unjustified. **Dissolved by Prop 14 corollary:** the coupling is between abstract structure (fixed, entailed) and concrete implementation (variable, discoverable). R3 searches implementations, not operations.

## Supersedes / Superseded by
Dissolved by Proposition 14 corollary (interpreter entailment).
