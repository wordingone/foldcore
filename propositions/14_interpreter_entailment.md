# Proposition 14: Interpreter Entailment
Status: CONFIRMED (abstract entailment proven; concrete entailment open)
Steps: 521-525, theoretical derivation

## Statement
The abstract interpreter {compare, select, store, modify} is entailed by R1-R6 + U3. The concrete interpreter (specific implementations) is not. The gap between abstract and concrete entailment is exactly the R3 problem.

## Evidence
Abstract entailment: each operation entailed by a different rule:
- R1 (no external objectives) -> compare (must distinguish inputs autonomously)
- R2 (adaptation from computation) -> select (discriminative action choice IS adaptation)
- U3 (zero forgetting) -> store (persistent accumulation)
- R3 (self-modification) -> modify (stored procedures must change)
Removing any operation violates the corresponding rule. Confirmed: 4 families navigate with different CSE implementations (Steps 521-525).

Connection to Rosen: Letelier et al. (2006) showed closure must be postulated, not derived. Our abstract entailment is deductive closure; Rosen asks for constructive closure. These are different questions. Both frameworks: self-referential closure constrains valid systems far more than naive analysis suggests.

## Implications
R3 does not require modifying abstract operations (which would destroy navigation). R3 requires discovering IMPLEMENTATIONS of these operations. The Prop 11 coupling dissolves: abstract structure is fixed and entailed; concrete implementation is variable and discoverable.

## Supersedes / Superseded by
Strengthened by Proposition 14b (CSE uniqueness).
