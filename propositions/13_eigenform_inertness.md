# Proposition 13: Eigenform Inertness
Status: CONFIRMED
Steps: 620-629

## Statement
Self-observation via F(s)(enc(s)) is necessary (Theorem 2) but not sufficient for Level 2+. The gap is informational: the graph stores the past; L2 requires predicting the future. Introspection identifies structure in visited states but cannot navigate to unvisited states.

## Evidence
- Step 620: F(s)(enc(s)) computes percentile thresholds. AVOID grows 0%->8%. L1=5/5 (matches baseline -- no improvement).
- Step 621: Adaptive observation frequency M->2000. Self-observation self-terminates. This IS the eigenform fixed point.
- Step 626: Freezing ops after 5000 steps has zero effect on L1. Eigenform inert even when active.
- Step 629: L2=0/5 even with L1-success tagging. Self-observation of existing graph cannot produce info about unvisited states.
- Step 625: Chain P3 is 7-53x SLOWER with eigenform. AVOID contaminates known paths.

Confirmed by Kauffman's eigenform theory (convergence) and Godel's incompleteness (self-referential limitations).

## Implications
R3 at L2+ requires something beyond self-observation: state-derived prediction of UNSEEN observations. Self-observation must target encoding (Prop 15/17), not action selection. The eigenform is active when targeting the right bottleneck (Prop 18: encoding, not actions).

## Supersedes / Superseded by
Scoped by Proposition 18 (eigenform reactivation via encoding attention).
