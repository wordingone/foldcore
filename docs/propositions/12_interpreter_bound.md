# Proposition 12: The Interpreter Bound
Status: CONFIRMED
Steps: 620-629, theoretical derivation

## Statement
The minimal frozen frame of any self-modifying substrate is the interpreter (compare-select-store) + the ground truth test (R5). The interpreter cannot rewrite itself without infinite regress. l_F as written is impossible for computable systems, but achievable IN EFFECT via expressive l_1: if state space encodes operations the interpreter executes, behavior is indistinguishable from l_F.

## Evidence
Derivation: Suppose compare could be modified by s. The MECHANISM by which s modifies compare is itself a fixed operation (modify_compare). This is now part of the frozen frame. Recursion terminates at the outermost interpreter, irreducible by construction. Von Neumann's constructor = CSE. Kleene's fixed point = eigenform. Mossio & Longo (2009): closure IS computable -- barrier is expressiveness, not computability.

## Implications
1. The question shifts from "can the system modify its interpreter?" (no) to "is the interpreter + state space expressive enough to simulate any modification?" (expressiveness question).
2. Codebook state (R^d centroids) cannot represent update rules -> l_F structurally impossible. Program-bearing state (op-codes as edge data) CAN represent operations -> l_F possible in principle.
3. No substrate has reached l_F — consistent with pessimistic answers but has not established impossibility.

## Supersedes / Superseded by
Strengthened by Proposition 14b (CSE uniqueness).
