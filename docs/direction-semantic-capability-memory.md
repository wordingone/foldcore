# [Research Direction] Semantic-Capability-Memory

> Version-controlled mirror of the-search#3. Authoritative source: GitHub issue.
> Last synced: 2026-05-29.

## Question

Can a substrate accumulate a semantic-program capability memory — beyond a seeded primitive algebra — such that solved AND failed tasks generate reusable abstract operators that improve future program synthesis on structurally novel tasks?

## Constitutional Framing (R0-R6)

- **R0 (no hand-coded task primitive):** the basis must be a UNIVERSAL, task-agnostic operator set (combinator/SK-style or a small total-functional core), NOT a hand-picked geometric primitive for ARC. Hand-coding a fractal/symmetry primitive to close a specific task imports the answer = R0 violation.
- **R2 (update signal IS the computation):** abstraction discovery must be FUSED into the solve loop — operators emerge from the search trace itself — not run as a separate offline sleep/compression phase. A DreamCoder-style wake/sleep compressor is R2-violating (update beside the computation, not the computation).
- **R3/R4:** a discovered operator must change future SOLVE behavior (R3), and that change must be tested against the pre-discovery state (R4: second-exposure / structural-transfer speedup).
- **R6:** no deletable component — if removing the capability-memory leaves synthesis unchanged, it is non-load-bearing.

## Test Set — the Out-of-Closure 395 (locked, D2.1)

The seed substrate's composition-closure covers ~0.7% of eval (oracle ceiling). 395/400 eval tasks are OUT of that closure = the structural-novelty test set. Locked at depth<=5, budget=20000. `5b6cbef5` is the canonical vocabulary-gap exemplar (fractal gene absent from the substrate vocabulary).

The metric is improvement on the 395 from accumulated operators — NOT closure-internal memoization speedup.

## Experiment Sequence

### E1 — R2-fusion feasibility → REVISE

Streaming abstraction discovery fused into solve; arms OFF / PASSIVE / INTENTIONAL.

**Result:** memoization ~4.5x within-closure speedup (PASSIVE: task-1 exp=9, tasks 2-10 exp=2 via absorbed `crop+up2`). INTENTIONAL == PASSIVE on degenerate family. R2-fusion mechanism viable; intentional reprioritization UNTESTED because family was degenerate (single compound = no composition variation, so reprioritization has nothing to bite on).

Artifacts: `incoming/arc-agi1-visa/03_R4_transfer_wall/E1_fusion.py`, `E1_result.json`

### E1b — Fair test, composition-varying family → STRONGER REVISE

Rerun on family whose composition varies (8 rot tasks + 2 novel fh tasks). Isolates the ANIMA separation: intentional (params depend on accumulated state) vs reactive.

**Result:**
- v2-INTENTIONAL wins on rot tasks 4-8: compound `crop+rot` accumulates success=2 after tasks 2-3 → pops at depth-1 with sb=-2 before crop (sb=-1). **COMPOUND-level accumulation, not component-level priority.**
- v2-INTENTIONAL LOSES on novel task 9 (fh): exp=6 vs PASSIVE=4. Wrong-op compound (`crop+rot.success=7`) pops first → fails → extra search.
- Deletion confirms: removing `crop+rot` improves task 9 from exp=6 → 4. Compound actively harmful.
- Root cause: h-heuristic uniquely selects crop at depth-1 (h=0). crop.success adds zero at depth-1. At depth-2, accumulated wrong-op success hurts novel tasks — exactly backwards from the intended mechanism.

Artifacts: `incoming/arc-agi1-visa/03_R4_transfer_wall/E1b_fusion.py`, `E1b_result.json`

### E2 — Universal/combinator basis on the 395 (the genuine break)

The R0-clean test: does a universal basis + fused capability-memory improve synthesis on structurally-novel out-of-closure tasks?

**Status:** PENDING. Unblocked by E1b STRONGER REVISE.

## Pre-Registered Design-Kill

- INTENTIONAL == PASSIVE on a composition-VARYING family (E1b) → reprioritization is inert; capability-memory is R6-deletable on this axis. **E1b triggered STRONGER REVISE, not design-kill.** INTENTIONAL actively LOSES on novel tasks — the failure mode is worse than inertness.
- E2 shows no transfer on the 395 → accumulated operators do not generalize beyond closure; direction fails the structural-novelty bar.

## Status

ACTIVE. E2 pending design. Engineering owner: Eli (the-search primary). Research design: Leo.
