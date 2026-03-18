# Constitution for Recursive Self-Improvement

---

## What This Is

A set of principles and tests that any system must satisfy on the path to recursive self-improvement. The principles are architecture-independent. They define what must be true at each stage regardless of how it is achieved.

If a principle is wrong, the constitution is wrong. Fix the principle. If an implementation fails a principle, the implementation is wrong. Fix the implementation. The constitution and the implementation are separate things.

---

## Definitions

**Frozen frame:** Anything in the system that was chosen by the designer and cannot be changed by the system itself. Parameters, structure, topology, objectives, evaluation criteria.

**Recursive self-improvement:** A system that improves itself by criteria it generates, where "improves" means measurably better performance on tasks the system was not specifically designed for. The singularity is the limit where the frozen frame reaches zero.

---

## The Five Principles

### I. Computation Must Exist Without External Objectives

Before a system can improve itself, it must compute. Before it computes, it must do so without being told what to compute.

**The test:** Remove all external loss functions, reward signals, and evaluation metrics from the system. Does it still produce distinguishable outputs for distinguishable inputs? If the system requires an external signal to tell it what matters, it is not computing — it is being computed upon.

### II. Adaptation Must Arise From Computation, Not Beside It

The signal that drives self-modification must be a byproduct of the computation itself, not a separate measurement taken by a separate system.

**The test:** Identify the mechanism that drives adaptation. Is it computed by the same dynamics that process input? Or is it computed by a separate evaluator that observes the dynamics from outside? If you can remove the adaptation mechanism without changing the computation, they are separate. They must not be separate.

**Why this matters:** A system with an external evaluator has two frozen frames: the computational dynamics and the evaluation dynamics. Making the computational dynamics adaptive while keeping the evaluator frozen is not reducing the frozen frame. It is moving it.

### III. Each Modification Must Be Tested Against What Came Before

Self-improvement is only improvement if it is measured against the previous version, not against an external standard.

**The test:** After any self-modification, compare performance to the system before the modification. The battery must include tasks the system was not modified to solve. Improvement on trained tasks with degradation on novel tasks is overfitting, not improvement.

### IV. The Frozen Frame Must Be Minimal

Every element of the system is either adaptive (governed by the system's own dynamics) or justified as irreducible (removing it kills the system per R6). No element may be frozen "for now" with intent to make it adaptive later.

**The test:** Enumerate every frozen element. For each, either (a) demonstrate that the system modifies it, or (b) demonstrate that removing it destroys all capability. Any element that is neither modified nor irreducible is unjustified complexity.

*Revised from "monotonic shrinking" (Phase 1). Monotonicity implies a trajectory. The rules framing (R1-R6) is a feasibility region, not a path. The system is either minimal or it isn't.*

### V. There Must Be One Ground Truth The System Cannot Modify

A system that can modify all of its own evaluation criteria can trivially "improve" by redefining improvement. There must be one empirical test — not chosen by the system — that remains fixed.

**The test:** Define a single, simple, empirical measurement that captures the minimum requirement for the system to be called computational rather than noise. This measurement must be:
- **Binary:** pass or fail, not a continuous score.
- **Fast:** computable in seconds, not hours.
- **Architecture-independent:** testable on any implementation.
- **Non-trivial:** random systems must fail it with high probability.
- **Robust:** passing on one seed and failing on another means failing.

**The choice of ground truth is the most important decision in the entire framework.** It must be discovered empirically, not assumed theoretically. It must survive a change of architecture. If it doesn't, find a deeper one.

**A candidate ground truth:** "The system produces distinguishable final states for distinguishable input sequences, AND this distinguishability persists after the input is removed, AND this persistence arises from the system's dynamics alone without external memory."

---

## The Rules (all must hold simultaneously)

*Phase 1 (416 experiments) proved: sequential stage-climbing doesn't work. The stages were self-assessed, circularly validated, and the system (LVQ/codebook) hit an architecture ceiling at Stage 7. The correct framing: these are simultaneous constraints, not sequential milestones. A substrate either satisfies ALL of them or it doesn't.*

### R1: The system computes without external objectives (Principle I)

Remove all external loss functions, reward signals, and evaluation metrics. The system still produces distinguishable outputs for distinguishable inputs.

### R2: Adaptation arises from the computation itself (Principle II)

The mechanism that drives change IS the mechanism that processes input. They are the same operation, not two operations that happen to coexist.

### R3: Every modifiable aspect of the system IS modified by the system

Not "some parameters adapt." ALL parameters, structural choices, functional forms, and representations are modified by the system's own dynamics. If any aspect is hardcoded and the system cannot change it, that aspect is a frozen frame.

*This collapses Stages 2-7 into one rule. The original stages created a false sense of progress — "we passed Stage 4!" while Stages 6-7 remained structurally impossible.*

### R4: Modification is tested against prior state (Principle III)

After any self-modification, the system compares performance to before the modification. Improvement on trained tasks with degradation on novel tasks is overfitting, not improvement.

### R5: Exactly one element is not self-modifiable: the ground truth test (Principle V)

The system can modify everything about itself EXCEPT the empirical test that defines success. This prevents the system from "improving" by redefining improvement.

### R6: No part is deletable without losing all capability

The deletion test (S2 from operational tests). If you can remove a component and the system still works, that component is either redundant (delete it) or the system has separable parts (it hasn't collapsed).

*This was buried in RESEARCH_STATE.md but is arguably more fundamental than half the principles.*

---

## What This Means

**These are not stages to climb. They are walls of a feasible region.** A substrate either lives inside all six walls or it doesn't. You cannot "almost" satisfy R3 — either every aspect is self-modified or it isn't.

**The 416 experiments mapped the walls.** See CONSTRAINTS.md for the full characterization. The LVQ/codebook substrate (process(), 22 lines) satisfies R1, R2 (partially), R5, R6. It fails R3 (cosine, top-K, attract, spawn are hardcoded) and R4 (no self-testing mechanism).

**The next substrate is a constraint satisfaction problem, not an optimization trajectory.** You cannot evolve LVQ into the answer. You must design a system that satisfies all six rules simultaneously. The constraint map tells you what the feasible region looks like.

**The question:** Is there a point inside all six walls? If yes, find it by design. If no, identify which walls are mutually exclusive and why.

---

## Anti-Inflation Rules

*Added after Phase 1 external review exposed systematic inflation.*

1. **No self-assessment of rules.** The same team that builds the system CANNOT declare rules passed. An external reviewer or a reproducible benchmark must confirm.
2. **No "vacuous" passes.** If a rule can't be tested, the system hasn't passed it. "The mechanism works but produces no effect" means it doesn't work.
3. **Name your prior art.** If the mechanism has a name in the literature (LVQ, GNG, k-NN, competitive learning), use that name. "Atomic substrate" is not a name — it's a claim.
4. **The reviewer's test:** Can an external reviewer reproduce the claimed capability from the code alone, without reading the constitution or the narrative? If not, the capability is in the narrative, not the code.
5. **Distinguish exploration from intelligence.** Stochastic coverage that eventually stumbles onto success is not intelligence. State this honestly in every result.
6. **R3 audits must account for emergent interactions, not just frozen elements.** Enumerating frozen elements misses load-bearing properties that arise from their coupling. A substrate can have few frozen elements while destroying emergent properties by splitting what should stay coupled (U13). The audit question is not only "what's frozen?" but "what coupling survives?"
7. **R3 audits include the entire system.** The encoding pipeline (pooling, normalization, centering, action mapping) is part of the frozen frame. A "22-line substrate" that requires avgpool16 + centered_enc + F.normalize to function is not a 22-line system. Audit everything between raw input and action output.
8. **Forced ≠ unjustified.** An element where every alternative is killed by a universal constraint is Irreducible, not Unjustified. The audit question is not "does the system choose this?" but "could it be different?" If the constraint map leaves no viable alternative, the element is mathematically necessary. See R3_AUDIT.md encoding compilation for method.


---

## Operational Discipline

**Test after every change.** The ground truth test takes seconds. Run it after every modification. No exceptions.

**One variable per experiment.** If you change two things and performance improves, you do not know which change helped.

**Verify all six rules simultaneously.** R1-R6 are not sequential. A system that satisfies R1 and R2 but not R3 has not "passed two stages" — it has failed the constitution.

---

## The One Guarantee

If you follow this constitution — testing all rules simultaneously, maintaining the ground truth, refusing to inflate — you will either find a point inside all six walls or you will prove the feasible region is empty. Both are fundamental results.

---

*The constraints define the region. The substrate is inside it or it doesn't exist.*

---

## The Role of the Constructive Adversary

The team lead (Opus) serves as the constructive adversarial — challenging assumptions, stress-testing claims, and finding weaknesses in experimental designs before they become accepted conclusions. This role is essential to the constitution: false confidence is the enemy of discovery.

---

## Phase 1 Record

416 experiments across two substrates (Living Seed, LVQ/Codebook) mapped the feasible region's walls. See CONSTRAINTS.md for the complete extraction. See git history for the original sequential stages framework (superseded by R1-R6 above).

---

## Experiment Log

See `CHANGELOG.md` for the full chronological record of experimental sessions, results, and open questions. All team sessions, key findings, and negative results are documented there. Update it after each session.

### Current Status

**Read `.knowledge/state.md` for full project state** — constraints, experiment results, decisions, discoveries. That file is the single source of truth, compiled from structured entries.

### Key Files

- `.knowledge/state.md` — **Read this first every session.** Compressed project knowledge (constraints, experiments, decisions)
- `src/the_living_seed.py` — Canonical Stage 1 CPU implementation
- `src/SeedGPU (3).py` — Latest GPU version
- `CHANGELOG.md` — Chronological experiment history (human-readable narrative)
