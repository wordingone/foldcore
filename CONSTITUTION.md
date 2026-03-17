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

### IV. The Frozen Frame Must Shrink Monotonically

At each stage of development, at least one element of the frozen frame must become adaptive state governed by the system's own dynamics. The frozen frame must never grow.

**The test:** Enumerate every frozen element before and after a stage of development. The list must be strictly shorter after. If a new frozen element was introduced, it must be accompanied by the elimination of at least two existing frozen elements.

**Why monotonic:** Non-monotonic reduction allows indefinite postponement. "We'll make this adaptive later" is how frozen frames become permanent. Enforcing monotonicity prevents the debt from accumulating.

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

---

## Legacy: The Stages (historical, superseded by Rules)

*The following stages were used in Phase 1 (Steps 1-416). They are preserved for historical reference but are superseded by the simultaneous Rules R1-R6 above. The stages created a false sequential framework that enabled circular self-assessment.*

Stages 1-8 as originally defined are documented in the git history of this file.

---

## Operational Discipline

**Test after every change.** The ground truth test takes seconds. Run it after every modification. No exceptions.

**One variable per experiment.** If you change two things and performance improves, you do not know which change helped.

**Do not skip stages.** Each stage depends on the previous stage's mechanisms. A system that jumps from Stage 2 to Stage 5 has three layers of unvalidated assumptions. When it fails, you will not know which assumption broke.

---

## The One Guarantee

If you follow this constitution — testing at every step, shrinking the frozen frame monotonically, maintaining the ground truth — you will either reach recursive self-improvement or you will find the exact point where it becomes impossible.

---

*The destination defines the path. Each step either shrinks the frozen frame or it is not a step.*

---

## The Role of the Constructive Adversary

The team lead (Opus) serves as the constructive adversarial — challenging assumptions, stress-testing claims, and finding weaknesses in experimental designs before they become accepted conclusions. This role is essential to the constitution: false confidence is the enemy of discovery.

---

## Amendments

### Amendment 1: Vacuous Stages (Session 12)

**Modifies:** Operational Discipline — "Do not skip stages."

**Evidence:** Sessions 6–11 (seven sessions) tested Stage 3 (eta adaptation) through five independent approaches: self-referential eta (Session 6, failed — bang-bang oscillation), delta_rz second-order signal (Session 6 partial, Session 8 failed — p=0.999 training, p=0.688 novel on 10 seeds), derivative tower analysis (Session 8 — collapses at order 1, autocorrelation 0.11), delta_stability (Session 11 — improvement +0.0068, within noise floor), delta_correlation (Session 11 — anti-signal, eliminated). Constraint c011 states directly: "the system is insensitive to learning rate heterogeneity." The mechanism works — eta becomes adaptive with healthy, non-trivial, non-degenerate distributions — but produces zero measurable performance difference. The theoretical explanation: eta is a scaling parameter on a flat performance landscape; the binding constraints are structural (beta/gamma globality, equation form).

**The problem:** The original rule prevents skipping stages to avoid unvalidated assumptions. But it does not account for stages that are empirically vacuous — where the frozen element can be made adaptive but doing so produces no effect. In this case, the rule creates a deadlock: you cannot pass Stage 3 (because "beats fixed" is undefined on a flat landscape), and you cannot proceed to Stage 4 (because you have not passed Stage 3). The unvalidated assumptions the rule was designed to prevent do not exist; the assumptions have been exhaustively validated as empty.

**Amendment:** A stage may be declared **vacuously passed** if ALL of the following hold:

1. Multiple independent approaches have been tested with adequate statistical power.
2. The mechanism works — the frozen element can be made adaptive and produces non-trivial, non-degenerate behavior.
3. No measurable performance difference exists between adaptive and frozen versions, confirmed at the project's current statistical resolution.
4. A theoretical explanation exists for why the element is not a binding constraint.

A vacuously passed stage marks the element as "adaptive (vacuous)" in the frozen frame. The frozen frame still shrinks. The stage is flagged as requiring revalidation if later stages reveal a conditional dependency.

---

### Amendment 2: Forward Viability Check (Session 15)

**Modifies:** Operational Discipline — stage completion criteria.

**Evidence:** Sessions 13–16 characterized all Stage 4 structural parameters. Delta is binding (+6%) but calibration-only at boundary. Beta/gamma are architecturally impossible (c002–c007). All other parameters non-binding. The Living Seed achieves frozen frame 6/8 — but Stage 7 requires the system to represent and modify its own update rule as first-class data. The Living Seed's equation is hardcoded in Python. No self-representation mechanism exists, and none can be added within the current architecture without introducing a fundamentally different substrate.

**The problem:** Without a forward viability check, a project can spend unlimited sessions optimizing within an architecture that cannot in principle reach the later stages. The constitution's monotonic reduction rule (Principle IV) prevents detecting this — each stage appears locally passable while the endpoint is structurally unreachable.

**Amendment:** Before any stage is declared passed — vacuously or otherwise — a **forward viability check** must be completed: verify the current substrate can in principle satisfy Stage 7 (self-representation of update rule as modifiable data). If not, halt stage progression and declare an **architecture ceiling** — the frozen frame minimum achievable by the current substrate. This is a scientific result, not a failure.

An architecture ceiling triggers substrate transition: document the ceiling, extract the constraint set as design requirements for the next substrate, and begin the next architecture.

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
