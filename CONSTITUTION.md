# Constitution for Recursive Self-Improvement

---

## What This Is

A set of principles and tests that any system must satisfy for recursive self-improvement. The principles are architecture-independent. They define what must be true simultaneously, regardless of how it is achieved.

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

*Clarification (revised 2026-03-21, post Hart debate):* R1 prohibits external loss functions, reward signals, and evaluation metrics — signals that encode what the system SHOULD achieve. Environmental observations (state transitions, event occurrences, interaction outcomes) are part of the input stream, not evaluative signals. Whether a specific use of environmental data introduces a frozen value judgment is governed by R3 (must be self-modifiable) and R4 (must be minimal or irreducible).

*History: The 2026-03-18 clarification ("must operate without external signal") over-restricted R1 beyond what Principle I says. The original principle prohibits three specific signal types, not all external information. A proposed reclassification (optimization targets vs consequence observations, 2026-03-21) was REJECTED after Hart debate — the original Principle I text was already correct. The fix was in the clarification, not the principle.*

**Step 432 result (critical):** Without external labels, P-MNIST classification = 9.8% (below chance). With external labels = 94.48%. The entire classification capability depends on external labels. Self-generated labels compound errors because softmax voting requires correct labels to produce correct predictions.

**Honest framing:** Classification (P-MNIST) is supervised — external labels are load-bearing, not just helpful. Navigation (LS20/FT09/VC33) IS R1-compliant — actions are self-generated by the substrate. The two benchmarks have fundamentally different R1 status.

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

## R3-Analogous Self-Examination

*Added 2026-03-21, revised after 2-round Hart debate. The search process is structurally ANALOGOUS to the substrate (Finding 12, external audit) but not identical. R3 was defined for dynamical systems with transition functions. The search is an LLM-driven research process. This section applies R3's logic by analogy, not by formal application.*

The search has persistent state (git repo, constraint map, skill definitions) and a transition function (each session reads state → processes → writes new state). Skill definitions are programs interpreted by the LLM. Modifying skills modifies behavior — but through an opaque interpreter, not transparent execution.

**Critical asymmetry (Hart, Round 2):** The LLM does not execute skill definitions like a CPU executes instructions. It INTERPRETS them through learned associations. Logically equivalent skill modifications may produce different behavior because the interpreter is opaque. ℓ_π for the search (skills rewriting skills) is prompt engineering of an opaque model, not program transformation. This is a fundamentally different operation than substrate self-modification.

| # | Element | Class | Evidence |
|---|---------|-------|----------|
| 1 | 4 modes (experiment/compress/birth/explore) | **U** | Designer-chosen. The search doesn't add or remove modes. |
| 2 | Mode ordering (exp → comp → birth → explore) | **I-prov** | Jun caught birth-before-compress (this session: eigenform birthed from stale data). Information flow dependencies constrain ordering. Alternative orderings untested. |
| 3 | Hart debate protocol | **I-prov** | Pre-Hart: 8-stage inflation survived internal review. Post-Hart: 3 inflation points caught in single session (commit eb6b0d7). Simpler alternatives (checklist) untested. |
| 4 | Constraint classification format (U/P/S/I/E) | **U** | Designer-chosen. The search doesn't modify how it classifies. |
| 5 | Constitution (R1-R6) | **M-ext** | R1 clarification modified (Hart debate, 2026-03-21). But modification was externally triggered (Jun directed compression). Mechanism internal, activation external. |
| 6 | Constraint map entries | **M** | Modified every compression iteration without per-entry external approval. U27 created and killed in one session. Genuinely self-triggered. |
| 7 | Skill definitions | **narrow U** | Content modified (compress now targets CONSTITUTION). Structure frozen (read context → act → commit). |
| 8 | Paper format and sections | **U** | Fixed academic structure. |

**Score: 1 M, 1 M-ext, 2 I-prov, 4 U.** The search modifies DATA (constraint map) but not OPERATIONS (skills, modes, classification format). The LLM (interpreter) is a categorically different frozen frame than substrate operations — it's an architectural constraint (model weights), not a design choice (cosine similarity). The search's R3 situation is WORSE than the substrate's.

**Testable prediction (eigenform analogy):** The eigenform series (Steps 620-629) showed self-observation identifies structure but cannot navigate to new territory (introspection ≠ foresight). If the analogy holds, this self-examination should identify frozen elements but NOT produce operational self-modification. Three outcomes in the next 5 sessions:
1. Leo independently modifies a skill definition BECAUSE of this audit (not because Jun directs it) → analogy is productive, eigenform prediction falsified
2. Jun reads audit → directs Leo to modify → externally triggered, same as R1 modification → proves nothing about self-modification
3. No operational modification occurs → eigenform inertness confirmed for the search

**Structural analogy (not identity):** The search and the substrate share a structural position: frozen interpreter + modifiable data. Both face the same question: can everything EXCEPT the interpreter become self-modifiable? But the interpreter scales differ by orders of magnitude (22 lines vs ~200B parameters), and the modification mechanisms are categorically different (transparent program transformation vs opaque prompt engineering). "Same shape, different substance" — useful for generating hypotheses, not for importing conclusions.

---

## Phase 1 Record

Experiments across four substrate architectures (Living Seed, ANIMA, FoldCore, TopK-Fold) mapped the feasible region's walls. See CONSTRAINTS.md for the complete extraction. See git history for the original sequential stages framework (superseded by R1-R6 above).

---

## Key Files

- `CONSTRAINTS.md` — U1-U24, I1-I9, S1-S21. The experimental record.
- `RESEARCH_STATE.md` — Full experiment log and honest assessment.
- `R3_AUDIT.md` — Frozen frame audits for all substrates.
- `INDEX.md` — File-by-file index of all experiments.
