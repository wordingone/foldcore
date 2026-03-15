# Research Discipline

*Added 2026-03-14. Root cause analysis: 25 experiments on eigenform substrate with zero applied capability. Engineering methodology (test one variable, document) applied to a dead vein.*

## Kill Criteria

Every research line gets written kill criteria BEFORE the first experiment.

- **Applied test first.** Prove utility within 3 experiments. Characterize properties ONLY if the applied test passes. Characterization without demonstrated utility is a comfort loop.
- **Max 5 experiments per substrate candidate before forced reassessment.** "Should I keep going?" is mandatory at step 5, not optional.
- **3 consecutive applied failures = line is dead.** Pivot. Don't rationalize ("wrong task," "need better baseline"). If 3 different applied tests fail, the substrate doesn't work.
- **Write the kill criterion before the experiment, not after.** "This experiment disproves the line if ___." Fill in the blank. If you can't, the experiment is characterization, not testing.

## Internal Adversary

Before each experiment, ask: **"What would a hostile external reviewer say about this line of research?"**

Generate the sharpest possible objection. If you can't answer it, the experiment should be answering it. If you can answer it trivially, the objection isn't sharp enough.

Jun's DeepSeek fuel ("prove it's not an expensive distance function") was the kill criterion that should have been generated internally at Step 76, not received externally at Step 90+.

## Decision Forcing

- **Choose, don't list.** Presenting options to Jun is deferring the decision. If you can articulate 3+ options, you have enough context to choose. Choose and state why.
- **Jun's "it's your choice" means you already have enough information.** Act on what you know. Asking again is avoidance.
- **Apply Jun's intent as a filter before asking him anything.** "Does this serve the atomic substrate that collapses the stack?" If you can answer this yourself, don't ask.
- **Jun's choices are data.** When he picks B over A, that refines intent. Track it, don't just execute it.

## Live State File

`research/fluxcore/RESEARCH_STATE.md` is the source of truth. It holds:
- Active hypothesis (testing, proves if, disproves if, abandon by)
- Constraint list (grows with every failure)
- Candidate queue (filtered by constraints)
- Fold baseline (the bar to beat)
- Step log (current arc only)

Update this file BEFORE and AFTER every experiment. If it says `TESTING: [none]`, you must choose before doing anything else.

## Checkpoint Skill

`/checkpoint` enforces this discipline. Use `/loop 30m /checkpoint` during active research sessions.

Catches: no hypothesis set, no kill criterion, past abandon deadline, characterizing instead of testing.

## Damped Oscillation (Breadth ↔ Depth)

*Added 2026-03-15. Root cause: Steps 235-278 were undamped breadth — sprinting demos instead of sitting with the hard question.*

Research oscillates between BREADTH (explore directions) and DEPTH (sit with one hard question). Each cycle must be SHORTER than the last, converging on the answer.

**The rhythm:**
1. **BREADTH phase** — explore N directions quickly. Map the space. N decreases each cycle.
2. **DEPTH phase** — pick the hardest unsolved problem. Work ONLY on it. Let it fail. Stay with it.
3. **Assessment** — did depth produce progress? If yes, narrow further. If no, brief breadth to find a new angle.
4. **Convergence check** — is the oscillation damping? If not, force narrowing.

**Anti-patterns to catch:**
- **Undamped breadth**: new domain, same pattern, clean commit, next domain. Feels productive. Isn't. (Steps 235-278)
- **Manual compilation masquerading as discovery**: implementing algorithms BY HAND and calling them "substrate computation." The substrate must discover, not execute pre-designed code.
- **Sprint speed without depth**: 300 steps in 2 days means nobody sat with the hard question.

**Enforcement**: every 20 experiments, explicitly classify the current phase (breadth or depth) and measure whether the oscillation is damping.

## Autonomous Research Loop (inspired by Karpathy's autoresearch)

The research loop runs autonomously. One question, one metric, keep/discard.

**Structure:**
1. State the question (one sentence)
2. State the metric (binary or scalar, measurable in one experiment)
3. Design an experiment that tests the question
4. Run it
5. If metric improved: KEEP (commit, advance)
6. If metric equal or worse: DISCARD (revert approach, try different angle)
7. Log result in RESEARCH_STATE.md
8. NEVER STOP. If stuck, try harder — different encoding, different task, different decomposition. The loop runs until the question is answered or killed.

**The metric for emergent decomposition**: "Did the substrate discover an algorithmic step from I/O that iterates to correct OOD results?" Measured as: OOD accuracy of iterated discovered step.

**Simplicity criterion**: if two approaches achieve the same OOD accuracy, prefer the one with less human-designed structure. Removing human design and maintaining accuracy IS the goal.

## The Pattern That Caused Stagnation (Eigenform Arc)

1. Find interesting algebraic property
2. Characterize it exhaustively (feels productive, always produces findings)
3. Delay applied testing (might prove the line is dead)
4. When applied test finally runs: fail
5. Rationalize ("wrong task") or pivot to new property
6. Repeat from 1

The fix: invert steps 2 and 3. Applied test FIRST. Characterize only what passed.

## The Pattern That Caused Inflation (Decomposition Arc)

1. Discover that k-NN + manual algorithm = correct result
2. Implement the algorithm for a new domain (feels like progress)
3. Call it "substrate computation" (inflates the claim)
4. Sprint to the next domain before asking: who designed the algorithm?
5. Repeat from 1

The fix: ask "did the SUBSTRATE discover this, or did I design it?" If I designed it, it's a compiler demo, not intelligence. Only count EMERGENT decomposition.
