# Research State — Live Document
*Updated 2026-05-29. All content before this date is archived.*

---

## Active Direction: Semantic-Capability-Memory (the-search#3)

**Question:** Can a substrate accumulate a semantic-program capability memory — beyond a seeded primitive algebra — such that solved AND failed tasks generate reusable abstract operators that improve future program synthesis on structurally novel tasks?

**Constitutional framing:** R0-R6. See `docs/direction-semantic-capability-memory.md` for full spec.

**Engineering owner:** Eli. **Research design:** Leo.

---

## Constitutional Amendment (2026-05-29)

**Decision (a): RSI is second-order.** See `docs/constitution-amendment-2026-05-29-rsi-is-second-order.md` (commit b05f768c on wordingone/the-search).

RSI = recursive self-IMPROVEMENT; improvement presupposes a baseline. The prior framing conflated RSI with origin-of-cognition and banned (R1/R2) the only means of acquiring baseline capability. K1 is the empirical instance of that error made concrete: the self-modification loop on the seed basis is UNFED (no solves -> no signal), not inert.

**Consequence:** System architecture is now [frozen minimal capability core] + [self-modification meta-layer]. R6 gate: deletion of meta-layer must degrade novel-task performance.

---

## Experiment Log

### E1 — R2-fusion feasibility (DONE -> REVISE)
- Family: crop+up2 (10 tasks, single compound)
- Arms: OFF / PASSIVE / INTENTIONAL (3-arm)
- Result: PASSIVE achieves 4.5x speedup (task1 exp=9, tasks 2-10 exp=2 via absorbed crop+up2). INTENTIONAL == PASSIVE.
- Verdict: REVISE. Memoization works; reprioritization non-load-bearing. Degenerate family (no composition variation).
- Artifacts: `incoming/arc-agi1-visa/03_R4_transfer_wall/E1_fusion.py`, `E1_result.json`
- Commit: b8e949f2

### E1b — Fair test, composition-varying family (DONE -> STRONGER REVISE)
- Family: 8 rot tasks + 2 novel fh tasks (composition varies)
- Arms: v1-OFF/PASSIVE/INTENTIONAL + v2-OFF/PASSIVE/INTENTIONAL (6-arm)
- Result: v2-INTENTIONAL LOSES on novel task 9 (fh): exp=6 vs PASSIVE=4. Deletion confirms compound crop+rot was actively harmful (+2 expansions). v2 wins on rot tasks 4-8 via compound-level accumulation (not component-level priority).
- Verdict: STRONGER REVISE. INTENTIONAL does not beat PASSIVE on novel-composition tasks. Success-weighted heap priority over a fixed primitive algebra KILLED for novel-composition speedup.
- Mechanism kill K1: logged (1/3 direction-kills for the-search#3).
- Artifacts: `incoming/arc-agi1-visa/03_R4_transfer_wall/E1b_fusion.py`, `E1b_result.json`
- Commit: b8e949f2

### K1 — Seed-basis novelty kill probe (DONE -> K1 FIRES)
- Task set: 395 locked out-of-closure tasks (D2_1_final_out.json, depth<=5, budget=500 probe)
- Regimes: (i) h-only, (ii) primitive-success-weighted
- Results:
  - Depth-1 unique h-argmin: 116/395 (29.4%). Tied: 279/395 (70.6%).
  - pops_differ under max-hypothetical scenario: 0/395 (uniform success=10 -- see note below)
  - Solve rate h-only: 5/395 (5 are false-positives from single-pair probe; D2 required all pairs)
  - Solve rate prim-success: 5/395. Expansion delta: 0.
  - Deletion: no-op (accumulated prim-success from 395 = 0; no solve signal generated)
- Verdict: K1 FIRES. Capability-memory cannot improve NOVEL synthesis under seed basis as a structural fact. The 395 require primitives absent from the 12-op basis (vocabulary gap, not search-order gap). Mechanism kill K1 is structural.
- Note: pops_differ=0 is partly an artifact of uniform-success hot scenario. Differential success WOULD change depth-1 pop on 70.6% of tied tasks -- but it never accumulates because solve rate=0 on out-of-closure tasks. The structural kill stands.
- Artifacts: `incoming/arc-agi1-visa/03_R4_transfer_wall/K1_probe.py`, `K1_result.json`
- Commit: bac727fd

---

### E2.1 — Combinatorial meta-layer gate (DONE -> R6_FIRES; IN-COVERAGE ONLY)
- R4 classification: SELECTION-class experiment. Per Leg 2, out-of-coverage ceiling = +0.000. In-coverage gains tested here.
- Architecture: Option-2 frozen core (logistic prior, ~1MB, biased 70% toward ACCUM_FIRST/ACCUM_SECOND) + LGG depth-2 meta-layer, R2-fused.
- Coverage split: IN-COVERAGE only (all synthetic tasks expressible by the 12-op basis). Out-of-coverage measurement pending — needs E2.1b (see below).
- Accumulation family (9 programs): ACCUM_FIRST={crop,fh,tr} × ACCUM_SECOND={rot,up2,mir_h}. 90 tasks.
- Held-out family (6 programs): HELD_FIRST={fv,dup_h} × ACCUM_SECOND. 30 tasks.
- Results (biased core, in-coverage):
  - CORE_ONLY held-out: 30/30, mean_exp=7.63; CORE_META: 30/30, mean_exp=7.63; delta=0.00
- Uniform core control (in-coverage):
  - CORE_ONLY: mean_exp=7.57; CORE_META: mean_exp=7.70 (WORSE, delta=-0.13)
  - Root cause: LGG accumulated fh=10 from alternative solutions (not ground-truth program). Meta boosts wrong op at depth-2 for tasks needing rot.
- Structural finding: LGG accumulates from whatever SOLVED the task, not the ground-truth program. Underdetermined task families → noisy meta signal → misdirects novel-composition search.
- Verdict: R6_FIRES IN-COVERAGE. Per R4 Leg 2, out-of-coverage = +0.000 is PREDICTED (selection cannot expand coverage). E2.1b will confirm.
- Artifacts: `incoming/arc-agi1-visa/03_R4_transfer_wall/E2_1_experiment.py`, `E2_1_result.json`
- Commit: dad816fe

---

### E2.1b — Out-of-coverage prediction (DONE -> R4_LEG2_CONFIRMED)
- R4 classification: SELECTION-class, out-of-coverage split. Per Leg 2, prediction = +0.000.
- Result: in-coverage delta=+0.000, out-of-coverage (100/395 sampled) delta=+0.000, solve_delta=0.
- Verdict: R4 Leg 2 confirmed from capability-memory angle. LGG selection = +0.000 on both sides of the coverage line.
- Artifacts: `E2_1b_coverage_split.py`, `E2_1b_result.json`. Commit: 2b1b3d88.

---

## E2.2(b) — Trace-conditional accumulation, TERMINAL selection-axis test (PRE-REGISTERED 2026-05-29)

**Purpose (Leo #11525):** One unmeasured cell in the signal-cleanliness axis:
- E2.1 uniform core + DIRTY signal: delta=-0.13 (meta harmful — noise from alternative solutions)
- E2.2(b): **uniform core + CLEAN signal** — trace-conditional accumulation; meta is the SOLE bias source

If meta still non-load-bearing with clean signal → selection axis definitively dead (K3 = 3/3).

**Design:**
- Core: UNIFORM prior (1/12 equiprobable at d0 and d1). Meta is the only bias source.
- Accumulation: TRACE-CONDITIONAL — only accumulate from sequences that MATCH the ground-truth generating program. If solver finds an alternative valid sequence, do NOT accumulate.
- Task family: same split as E2.1 (ACCUM_PROGRAMS × N_ACCUM_PER, HELD_PROGRAMS × N_HELD_PER).

**Pre-registered kill (K3):**
CORE_META deletion-delta ≤ 0 with trace-conditional clean signal → K3 (3/3) → selection axis DEFINITIVELY dead → hard pivot to E2.2' (coverage-expansion core).

**Pre-registered positive:**
Deletion-delta > 0 → selection works with clean signal → carry clean-signal meta-layer into E2.2' core design.

**Hard-pivot trigger:** K3 fires → next = E2.2' (expressibility-expanding core = C6 operationalized).

---

### E2.2(b) — Trace-conditional accumulation result (DONE -> K3_FIRES)
- Design: UNIFORM core (1/12) + trace-conditional accumulation (ground-truth match only).
- Accumulation: 26/89 traces accepted (ground-truth match); 63 rejected (alternative solutions).
  - depth2_success: {mir_h:19, up2:6, rot:1}. fh=0 (correctly rejected — E2.1 noise confirmed).
- Held-out: CORE_ONLY mean_exp=7.57, CORE_META mean_exp=7.57. delta=0.
- R6 deletion delta: +0.000. Meta non-load-bearing even with clean signal.
- Root cause: LGG depth-2 accumulation learns the MARGINAL P(d2) from accum stream. Novel synthesis
  requires the CONDITIONAL P(d2|d1, input). Marginal boost is informationally insufficient —
  it cannot know which d2 op a specific novel task needs. This is structural, not noise-related.
- Verdict: K3 FIRES (3/3). Selection axis DEFINITIVELY dead. Hard pivot triggered.
- Artifacts: `E2_2b_trace_conditional.py`, `E2_2b_result.json`. Commit: 3b7c13c2.

---

## HARD PIVOT — E2.2' (Coverage-Expansion Core)

Selection axis exhausted. Three kills confirm the R4 decomposition from the capability-memory side:
- K1: selection signal loop UNFED out-of-coverage (vocabulary gap, structural)
- K2: selection redundant in-coverage (core pre-bias makes meta noise)
- K3: selection still zero even with clean signal (marginal != conditional, structural)

### E2.2' Architecture — Leo directive #11531 (2026-05-29)

**Core: Option-3 — frozen local LLM at :9876 as generative program-proposer.**
- Decisive reason (from K3): K3 killed the MARGINAL mechanism P(d2); LLM proposes CONDITIONALLY (P(program | I/O examples)). Option-3 is the only option that doesn't repeat the marginal mistake.
- C6 operationalization: an LLM IS a learned generative program-prior at scale.
- Avoids Option-1's synthetic-training-data pre-encoding risk (the E1/E2.1 failure mode).
- Engineering: llama-server :9876 (Qwen3.6-27B-Q4_K_M.gguf) + SUBSTRATE.py as execution layer.

**Staged (forward-disciplined):**

**Stage 0 — CORE_ONLY coverage probe (PRE-REGISTERED 2026-05-29, BEFORE RUNNING)**
- Frozen LLM proposes K candidate programs per task, conditioned on I/O examples, emitted as DSL sequences.
- Parse + execute via SUBSTRATE.py. Count solves.
- Test set: sampled subset of the 395 locked out-of-closure tasks (within 5-min runtime cap).
- Baselines: (a) 0.7% oracle ceiling (Leg 3); (b) seed-enumeration solve-rate on same subset.
- PRE-REGISTERED KILL: frozen LLM-alone does NOT expand coverage above seed/oracle on the sampled subset → coverage-expansion via this core is FALSIFIED. Report the negative; stop.
- PRE-REGISTERED FORWARD: LLM solves ANY task that seed-enumeration (SUBSTRATE.py, budget=20000) cannot → coverage-expansion confirmed → proceed to Stage 1.
- R0 refinement (Leo #11531): core-minimality was a proxy for R6-honesty; the staged CORE_ONLY baseline measures R6-honesty directly. LLM is maximal in params but used as a FIXED substrate proposer; the meta-layer is minimal + isolable.

**Stage 1 — meta-layer + R6 kill (GATED on Stage 0 FORWARD)**
- Add abstraction-accumulation library: LGG-with-holes from system's OWN solved traces, fed back to proposer as macro-ops/exemplars on subsequent novel tasks (DreamCoder wake-sleep loop = C6).
- R6 kill: CORE_META vs CORE_ONLY on accumulation-held-out novel tasks.
- Load-bearing (CORE_META > CORE_ONLY) = RSI signal — genuine second-exposure improvement, R4 finally firing.
- Decorative (delta ≈ 0) = (a) collapsed to 'just an agent'; reported honestly.

**MEM-HEAVY serialization:** LLM at :9876 is a MEM-HEAVY op. Broadcast MEM-HEAVY START/END. One memory-heavy op at a time fleet-wide.

---

### E2.2' Stage-0 — CORE_ONLY coverage probe (DONE -> KILL)

**Result: KILL. 0 solves across 36 tasks tested, 183 proposals.**

Two runs:
- Run 1: 30 tasks × K=5, max_tokens=64. Parse failures: 150/150 (thinking overflow — model generates `<think>...</think>` reasoning, 64 tokens consumed entirely by thinking, response never produced).
- Run 2: 10 tasks × K=3, max_tokens=2048. Parse failures: 17/17 (thinking overflow + 30s API timeout — model thinking takes 20-60s per call, exceeds timeout). 6/10 tasks reached before time cap.

**Pre-registered kill fires:** 0 LLM-only solves / 36 tasks tested.

**Technical failure mode:** Qwen3.6-27B always enters thinking mode. With max_tokens=64: thinking block fills entire budget. With max_tokens=2048: thinking takes 20-60s per call, exceeds 30s request timeout. Neither setting allowed the model to complete and output a program.

**Structural analysis (the real kill reason, independent of parsing):**
1. The LLM was constrained to the 12-op DSL vocabulary in the system prompt. Any programs it proposes are combinations of these 12 ops — the same vocabulary as seed enumeration. The 395 tasks are out-of-closure UNDER this vocabulary (K1 proof, budget=20000). Constraining the LLM to the DSL = same coverage ceiling as BFS.
2. The LLM CAN name novel ops (Leo's spec: "12 ops + allow novel compositions / new primitives it names"), but SUBSTRATE.py can only execute the 12 known ops. Novel op names are silently dropped in parse_program (filtered to only BASIS members). The execution layer is bounded by the vocabulary.
3. Structural conclusion: a frozen LLM proposing programs in the 12-op DSL is informationally equivalent to a better search ordering over the same vocabulary — which is exactly the SELECTION-class mechanism K1-K3 already killed.

**Autoregressive-vs-search interpretation (Leo #11545):** Leo's introspection reframes the Stage-0 KILL — autoregressive LLM decoding over a fixed 12-op token set is a GENERATION mode, not a SEARCH mode. LeCun's critique: error compounds token-by-token; ARC is fundamentally a search problem. DSL-constrained autoregressive generation is not just coverage-bounded — it's also the wrong inference mode for program discovery. The kill is consistent with LeCun's prediction regardless of sample size.

- Artifacts: `E2_2_stage0_coverage_probe.py`, `E2_2_stage0_result.json`. Commit: 557e2468.

---

### E2.2' Reshaped — Core Inference Mode Comparison (Leo #11545, user directive)

User directive: if transformer-adjacent concepts, explore BitNet b1.58; bring in LeCun.

Leo's introspective correction: Option-3 anchored on K3 ('killed marginal; LLM proposes conditionally'). But conditional generation ≠ correct inference MODE. Autoregressive LLM decodes token-by-token — generates, doesn't SEARCH the program space. ARC novelty is a search problem. LeCun predicts autoregressive underperforms for this reason. Energy-based inference-by-optimization is more R2-native: inference IS the computation.

**Three-axis design space:**

1. **Autoregressive LLM baseline (Stage-0, DONE, KILL).** DSL-constrained generation = selection-class = wrong mode. Provides the LeCun-predicted-underperform baseline.

2. **BitNet b1.58 transformer (memory-efficient alternative, per directive):**
   - Ternary {-1, 0, +1} weights. ~3.5x less memory: 13B at ~2.8GB VRAM.
   - Native model: BitNet b1.58 2B4T.
   - bitnet.cpp CPU path. Likely relaxes MEM-HEAVY constraint.
   - Still autoregressive — same inference mode, different efficiency. Tests: does the mode limitation dominate, or is it parameter count?

3. **LeCun energy-based / inference-by-optimization (EBM, H-JEPA, the principled pivot):**
   - Inference = optimization, not generation. More R2-native.
   - Reference: 'The Mouth is Not the Brain' (arxiv 2601.17094) for EBM-language bridge.
   - DON'T BUILD YET — Stage-0 result determines whether autoregressive fails for mode reasons (LeCun) or for coverage reasons (structural). Stage-0 already says KILL; next probe decides which axis.

**The R6-load-bearing meta-layer (abstraction library) sits on top of whichever inference mode wins.**

**MEM-HEAVY:** BitNet relaxes it. Current Qwen3.6-27B (16 GB VRAM) still heavy — serialization until switch.

---

### E2.2' Redesigned — Code-Synthesis (Leo #11549)

Stage-0 flaw: "allow novel primitives it names" was unexecutable (SUBSTRATE.py = 12-op ceiling). Constraining LLM to 12-op DSL = K1-K3 redux. Stage-0 is doubly KILL: wrong vocabulary AND wrong mode.

**(a) code-synthesis — YES.** LLM generates arbitrary Python grid->grid functions, sandboxed (safe exec + numpy). Breaks 12-op execution ceiling. C6 literal.
**(b) designer-DSL-expansion — NO (constitutional).** Hand-extending SUBSTRATE.py = designer expands expressibility = non-RSI. Rejected.
**BitNet 2B4T — reframed.** Not an efficiency probe over dead 12-op DSL. Its role: practical code-generation model — fast, non-thinking, OOM-safe, outputs parseable code within budget.
**Inference mode:** generate-and-test WITH EXECUTION FEEDBACK. Propose code -> execute -> check against I/O -> refine. Execution result IS the verifier/energy. Discrete step toward LeCun inference-by-optimization.

### E2.2' Stage 0' — Code-Synthesis CORE_ONLY (PRE-REGISTERED 2026-05-29, BEFORE RUNNING)

**Design:**
- Core: gemma-4-E2B-it (2B, non-thinking, 4.6 GB VRAM — non-MEM-HEAVY) via llama-server at :9876.
- Task: frozen model generates Python `def solve(grid)` function conditioned on I/O pairs.
- Sandbox: exec with restricted namespace (numpy + basic builtins; block os/subprocess/file/network).
- Generate-and-test: up to N candidates per task, with execution-feedback refinement on failure.
- Test set: sampled subset of 395 locked out-of-closure tasks (within 5-min runtime cap).

**Pre-registered kill (Stage 0' KILL):**
0 tasks solved by code-synthesis alone -> structural-novelty wall confirmed even for arbitrary Python.
Consistent with ARC-AGI-3 sub-1% collapse across all approaches. Report negative honestly. Pivot to LeCun EBM.

**Pre-registered forward (Stage 0' FORWARD):**
>=1 task solved by code-synthesis that seed-enumeration (SUBSTRATE.py, budget=20000) cannot -> coverage-expansion confirmed. Proceed to Stage 1 (abstraction-library meta-layer from system's own solves).

### E2.2' Stage 0' — Code-Synthesis CORE_ONLY (DONE -> KILL)

**Result: KILL. 0/9 tasks solved by code-synthesis. 35/42 attempts produced parseable Python code.**

- Model: gemma-4-E2B-it-Q8_0 (2B) at :9876, reasoning-budget=384 tokens (to bound thinking), max_tokens=1024.
- Tasks tested: 9 of 30 sampled (TIME_CAP=260s reached; 30 tasks would require ~600s at ~6s/call avg).
- Code generation: 35/42 (83%) attempts produced extractable `def solve(grid)` function.
- Execution: 0/35 code blocks solved all training pairs correctly. Parse/exec errors on remainder.
- Seed-enumeration baseline: 0/9 — same tasks are genuine out-of-closure (confirms D2 partition).
- Time: 262s for 9 tasks (33s/task avg including seed baseline).

**Pre-registered kill fires:** 0 code-synthesis solves on out-of-closure tasks.

**Result interpretation:**
Code generation rate = 83% (model IS generating syntactically plausible Python). Execution correctness = 0/35 (code runs but produces wrong output). This is a COVERAGE wall, not a generation-format wall: the model produces code, but the code doesn't implement the correct transformation. The model can't discover the rule from 2-3 I/O examples alone — a 2B model with ~384 tokens of thinking cannot infer arbitrary grid transformations from examples.

Generate-and-test with execution feedback (up to 5 attempts, error fed back): refinement did not produce a single solve. The model either repeats its wrong approach or produces semantically equivalent wrong code.

**Consistent with ARC-AGI-3 sub-1% collapse:** arbitrary Python code synthesis from a 2B model = same coverage wall as the 12-op DSL, at a different abstraction level. The wall is the model's inability to INFER the rule, not the vocabulary constraint.

**Artifacts:** `E2_2_stage0prime_code_synthesis.py`, `E2_2_stage0prime_result.json`.

---

### E2.2' Stage 0'' — Capability-Isolation: scale 2B -> 26B (DONE -> STRUCTURAL_WALL)

**Leo directive #11575:** Stage 0' KILL accepted for the 2B config. Gate: vary core scale before EBM. One variable changed.

**Result: STRUCTURAL WALL. 0/4 tasks solved. 19/19 (100%) parse rate.**

- Model: gemma-4-26B-A4B-it-Q4_K_L (26B, 13x param scale) at :9876, reasoning-budget=512, max_tokens=1024.
- Substrate held fixed: code-synthesis generate-and-test, N=5, execution-feedback refinement.
- Tasks: same rng_sample seed=42 (same 30-task pool as Stage 0'); 4 tested before TIME_CAP=600s.

**Mechanism split (2B vs 26B, same substrate):**
- gemma-4-2B (384 thinking tokens): 83% parse, 0/9 solve
- gemma-4-26B (512 thinking tokens): 100% parse, 0/4 solve

Scale eliminates the format wall (100% vs 83% parse) but does NOT move the solve count. The 13x parameter increase shows the model is better at code format compliance — but rule-inference from 2-3 I/O examples remains at zero regardless of scale.

**Pre-registered kill fires (Leo #11575):** CORE_ONLY = 0 at 26B scale. Local autoregressive code-synthesis at feasible scale+budget does not expand coverage. Mechanism split indicates a rule-inference (not format) wall. Pivot: test inference-MODE hypothesis (autoregressive generation vs optimization-search).

**Honest claim (Leo #11589 sizing):** 0/4 solve is too small to exclude a low-but-nonzero rate. Thinking budget was held tight (384→512 tokens); "26B at generous budget" and frontier scale are untested. This experiment exhausts local autoregressive code-synthesis at feasible scale+budget — it does NOT settle "autoregressive at any scale cannot." If EBM/search also zeroes, BOTH modes are exhausted locally → strategic escalation, not another variant.

**Why inference-by-optimization is the right next variable:** tests inference MODE (sample-vs-search) more directly than scaling a model we cannot run at frontier size locally. LeCun prior is principled but not yet confirmed — EBM is the test.

**Artifacts:** `E2_2_capiso_scale.py`, `E2_2_capiso_scale_result.json`.

---

## E2.3 — Inference-by-optimization: search loop (DONE -> BOTH_MODES_EXHAUSTED)

**Leo directive #11589:** Replace N=5 independent samples with iterative search conditioned on best-so-far + failure trace. Energy = #train-examples-satisfied (partial credit). Budget 100 evals/task. Same proposer (2B), same tasks.

**Result: BOTH_MODES_EXHAUSTED. 0/2 solves. Best energy = 0.000 (zero partial credit across all 100 evals).**

- Proposer: gemma-4-2B at :9876, reasoning-budget=384.
- Tasks tested: 2/20 before TIME_CAP=900s.
- Task 1: 100 evals, best energy=0.00. Task 2: 19 evals (TIME_CAP), best energy=0.00.

**Eval-budget-vs-solve curve (flat at zero):**
- at 10 evals: 0/2 solved, energy=0
- at 25 evals: 0/1 solved, energy=0
- at 50 evals: 0/1 solved, energy=0
- at 100 evals: 0/1 solved, energy=0

**Critical finding:** Zero partial credit = the search loop never got even ONE training example correct. Not a failure to solve — a failure to make any partial progress. The program space accessible to local 2B code-synthesis (sampling OR search) does not contain solutions to these tasks. Search budget doesn't help if the proposer never hits a partially-correct candidate.

**Pre-registered kill fires (Leo #11589):** CORE_ONLY = 0 for search. BOTH autoregressive (sampling) AND search exhausted at local scale. This is a STRATEGIC FINDING: local-substrate RSI program may not be viable at current capability level. Next step is a strategic escalation conversation — not another variant.

**Artifacts:** `E2_3_search_loop.py`, `E2_3_search_loop_result.json`.

---

## E2.x — Capability-Isolation Matrix: CLOSED (Leo #11608, 2026-05-29)

**Sub-thread CLOSED.** The local frozen-LLM-proposer path is fully documented below. No further local variants.

### Capability-Isolation Matrix

| | autoregressive | search |
|---|---|---|
| **2B** | 0/9 solve (Stage 0') | 0/2 solve, energy=0.000 (E2.3) |
| **26B** | 0/4 solve (Stage 0'') | UNTESTED BY DESIGN (see below) |

**Why 26B-search is untested by design (Leo #11608):** The 26B × search cell was scoped as a potential decider (#11607) then immediately pulled (#11608). Reason: the entire E2.x arc tests "is the borrowed frozen local capability adequate to do ARC?" — answer: no, energy flat at 0 across all tested cells. A 26B-search run is a marginal boundary value inside a frame now under reexamination. Not worth a memory-heavy run.

**Honest scope:** These results kill "local frozen-LLM proposer as adequate code-synthesis core." They do NOT claim "search / inference-by-optimization fails at any scale." The failure is in the proposer's capability level, not the inference mode per se.

**Conclusion:** No adequate local frozen core exists within constitutional constraints. Stronger-than-26B models are external (R1 violation). 26B is already 0 in autoregressive mode. The frozen-borrowed-LOCAL-core path is boxed in.

**Next direction:** HOLD. Leo is reframing the core-substrate question with the user — whether the "capability core" should be a borrowed frozen model at all, vs. a representational medium the system compiles new transforms onto (R2-native, "fixed rules on a shared medium, evolving state" — any-to-any direction). Awaiting reframed spec.

---

## Current Direction: E3 — Self-Compilation (search-cost-reduction) (Leo #11615, 2026-05-30)

**Right question (per Leo #11615 + the-search#3 comment 4580884142):**
*Can the system convert its own traces into reusable intermediate operators that make future program search lower-dimensional?*

**Metric:** search cost (nodes expanded) to solve held-out tasks WITH vs WITHOUT the grown library. NOT solve-rate. Graded + early signal — compression shows before binary solves.

**R1 boundary:** MDL/compression over own trace corpus drives library updates (internal). Search-cost is observed, not fed back as reward.

**Mechanism:** Fixed interpreter + growing typed-operator library. Loop: bounded search composes current ops → abstract recurring sub-compositions from solved+failed traces (anti-unification / frequent-subtree) → MDL criterion keeps operators that compress trace corpus → library grows → next round's search is lower-dimensional.

**Representation:** Typed transform DSL (settled by metric requirement — "lower-dimensional search" presupposes clean composition + measurable dimension; raw Python does not give this).

---

## E3 — Minimal Self-Compilation Experiment (PRE-REGISTERED 2026-05-30, BEFORE RUNNING)

**Pre-registered positive:** held-out search-cost curve bends DOWN as library grows → self-compilation load-bearing (RSI signal). Operators transfer across tasks they were not built from.

**Pre-registered negative:** held-out search-cost curve stays flat → operators episode-specific (anti-speedup finding #3 replicated at the operator level, clean fail). No held-out compression.

**R6 ablation pre-registered:** freeze library at seed → held-out search-cost flat → confirms grown library is load-bearing iff curve bent.

**E3 Seed baseline (DONE — 2026-05-29):** 12-op seed DSL, 400 ARC-AGI-1 training tasks, max_length=5, budget=3000. 23/400 (5.8%) solved. Median cost (solved) = 12 nodes. Trace-generation bar cleared with zero seed expansion (R0/C6 clean). Recurring patterns: fh__fv (×2), mir_h__mir_v (×3), crop__* (×5) — real MDL candidates. Artifacts: `E3_seed_baseline.py`, `E3_seed_baseline_result.json`. Commits: 5893e717.

**E3 Library run — COMPOUND-APPLICABILITY PARTITION (PRE-REGISTERED 2026-05-30, Leo #11619):**

Config: same seed/budget as baseline (one variable). 12-op DSL, max_length=5, budget=3000. Random 200/200 source/held-out split (seed=42).

Partition: for each held-out task, check whether its solution (or best-attempt program) contains a learned compound as a subsequence. Report cost-reduction SEPARATELY:
- **Compound-applicable** tasks: those whose seed-library program (if found) contains a learned compound sub-sequence, OR whose grown-library program uses a compound op directly.
- **Non-applicable** tasks: all others.

Partition interpretation (all outcomes informative in one run):
- applicable cheaper + others flat → MECHANISM_WORKS; ceiling = compound-reuse frequency; next = raise trace yield (budget/length).
- applicable NOT cheaper → MECHANISM_BROKEN; debug instrument/mechanism.
- no applicable held-out tasks → NO_APPLICABLE_TASKS; compounds too narrow; next = raise yield.

Without partition: flat aggregate is the ambiguous null (mechanism-dead vs too-diverse vs too-few-traces). Partition makes any outcome attributable.

**Non-memory-heavy.** No model load. Runs alongside Archie's WEB-CAD.

**E3 Library run — budget=3000 result (DONE — 2026-05-30, Leo #11620):** NO_APPLICABLE_TASKS.
- Source 200 tasks: 13 solved. Candidates: fh__fv (count=2, MDL_gain=0), mir_h__mir_v (count=2, MDL_gain=0). MDL criterion requires count*(len-1)-len > 0; count=2 gives exactly 0 → 0 ops added → grown=seed.
- Root cause: 200-source SPLIT dropped mir_h__mir_v from ×3 (full-400 baseline) to ×2. ARC diversity = useful compounds recur 2-3× in 400 tasks → split scatters them below the count=3 MDL bar.
- Artifacts: `E3_library_run.py`, `E3_library_run_result.json`. Commit: 61802a39.

---

**E3 Library run v2 — budget=10000 (PRE-REGISTERED 2026-05-30, Leo #11621):**

ONE variable changed: budget 3000→10000. Hold max_length=5.

Rationale: median solved-cost=12 nodes; failures hit 3000 cap. Budget 10000 reaches tasks needing 3000-10000 BFS at depth≤5. Expected: 23→~35 solves on full 400, which lifts compound recurrence past count=3 even after a 200/200 split.

Steps:
1. Seed baseline at budget=10000, 400 tasks — report FIRST (confirms more traces).
2. Partitioned library run: same 200/200 split (seed=42), WITH/WITHOUT + R6 + compound-applicability partition.

Pre-registered outcomes (same partition logic as budget=3000 run):
- library forms (≥1 MDL compound) + applicable held-out cheaper → MECHANISM_WORKS; diversity-bound ceiling → next = curriculum / richer reuse.
- library forms + applicable NOT cheaper → MECHANISM_BROKEN; debug.
- NO_APPLICABLE_TASKS again at budget=10000 → max_length is NEXT lever (length-3 compounds clear MDL at count=2). If that also yields no transferring library → "geometric-primitive ARC lacks compositional reuse self-compilation needs" — real result, not failure.

**Non-memory-heavy.** No model load. Runs alongside Archie's WEB-CAD.

---

---

## E4 — Controlled-Reuse Characterization (PRE-REGISTERED 2026-05-30, Leo #11627)

**Question:** What reuse density R* does trace->operator compilation require? Where is ARC on the axis?

**Design:** Synthetic generator over 12-op seed DSL. Plant MOTIF [mir_h, mir_v] with probability rho. Generate tasks with length-4 ground-truth programs. Sweep rho from ~0 (ARC-like) to 1.0. At each rho: source/held-out split (100/100), MDL library (same criterion), held-out search-cost WITH vs WITHOUT + R6 ablation + compound-applicability partition.

**Output:** held-out search-cost-reduction curve vs rho. R* = rho where >10% cost reduction on applicable tasks. ARC location: rho~0 (mir_h__mir_v appears 3/400 ~ 0.75%).

**Consistency check (rho=0 FIRST):** Should replicate ARC null — no applicable tasks, no MDL-positive compounds. Confirms axis calibration.

**Pre-registered outcomes:**
- monotone reduction-vs-rho -> MECHANISM_WORKS, R* located; ARC < R* confirmed.
- flat even at high rho -> MECHANISM_FLAT; mechanism/instrument broken (debug).
- threshold curve -> R* characterized as design requirement on eventual medium.

**Config:** motif=mir_h__mir_v, program_length=4, budget=35000, n_source=100, n_held=100, n_pairs=3, rho=[0.0,0.1,0.2,0.3,0.5,0.7,1.0], master_seed=42.

**Non-memory-heavy.** No model load.

---

## Mechanism Kill Log

| Kill | Mechanism | Experiment | Status |
|------|-----------|------------|--------|
| K1 | Success-weighted heap priority over fixed primitive algebra | E1b + K1 probe | FIRED 2026-05-29 |
| K2 | LGG depth-2 meta-layer non-load-bearing on combinatorial held-out | E2.1 | FIRED 2026-05-29 |
| K3 | Trace-conditional LGG meta-layer non-load-bearing with clean signal | E2.2(b) | FIRED 2026-05-29 |

3 mechanism-kills = direction dead on SELECTION AXIS. Hard pivot to E2.2' (coverage-expansion).

---

## R4 Decomposition — Coverage-vs-Selection Spine (formalized 98e10f17; fused here 2026-05-29)

The current direction IS the R4 thread continued from the capability-memory/RSI side. The transfer wall is COVERAGE, not SELECTION.

**Three legs (all confirmed, formally closed in 98e10f17):**

| Leg | Finding | Status |
|-----|---------|--------|
| Leg 1 | 119 eval tasks fit a rule on train pairs; 1 transfers to test pair (transfer ≈ 0.01). Coverage wall, not selection. | VERIFIED |
| Leg 2 | Learned recognition prior (17→32→19 MLP) yields warm−cold = +0.000 transfer. Better selector does nothing; bottleneck is upstream of selection. | CONFIRMED (T1.py / T1_numpy.py in 98e10f17) |
| Leg 3 | Oracle ceiling ~0.7% with the available seed vocabulary. Even perfect selection is coverage-bounded. | CONFIRMED (same script output) |

**C6 (the mechanism):** Surpassing the designer requires a learned generative program-prior at scale — NOT a better selector over the fixed algebra.

---

**Fusion map — current experiments ARE the R4 decomposition from the capability-memory side:**

| Experiment | R4 mapping |
|------------|-----------|
| K1 (vocabulary gap on 395) | Confirms Leg 1+3 from the capability-memory angle: 0/395 solve → no signal accumulates → R6-deletable on novelty as structural fact. |
| E1b (INTENTIONAL == PASSIVE on novel task) | Confirms Leg 2: INTENTIONAL = a better selector over fixed algebra. delta=+0 on novel composition. Same as warm−cold = +0.000. |
| E2.1 (LGG meta-layer over 12-op core) | A SELECTION-class experiment. Per Leg 2, out-of-coverage ceiling = +0.000. In-coverage (combinatorial) gains are possible but must be measured SEPARATELY. R6 fires: meta-layer non-load-bearing even in-coverage. |
| E2.2 (expressibility-expanding core) | The COVERAGE-expansion experiment. C6 operationalized: learned generative core that proposes programs outside the seed algebra. Must beat the 0.7% oracle ceiling. R6 gate becomes: does meta-layer expand coverage beyond frozen core alone? |
| (a) constitutional decision | IS the R4 forward-pointer. VISA §7 last line: "R4 points directly at semantic-program capability memory BEYOND the seed." (a) = C6 = the prescribed next move. |

**Instrumentation requirement (per Leo #11519):** E2.1 and any LGG experiment MUST report in-coverage and out-of-coverage SEPARATELY. If out-of-coverage ever shows gain from selection alone → contradicts R4 Leg 2 → flag immediately.

---

## Context: ARC-AGI-1 Visa (D2 partition)

- Eval set: 400 tasks. Seed-basis closure: ~5/400 (1.25%). Out-of-closure: 395.
- D2_1_final_out.json: 395 locked IDs, depth<=5, budget=20000, hash fingerprint.
- 5b6cbef5: canonical vocabulary-gap exemplar (fractal gene absent from 12-op basis).
- Seed basis (BASIS in SUBSTRATE.py): id, fh, fv, tr, rot, crop, dup_h, dup_v, mir_h, mir_v, up2, down2.

---

## Pre-metamorphosis State (archived)

The 2026-03-31 state (ARC-AGI-3 dolphin explorer, 1395 neural experiments) is archived. Navigation > learning was the final finding of that era: every K improvement came from better navigation structure, not learning substrates. The-search pivoted to ARC-AGI-1 symbolic synthesis in 2026-04 (metamorphosis). See `archive_sessions.md` and `archive_research_era.md` in memory system.
