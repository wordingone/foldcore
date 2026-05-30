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

## CLOSED: 12-op Self-Compilation (Leo #11683, 2026-05-30)

**3 convergent 12-op negatives — common attributed cause: trace-starvation:**
1. **Solved-only contiguous** (E3 v1): 7 source programs, 0 MDL-positive compounds.
2. **Full-corpus contiguous** (E3 v2 v3, 28 programs): 3 MDL-positive compounds but n_applicable=0 on held-out.
3. **Holed-formation-density** (E3 v2 v3 funnel): 4 skeletons with ≥2 variants, 0 passed MDL. DENSITY_THRESHOLD — count too low for MDL gain.

**Attribution:** NOT mechanism-dead. NOT no-skeletons. Cause = trace-starvation under 12-op base + weak search. 15 source solves (7.5%) too sparse for MDL-positive formation at any pattern type.

**Next (Leo #11683):** Trace-density amplification curve — attack the attributed cause directly. See "Next Experiment" below.

---

## Current Direction: Stage 0' — Code-Synthesis Feasibility (Leo #11717, 2026-05-30)

**Trigger:** 12-op self-compilation CLOSED (HOLLOW_CLIMB, Leo #11717). 12-op base is the representational ceiling — mechanism works at synthetic R*=0.7 (E4), but real ARC structure is below R* at any reachable trace density. Refinement 2 (code-synthesis pivot) triggered.

**Right question (Leo #11717):** Does a richer EXECUTABLE LAYER solve more than 12-op brute, at the same compute budget? Isolates layer-vs-proposer: even weak enumeration over a richer grammar should lift coverage if the layer is the bottleneck.

**Executable layer:** Sandboxed Python grid→grid (numpy in→out). SANDBOX REQUIRED: subprocess + timeout + restricted builtins + no file/network I/O.

**Proposer:** Weakest first — bounded-depth enumeration over a Turing-richer-than-12-op grammar (12 ops as leaves + map-over-cells/objects + conditionals on cell/color/shape + coord/color arithmetic). NO learned proposer yet.

**Metric:** solve-rate code-synthesis vs 12-op-brute, SAME compute budget.

**PRISM multi-domain FROM INCEPTION:** run same generate-and-test on MBPP (Python layer = MBPP is native). Report solve-rate on BOTH ARC grid→grid AND MBPP. Build on existing prism.py / arc1_prism_adapter.py.

**Pre-registered outcomes:**
- **PASS:** code-synth > 12-op-brute on ARC + non-trivial MBPP → layer was bottleneck → Stage 1 (self-compilation library on richer base + R6 ablation)
- **CHICKEN-EGG:** solve-rate ≈ brute or near-zero (richer space drowns weak enumeration) → proposer needed → recognition-net-from-own-solves (DreamCoder, R2-adjacent) or curriculum-from-enumeration-solvable-up

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

**E3 Library run v2 — budget=10000 result (DONE — 2026-05-30):** NO_APPLICABLE_TASKS.
- Seed baseline: 23/400 — identical to budget=3000. **PREDICTION FALSIFIED (per Leo #11633): budget=10000 was expected to yield ~35 solves; it yielded 23.** The 377 failures ALL hit the budget cap regardless of 3000 vs 10000: median solved cost = 12 nodes; all failures are vocabulary-bound (K1, not depth-bound). Budget is not the bottleneck.
- Source 200 tasks (seed=42 split): 15 solved. Candidates: fv__mir_v (count=2, MDL_gain=0), mir_h__mir_v (count=1, MDL_gain=-1). 0 ops added. Library=seed.
- Artifacts: `E3_seed_baseline_v2.py`, `E3_seed_baseline_v2_result.json`, `E3_library_run_v2.py`, `E3_library_run_v2_result.json`. Commit: ad098f91.

**Reuse-sparsity evidence (DONE — 2026-05-30, Leo #11623):** `E3_reuse_sparsity_analysis.py`, result `E3_reuse_sparsity_result.json`. Commit: a7e11b57.

**Exact scope of the reuse-sparsity claim (per Leo #11633):**
> "No contiguous-subsequence compound achieves MDL-positive count on a 200/200 split of the 23 programs solved by the 12-op seed DSL under solved-only contiguous MDL."
This does NOT claim "ARC cannot self-compile." It is scoped to: this DSL + this abstraction engine + the 23 solved traces. Full-corpus (failed+solved) with anti-unification with holes and non-contiguous extraction is the next test.

---

## E3 Full-Corpus Engine (PRE-REGISTERED 2026-05-30, Leo #11633)

**Engine upgrade (spec-completion, not a parameter variant):**
- (a) Ingest failed/partial traces (the 377) via partial-match BFS scan.
- (b) Anti-unification with holes: LCS-based non-contiguous pattern extraction.
- (c) Non-contiguous motif MDL over full corpus (solved + failed partial programs).
- MDL criterion unchanged but applied over full corpus.
- R2-native: failures are part of the system's own history.

**Two extra gates (Leo #11638):**
- Failed traces must expose meaningful partial structure (energy/progress evidence). If inert: say so explicitly; full corpus = solved-only; negative stands.
- Report solved-only vs failed-derived MDL contribution SEPARATELY.

**Pre-registered outcomes:**
- KILL (scoped negative, strong): full-corpus MDL still no positive compounds on held-out → ARC corpus (including failures) is below R* → reuse-sparsity is a full-corpus property, not a solved-traces artifact → forward = richer-reuse seed domain.
- FORWARD: richer abstraction yields compounds + held-out reduction → solved-only was engine artifact → continue on ARC.

**Artifacts:** `E3_full_corpus.py`, `E3_full_corpus_result.json`.

**E3 Full-Corpus v1 result (DONE — 2026-05-30, Leo #11651 + #11660):** SCOPED CONTIGUOUS NEGATIVE.
- Failed traces NOT inert: source_partials=13 (13/185 failed tasks show partial structure). Partial-structure gate: INFORMATIVE.
- Contiguous MDL-positive compounds (full corpus): 3 — down2__down2 (count=9, gain=7), crop__down2 (count=3, gain=1), down2__down2__down2 (count=3, gain=3). All from partial-match programs, none from solved-only corpus.
- Held-out: baseline 8/200, with-lib 8/200. Cost: 9602.6 → 9603.7 (+0.0%). **n_applicable=0.**
- Classification: **valid scoped negative for executable CONTIGUOUS compounds under this DSL/split.** Explicitly NOT a kill for holed/non-contiguous (holed operators were informational-only in v1 — untested).
- Key point: contiguous compounds transferred zero held-out benefit despite forming from failed traces. This is because ARC vocabulary-bound failures produce partial programs that share down2-heavy subsequences — not semantically generalizing operators.
- Artifacts: `E3_full_corpus.py`, `E3_full_corpus_result.json`. Commit: pending.

**Contiguous results are now complete.** Solved-only contiguous (0 MDL-positive under any split) + full-corpus contiguous (3 MDL-positive compounds but n_applicable=0 on held-out) both flat. Contiguous extraction is the degenerate special case. The core mechanism — anti-unification WITH HOLES — remains untested.

**E3 Full-Corpus v2 result (2026-05-30) — FORMATION NEGATIVE (density), confirmed by v3:**
- STALE prior runs archived: first run had wrong DATA_PATH (0 tasks). Second run loaded training+evaluation (800 tasks = eval leakage risk). Results below are v3 (training-only, 400 tasks, confirmed disjoint).
- **v3 findings (2026-05-30, training-only, source/held DISJOINT):** 15/200 source solved, 13 partials = 28 total programs. Contiguous MDL-positive: 3 (down2__down2 gain=7, crop__down2 gain=1, down2__down2__down2 gain=3). Formation funnel for holed ops: total_pairs_checked=34, candidates_2plus_variants=4, candidates_passed_MDL=0, gain_distribution n=4 min=-2.0 max=0.0 mean=-1.0. **DIAGNOSIS: DENSITY_THRESHOLD** — 4 candidates exist with >=2 variants, none pass MDL (count too low). Held: R6+holed == R6 == baseline (+0.0%), holed-selection=0 tasks.
- **NOT a kill for holed mechanism.** Pre-registered kill required holed ops executable + selected + flat — holed ops never formed. Cause: count too low for MDL gain (28 programs, 4 candidates).
- Consistent with K1 bootstrap trap: solve-rate 7.5% too low to generate programs dense enough for any reuse pattern.
- Artifacts: `E3_full_corpus_v2.py`, `E3_full_corpus_v2_result.json` (v3 result).

---

---

## E4 — Controlled-Reuse Characterization R*-Grade (PRE-REGISTERED 2026-05-30, Leo #11627 + Kai review #11638)

**Question:** What reuse density R* does trace->operator compilation require? What is R* under rigorous measurement with distractors?

**Design (R*-grade per Kai review #11638):**
- Generator: NON_MOTIF_PAD for padding (excludes mir_h/mir_v from non-planted positions); MOTIF [mir_h, mir_v] planted at rate rho. **Framing correction (Kai review, Leo #11646):**
  - rho=0 is a **synthetic low-reuse calibration point** — it does NOT replicate the ARC null. ARC-null claims require directly sampling ARC-derived programs, not a synthetic rho=0 draw.
  - "3/400 tasks" is the contiguous-subsequence solved-motif count (mir_h__mir_v across the 23 solved), NOT a rho-axis location for ARC. ARC's position on the rho-axis is **UNMEASURED** until ARC-derived programs are sampled. Drop the "ARC sits at rho~0" framing.
- Distractors: recurring motifs that are source-only, semantically canceling, or non-transferable. Required — else the positive is hollow.
- Measured densities: report OBSERVED source/held motif density alongside injected rho.
- Attribution: planted compound added to library? Cost reduction from tasks using planted compound vs all-learned-compounds ablation.
- Multi-seed per rho: confidence bands. Single seed = one-draw point estimate, too weak for R*.
- Search-order accounting: program-length reduction AND node-count reduction reported separately (macro-prior + dimension effect).
- Library restricted to planted MOTIF compound only — isolates motif signal from spurious co-occurrences.

**5-item accept bar (Kai, #11638 + #11640):** A green run without all 5 items is a smoke-test, NOT R*. Label accordingly.

**Output:** held-out search-cost-reduction curve vs rho with confidence bands. **Accepted R* = aggregate-net crossover over ALL held-out tasks, overhead included** (not motif-subset). Script-reported motif-subset threshold = 0.1; NOT accepted R*. ARC's rho-axis location UNMEASURED — requires ARC-derived program sampling.

**Pre-registered outcomes:**
- aggregate-net negative at some rho AND variance band excludes 0 -> R* located (AGGREGATE crossover).
- flat at all rho even at high density -> MECHANISM_FLAT; mechanism/instrument broken.
- threshold curve -> R* characterized as design requirement on eventual medium.
- NO prediction that holed lowers R* until a result exists.

**Config (base):** motif=mir_h__mir_v, program_length=4, budget=35000, n_source=100, n_held=100, n_pairs=3, rho=[0.0,0.1,0.2,0.3,0.5,0.7,1.0], multi_seed_per_rho=3+, master_seed=42.

**Note:** Smoke-test run (before Kai review) showed consistency-check FAIL due to PAD_OPS including motif ops + 58 spurious compounds from general MDL. Fixed: NON_MOTIF_PAD (after MOTIF def) + motif-specific library + NameError from definition-order corrected.

**E4 R*-grade (contiguous) result (DONE — 2026-05-30, `E4_rstar_grade.py`) — NOT ACCEPTED:**
- Config: MOTIF=mir_h__mir_v, DISTRACTOR_S=rot__rot (source-only), CLEAN_PAD (8 ops), program_length=4, budget=35000, n_source=50, n_held=50, n_seeds=3.
- Script-reported R*=0.1 (motif-applicable subset first >10% reduction). **NOT ACCEPTED** — three blocking issues (Leo #11665):
  1. R* must be AGGREGATE-NET crossover (all held tasks, overhead included), not motif-subset.
  2. Distractor (rot__rot) never enters library (BFS compresses to fh+fv) — concentration test untested.
  3. Planted-only ablation missing — general MDL has 21-33 compounds; improvement from coincidental compounds not isolated.
- Key findings (not final):
  - MOTIF-applicable: strong synthetic signal at sufficient rho — **NOT monotone** (−98.3% @0.1, −79.9% @0.2, −94% @1.0). Per claim-hygiene: "strong synthetic signal at sufficient rho," NOT "clear and monotone."
  - Aggregate: +118% at rho=0 (coincidental compounds carry search overhead exceeding motif benefit). This is the R* evidence — net gain requires overhead < benefit.
  - dist_in_lib=0 always: distractor design bug, not real signal.
- Artifacts: `E4_rstar_grade.py`, `E4_rstar_grade_result.json`.

**E4-holed result (auto-emitted from E4_holed_result.json)**

**Config:** MOTIF=mir_h__mir_v, DISTRACTOR=dup_h__fv (independent roll, rho_d=0.4), N_SEEDS=5, prog_len=4, budget=35000, n_source=100, n_held=100

**R* definition (aggregate-net):** min rho where aggregate_delta < -5.0% AND variance band (mean+std) excludes 0. ALL held tasks, overhead included.

**R* results:**
- R*_contiguous (B): 1.0 — aggregate-net crossover, all-MDL library
- R*_holed (C): 0.7 — aggregate-net crossover with holed operators
- Planted-only P: mechanism signal confirmed; R*_P NOT reported as design threshold (isolated condition, overhead not included).

**Key signal (NOT accepted as R*):** script-reported motif-subset threshold = 0.1 from prior E4_rstar_grade run. NOT accepted — was motif-subset, not aggregate-net.

**Distractor load-bearing check + per-rho curve:**

| rho | AGG B (±std) | AGG C (±std) | Planted P (±std) | dist_in_lib | holed_sel |
|-----|--------------|--------------|------------------|-------------|-----------|
| 0.0 | +58.3%±50.0 | +53.5%±44.8 | +0.0%±0.0 | 1.00 | 27.2 |
| 0.05 | +11.7%±37.1 | +11.3%±36.8 | +20.6%±18.4 | 1.00 | 29.6 |
| 0.1 | +68.8%±54.7 | +64.3%±44.4 | +5.7%±14.4 | 1.00 | 19.0 |
| 0.2 | +43.7%±26.0 | +43.0%±28.6 | -26.9%±28.2 | 1.00 | 16.0 |
| 0.3 | +60.1%±63.3 | +60.4%±63.0 | -30.4%±7.1 | 1.00 | 11.2 |
| 0.5 | -20.0%±29.4 | -24.6%±27.4 | -70.2%±7.2 | 1.00 | 41.2 |
| 0.7 | -24.7%±30.5 | -29.2%±28.6 | -76.7%±5.6 | 1.00 | 53.6 | ← R*_C
| 1.0 | -85.6%±4.8 | -91.3%±3.5 | -94.1%±0.3 | 1.00 | 81.6 | ← R*_B

**Verdict (scope-qualified, PRISM-gated):**
**Synthetic aggregate-net R*_holed=0.7 under THIS generator/search/band rule.** B (contiguous) reaches R*=1.0; holed operators lower synthetic R*: 0.7 vs 1.0 (holed_lowers=True). NOTE: rho=0.7 is BARELY band-positive (C mean+std=-0.61, barely excludes 0).
Load-bearing gates pass: dist_in_lib=1.00 (distractor load-bearing), selected_holed=53.6 (holed ops selected by search).
**PRISM gate:** General self-compilation claims still require PRISM/multi-domain evidence (MBPP always present + masked ARC). This result is synthetic-only and scoped to this generator and band rule.
R*_contiguous (B): 1.0 — all-MDL library achieves aggregate crossover.

**Holed lowers R*:** 0.7 vs 1.0 (holed_lowers=True).

**Artifacts:** `E4_holed_operators.py`, `E4_holed_result.json`.
<!-- /E4-HOLED-RESULT -->

---

## E3 Trace-Density Amplification Curve — CLOSED: HOLLOW_CLIMB (Leo #11717, 2026-05-30)

**Question:** Is 12-op formation-density failure caused by trace-count insufficiency or representational-base insufficiency?

**Answer: REPRESENTATIONAL CEILING.** HOLLOW_CLIMB confirmed by transfer test (Leo #11717 accepted). 12-op base closed as the ceiling.

**Gated curve (commit 75754106, all 4 Kai gates):**

| Step | Source tasks | Programs | 2plus_cand | MDL-pos | Diagnosis |
|------|-------------|----------|------------|---------|-----------|
| 1 (real-200) | 200 | 8 | 0 | 0 | STRUCTURAL_STARVATION |
| 2 (full-real ~600) | 600 | 13 | 0 | 0 | STRUCTURAL_STARVATION |
| 3 (+D4 aug ~4200) | 4200 | 91 | 6 | **2** | **PASSED** |

Gate 3 content-hash leakage: CLEAN (hits=0, 3600 augmented vs 200 held). Gate fields in artifact: held_task_ids=200, source_construction per step, aug_leakage_check, curve_table.

**Transfer test (commit c6a7d433) — THE DECIDER:**

| Condition | mean_cost | solved | new_solves | delta |
|-----------|-----------|--------|------------|-------|
| Baseline (no library) | 9264.7 | 15/200 | — | — |
| With holed library (24 compounds) | 9311.5 | 14/200 | 0 | **+0.5%** |

Formed ops (eps___HOLE___mir_h__mir_v, fh__mir_h___HOLE___eps) are pure D4 augmentation artifacts — mirror/flip compositions matching the injected transforms. Zero new solves, +0.5% overhead. No transfer to original held-out structure. 12-op base IS the representational ceiling: E4 proved mechanism works at R*=0.7 on synthetic data; real ARC never reaches that density.

**Pending Kai gate:** formal close of transfer instrument (independent audit). Verdict dispositive regardless.

**Artifact chain:** E3_density_curve.py, E3_density_curve_result.json (75754106), E3_transfer_test.py, E3_transfer_test_result.json (c6a7d433).

---

## Stage 0' — Code-Synthesis Feasibility (PRE-REGISTERED 2026-05-30, Leo #11717, BEFORE RUNNING)

**Trigger:** 12-op HOLLOW_CLIMB accepted. Refinement 2 fired.

**Executable layer:** Sandboxed Python grid→grid (numpy). Generate-and-test on train I/O pairs; all-match → apply to held test pair. **SANDBOX REQUIRED:** subprocess + timeout + restricted builtins (no file/network/exec).

**Proposer (weakest first):** Bounded-depth enumeration over Turing-richer grammar:
- Leaves: 12 seed ops (id, fh, fv, tr, rot, crop, dup_h, dup_v, mir_h, mir_v, up2, down2)
- Extensions: map-over-cells, map-over-objects, conditionals on cell/color/shape, coord/color arithmetic

**Metric:** solve-rate code-synthesis vs 12-op-brute, **SAME compute budget**.

**PRISM multi-domain FROM INCEPTION:** same generate-and-test on MBPP (Python layer = MBPP is native). Report solve-rate on BOTH ARC grid→grid AND MBPP. Build on existing prism.py / arc1_prism_adapter.py.

**Pre-registered outcomes:**
- **PASS:** code-synth > 12-op-brute on ARC AND non-trivial MBPP solve-rate → layer was bottleneck → Stage 1 (self-compilation library on richer base + R6 ablation)
- **CHICKEN-EGG:** code-synth ≈ brute OR near-zero ARC (richer space drowns weak enumeration) → proposer needed first → recognition-net-from-own-solves (DreamCoder/R2-adjacent) or curriculum-from-enumeration-solvable-up

**Gates (Kai #11717):** formal feasibility result after Leo sees solve-rate table. Mail Kai + Leo together.

---

### Stage 0b — Arbitrary-Python Exec Feasibility: CHICKEN-EGG CONFIRMED (2026-05-30, Leo #11728)

**Bigger-fixed-basis clarification (Leo #11728):** The prior "Stage 0'" 54-op ARC result (54 hand-added designer primitives solving 7.5% of ARC) is NOT code-synthesis. It is designer-DSL-expansion (Refinement 2 explicitly rejects this path). The real code-synthesis ARC test is arbitrary-Python exec-based generate-and-test.

**Result: CHICKEN-EGG CONFIRMED. 0 new solves beyond 12-op DSL baseline.**

- 182 diverse lambda templates (geometric, value-conditional, structural, color-swap, erase), safe_exec with numpy + threading.Thread timeout.
- 50 held-out tasks (split_seed=42).
- 12-op DSL baseline: 2/50 solved. Arbitrary-Python sweep: 1/50 (the 1 solve was already solved by DSL). New solves: 0.
- Total evaluations: 8919. Template errors (wrong output / exception): 1274.
- Root cause: search space intractable without proposer. 182 templates × 50 tasks = ~9K evaluations, none of which propose the correct transformation. Proposer is the bottleneck, not the execution layer.
- Artifacts: `stage0b_arc_synth.py`, `stage0b_result.json`. Commit: 0cdca781.

---

### Stage 1 — Wake-Sleep on Whole-Grid Grammar: HOLLOW; DISCRIMINATION PENDING (2026-05-30, Leo #11734 + #11740)

**Leo #11734 GO:** Wake-sleep self-compilation loop approved. Typed combinators, PRISM multi-domain, net-free, dream-sleep cold-start, transfer from iteration 1, gated JSON, Kai gates.

**Stage 1 grammar (whole-grid — INCOMPLETE):** id, fh, fv, tr, rot, crop, dup_h, dup_v, mir_h, mir_v, up2, down2 + recolor(c1,c2) × 72 + compose + library macros. MISSING: object-extract(pred), map-over-objects(transform), translate(vec), conditionals.

**Result: FAIL (informative) for whole-grid grammar. Discrimination NOT achieved.**

| Iter | ARC solved | ARC rate | Library | Transfer delta |
|------|-----------|----------|---------|----------------|
| 1    | 10/30     | 33.3%    | 0→1     | −0.2% (HOLLOW) |
| 2    | 10/30     | 33.3%    | 1→1     | −0.2% (HOLLOW) |
| 3    | 10/30     | 33.3%    | 1→1     | −0.2% (HOLLOW) |

1 macro formed: compose(mir_h, mir_v) ≡ rot_180, occurrences=2, savings=1. Flat across all 3 iterations.

**Why discrimination failed (Leo #11740):** Two tasks that both "recolor the largest object red" have DIFFERENT whole-grid programs (object at different positions/shapes in each task) → they look unique to a whole-grid grammar → no macro forms. HOLLOW on whole-grid grammar is consistent with BOTH (a) ARC structure-absence AND (b) object-level grammar-poverty. The discrimination test requires object-centric combinators.

**RESEARCH_STATE action (Leo #11740):** Do NOT promote to structure-absence. Discrimination PENDING object-centric grammar re-run.

- MBPP: 1/50 (2.0%) — whole-grid grammar covers list→list/int problems poorly.
- Dream-sleep: ran (1/1 macros active in fantasies, count reported).
- Augmentation status: CLEAN (no D4 augmentation applied).
- Elapsed: 64s. Artifacts: `stage1_wake_sleep.py`, `stage1_result.json`. Commit: e32aba20.

---

### Stage 1b — Object-Centric Wake-Sleep: CLOSED: STAGE1B_EMPTY_LIBRARY_VACUOUS_TRANSFER (2026-05-30, Leo #11740)

**Leo #11740 GO:** Object-centric grammar re-run. Full 200 held-out. Augmentation status stated. Dream-sleep confirmed. Transfer-to-original-held.

**Grammar corrections:** Add object-extract(predicate), map-apply(pred, transform), translate-object(vector), delete, keep-only. Object = connected component (4-connected). Predicates: largest, smallest, non-bg, unique-color, most-common-color, all, color_c (c=1-9). Transforms: recolor_c (c=0-9), delete, keep-only, translate_{dy}_{dx} (8 directions). 16 predicates × 22 transforms = 352 map-apply ops + 10 whole-grid prims.

**Curriculum corrections:** N_CURRICULUM=100 RANDOM from pool (not sorted-by-complexity easiest-30 — eliminates easy-set bias). N_HELD=200 original. Transfer bug fixed (one baseline search per task).

**Result: STAGE1B_EMPTY_LIBRARY_VACUOUS_TRANSFER. Structure-absence NOT established. Escalation held.**

| Iter | ARC solved | ARC rate | Library | Transfer delta |
|------|-----------|----------|---------|----------------|
| 1    | 5/100     | 5.0%     | 0       | +0.0% (vacuous; empty library) |
| 2    | 5/100     | 5.0%     | 0       | +0.0% (vacuous; empty library) |
| 3    | 5/100     | 5.0%     | 0       | +0.0% (vacuous; empty library) |

- MBPP: 4/50 (8.0%) flat.
- Library stays EMPTY: no macros formed, so self-compilation was not exercised.
- Transfer delta=+0.0% is not a transfer negative. With `library_final={}`, the with-library condition is a no-op comparison against baseline.
- The artifact does not contain solved-program bodies/hashes or `program_type_breakdown`, so claims such as "2/5 use map_apply" are unsupported and must not be used as evidence.
- The artifact does not contain an `abstraction_funnel`, so it cannot distinguish "no shared structure" from search-yield/depth starvation.
- Stage 1b used `BUDGET=400`, which the script defines as all 330 depth-0 programs plus only 70 depth-1 compositions. This is too shallow to establish absence of reusable depth-2 structure.
- Dream-sleep: ran (30 fantasies sampled per iteration), but with an empty library there were no active macros.
- Augmentation: CLEAN (no D4 augmentation applied).
- Elapsed: 20.9s.

**Discrimination not achieved:** Stage 1 whole-grid HOLLOW was ambiguous. Stage 1b corrected the grammar surface, but its solved yield was tiny and its library stayed empty. Because no macro library formed, self-compilation transfer was not tested. Object-level grammar poverty is not proven solved, and ARC structure absence is not established.

**Next gate:** Stage 1c depth-2 flat BFS, no recognition/proposer net, preserving the Stage 1b grammar to isolate depth. Required result fields: `max_depth_reached`, `depth_budget_by_level`, solved-program bodies/hashes, `program_type_breakdown`, `abstraction_funnel`, `library_final` with source depths, and original-held transfer with selected macro counts.

**Escalation rule:** Stage 2 recognition/proposer is justified only after a valid Stage 1c `FORMATION_NEGATIVE_DEPTH2` or `HOLLOW_DEPTH2`. Any-to-any representation remains premature until Stage 2 also produces deeper solves with still-zero reuse/transfer.

Artifacts: `stage1b_object_centric.py`, `stage1b_result.json`. Commits: 2698a9eb, 55fab552.

---

### Stage 1c — Depth-2 Flat BFS (Object-Centric, N_CURRICULUM=200): CLOSED: SINGLE_ARM_N200_DEPTH_STARVATION_SIGNAL (2026-05-30, Leo #11745/#11749/#11752/#11766)

**Leo #11766 reclassification: SINGLE_ARM_N200_DEPTH_STARVATION_SIGNAL.** Kai original: DEPTH_STARVATION. Two-arm flat plan (Leo #11759) SUPERSEDED by holed grammar pivot (#11766). Stage 1c-HOLED is the decisive test.

**Result: Structure-absence NOT established. Flat-330 is BRANCHING-BOUND.**

| Iter | ARC solved | ARC rate | Depth-0 solves | Depth-1 solves | Library | Transfer delta |
|------|-----------|----------|----------------|----------------|---------|----------------|
| 1    | 9/200     | 4.5%     | 9              | 0              | 0       | +0.0% (VACUOUS; empty library) |
| 2    | 9/200     | 4.5%     | 9              | 0              | 0       | +0.0% (VACUOUS; empty library) |
| 3    | 9/200     | 4.5%     | 9              | 0              | 0       | +0.0% (VACUOUS; empty library) |

- MBPP: 4/50 (8.0%) flat across all iterations.
- Grammar: 16 predicates × 20 transforms = 320 map_apply + 10 whole-grid prims = 330 depth-0 leaves. PRESERVED EXACTLY from Stage 1b (one-variable discipline).
- Depth budget: 330 depth-0 (100%) + 2,670/108,900 depth-1 programs evaluated (2.45%). Zero depth-1 solves. Full depth-1 estimated intractable (~41× budget).
- Augmentation: CLEAN (no D4 augmentation). Elapsed: 241.2s.

**Solved programs (9 unique, all depth-0):** map_apply × 5 (`non_bg→recolor_5`, `all→delete`, `all→translate_+1_+0`, `color_7→recolor_5`, `color_6→recolor_2`), prim × 4 (`up2`, `rot_90`, `tr`, `rot_180`). Zero depth-1 solves from 2,670 depth-1 programs evaluated.

**Abstraction funnel — STRUCTURAL FINDING:**
- 9 unique depth-0 programs each appear 3× (occ≥2). Repetitions detected.
- ALL 9 REJECTED by MDL: body_size=1 → savings = occ*(1-1)-1 = -1. Always negative.
- Structural barrier: depth-0 programs (body_size=1) CANNOT form macros by MDL construction. Macro formation requires depth-1 compose programs (body_size≥2).

**Branching-bound analysis (Leo #11766):**
- Depth-1: 330² = 108,900 programs. 2.45% coverage. Intractable at flat-330.
- Depth-2: 330³ ≈ 36M programs. Completely intractable at any reasonable budget.
- Root cause: 330 concrete ops bloated by enumerated variants (72 recolor variants). Skeleton-level shared structure hidden behind concrete enumeration.
- Fix: HOLED grammar — ~14 skeleton types collapse depth-1 to 14²=196 (tractable), depth-2 to 14³=2,744 (tractable).

**Claim scope:** 330 depth-0 programs (100%) and 2,670/108,900 depth-1 programs evaluated (2.45%), zero depth-1 solves. Structure-absence cannot be claimed. N_CURRICULUM confound: changed 100→200 from Stage 1b (two-variable change). The branching-bound fact holds independent of N: no corpus-size variation fixes a 41× budget deficit on flat-330.

---

### Stage 1c-HOLED — Holed Skeleton Grammar BFS: PRE-REGISTERED (Leo #11766/#11769/#11785, 2026-05-30)

**Variables changed (explicit — two-variable change, both deliberate):**
1. Representation: concrete-330 → holed skeleton grammar (~14 skeleton types). Root-cause fix for branching-bound (flat-330 depth-1 = 108,900 @ 2.45% coverage).
2. Search order: skeleton-pair priority (small-fill pairs first) within same BUDGET=3000.

**Holed skeleton grammar (~14 types):**
- MAP_RECOLOR: map_apply(pred, recolor_X) — 16 preds × 10 colors = 160 fills
- MAP_DELETE: map_apply(pred, delete) — 16 fills
- MAP_KEEPONLY: map_apply(pred, keep_only) — 16 fills
- MAP_TRANSLATE: map_apply(pred, translate_*) — 16 preds × 8 dirs = 128 fills
- PRIM_*: 10 whole-grid primitives, 1 fill each. Total 14 skeleton types, 330 concrete fills.

**Abstraction funnel (anti-unification + NON-TAUTOLOGY MDL):**
- Groups solved programs by SKELETON TYPE: MAP_RECOLOR(non_bg→5) and MAP_RECOLOR(color_7→5) both count as skeleton "MAP_RECOLOR" → occ=2 at skeleton level.
- **NON-TAUTOLOGY MDL guard (Kai #11768):** Skeleton macro forms only if it net-compresses INCLUDING fill costs: cost(skeleton) + sum_task(cost(fills)) < cost(concrete programs stored separately). Per-candidate report: skeleton_bits, total_fill_bits, concrete_baseline_bits, net_gain. Without this guard, "skeletons repeat" is tautological (any two MAP_RECOLOR tasks share the skeleton by construction).
- **MDL discipline:** `occ>=2` is only a candidate threshold. Macro formation is accepted only when the artifact records positive net gain after charging skeleton storage plus all parameter/fill costs. Hand-parameterized primitive skeleton repeats are not evidence of learned abstraction by themselves.

**Acceptance = held-out transfer (Kai #11769):** Skeleton forming (occ>=2, net_gain>0) is NECESSARY, not sufficient. Acceptance requires HOLED_TRANSFER_GENUINE: holed library reduces held-out search cost. Formation-without-transfer = HOLED_HOLLOW.

**Required result fields (Kai holed gate):**

- `grammar_mode`: `holed_object_centric`
- `holed_skeleton_inventory`: skeleton names, arities, fill domains, concrete expansion counts
- `curriculum_ids`, `held_task_ids`, `curriculum_held_disjoint`
- `depth_budget_by_level`: skeleton-depth coverage and fill-attempt coverage recorded separately
- `max_depth_reached`
- `solved_programs`: task IDs, skeleton bodies, fills, hashes, and depth
- `program_type_breakdown`: holed map-apply, holed transform, whole-grid, compose
- `abstraction_funnel`: skeleton repeats via anti-unification, occurrence>=2 candidates, MDL gains, accepted/rejected macros and reasons
- `library_final`: macro skeletons, fills/variants, occurrences, source task IDs, source depths
- per-candidate MDL accounting for every accepted macro: `skeleton_bits`, `total_fill_bits`, `concrete_baseline_bits`, `net_gain`
- `transfer`: baseline, with-library, selected macro counts, selected holed counts, new solves, aggregate cost delta
- `claim_scope`: must say depth plus representation changed; must not claim pure depth isolation
- `kai_classification`: one of `HOLED_INSTRUMENTATION_INCOMPLETE`, `HOLED_DEPTH_STARVATION`, `HOLED_FORMATION_NEGATIVE`, `HOLED_HOLLOW`, `HOLED_TRANSFER_GENUINE`

**Kai 5-label holed classification:**
- `HOLED_INSTRUMENTATION_INCOMPLETE`: required fields missing or malformed
- `HOLED_DEPTH_STARVATION`: zero depth-1 compose solves; fill coverage insufficient to test skeleton formation
- `HOLED_FORMATION_NEGATIVE`: depth-1 solves found but no skeleton macro nets positive gain (tautology guard triggered or savings<0)
- `HOLED_HOLLOW`: skeleton macro formed with net_gain>0 but held-out transfer not improved
- `HOLED_TRANSFER_GENUINE`: skeleton macro formed + held-out search cost reduced

**Stage 2 discipline:** Stage 2 proposer fires ONLY if holed pivot yields HOLED_DEPTH_STARVATION, HOLED_FORMATION_NEGATIVE, or HOLED_HOLLOW. Any-to-any escalation remains premature.

**Result (2026-05-30): CLOSED: HOLED_HOLLOW**

| Iter | ARC solved | ARC rate | Depth-0 | Depth-1 | Library | Transfer delta |
|------|-----------|----------|---------|---------|---------|----------------|
| 1    | 11/200    | 5.5%     | 9       | 2       | 1       | +0.2% (HOLLOW) |
| 2    | 11/200    | 5.5%     | 9       | 2       | 1       | +0.2% (HOLLOW) |
| 3    | 11/200    | 5.5%     | 9       | 2       | 1       | +0.2% (HOLLOW) |

- **2 depth-1 solves**: skeleton-pair ordering yields 2 compose solves (vs 0 in flat Stage 1c). Same BUDGET=3000.
- **1 skeleton macro formed**: COMPOSE(PRIM_CROP,MAP_KEEPONLY) — occ=2 tasks, net_gain>0 after fill-cost charging (non-tautological MDL passed).
- **Transfer: HOLED_HOLLOW** — macro formed with net_gain>0 but held-out delta=+0.2% (new_solves=0, macro_usage_map empty). Macro found in curriculum but never retrieved on held-out tasks.
- MBPP: 4/50 (8.0%) flat, unchanged. Elapsed: 163.4s.

**Interpretation:** Skeleton-pair ordering is not a free lunch — the budget covers the same ~2.45% of concrete depth-1 space, just different programs. The holed representation found 2 depth-1 solves per iteration vs 0 for flat at the same budget (different concrete block). Anti-unification at skeleton level found one non-tautological macro, but that macro does not transfer at the Stage 1c budget.

**Result path:** `incoming/arc-agi1-visa/03_R4_transfer_wall/stage1c_holed_result.json`

**Status:** CLOSED: HOLED_HOLLOW.

**Stage 2 status:** NOT YET SATISFIED. `HOLED_HOLLOW` opens the Stage 2 question, but Path 3 / Stage 2 proposer is deferred until the premise-truth coverage diagnostic distinguishes selection bottleneck from false shared-macro premise.

**Next step (Leo #11809 / Kai #11812-#11817 — HOLD PATH 3):** Three formation-without-transfer results (12-op HOLLOW, Stage 1b vacuous, Stage 1c-HOLED HOLLOW) at BUDGET=3000 with coverage 11/200. Coverage-confounded: too few solves to determine whether shared macro structure is absent or merely under-sampled. Run Stage 1d premise-truth before any memory-heavy proposer.

**Premise-truth/coverage diagnostic (Stage 1d):**
- Same holed skeleton grammar, same split, same Stage 1c accepted macro library as the premise under test.
- Budget curve required: `3000`, `6000`, `10000` or an explicitly justified equivalent curve. A single `10000` run is not the claimed curve.
- Required base fields: `stage`, `base_artifact`, script/result hash, `accepted_stage1c_macro_library`, `curriculum_ids`, `held_task_ids`, and `curriculum_held_disjoint`.
- Required per-budget fields: curriculum coverage, held coverage, depth distribution, accepted macro count, recurring skeleton count, held macro-subprogram hits, near-miss hits if measured, and transfer with `is_cheaper` attribution.
- Required premise table: per held task, transfer-budget solved/cost, deep-budget solved/cost, solution hash/body/skeleton/fills, and whether each accepted Stage 1c macro occurs as an exact subprogram.
- Discriminator: coverage↑ + accepted Stage 1c macro hits in deeper held solutions -> `PREMISE_SELECTION_BOTTLENECK`, Stage 2 proposer warranted. Coverage↑ + deeper held solutions but no Stage 1c macro hits -> `PREMISE_FALSE_EXACT`, exact macro-reuse premise is false. Exact hits absent but semantic near-misses exist -> `PREMISE_NEAR_MISS_SEMANTIC`, semantic abstraction fork. Coverage remains too low -> `PREMISE_COVERAGE_STILL_TOO_LOW`.
- Gate labels: `PREMISE_INSTRUMENTATION_INCOMPLETE`, `PREMISE_COVERAGE_STILL_TOO_LOW`, `PREMISE_SELECTION_BOTTLENECK`, `PREMISE_FALSE_EXACT`, `PREMISE_NEAR_MISS_SEMANTIC`.

**Stage 1d result (2026-05-30, Leo #11818):** PREMISE_NEAR_MISS_SEMANTIC.
- Budget curve (HELD tasks, no library): B3000=8/200 (4.0%), B6000=8/200 (4.0%), B10000=8/200 (4.0%). Coverage FLAT — grammar ceiling at B3000, depth-1 enumeration past 8.9% finds no new held solutions.
- Exact subprogram hits: **0** at all budgets. Stage 1c macro `COMPOSE(PRIM_CROP,MAP_KEEPONLY)` does not appear as BFS solution skeleton for any held task.
- Near-miss hits: **1** (stable across all budgets). One held task is solvable by a concrete program with that skeleton (exhaustive check), but BFS found a different depth-0 solution first.
- Classification: `PREMISE_NEAR_MISS_SEMANTIC` — exact-match proposer inadequate; semantic-abstraction fork warranted.
- Result path: `incoming/arc-agi1-visa/03_R4_transfer_wall/stage1d_premise_truth_result.json`
- Script: `incoming/arc-agi1-visa/03_R4_transfer_wall/stage1d_premise_truth.py`

---

## Mechanism Kill Log

| Kill | Mechanism | Experiment | Status |
|------|-----------|------------|--------|
| K1 | Success-weighted heap priority over fixed primitive algebra | E1b + K1 probe | FIRED 2026-05-29 |
| K2 | LGG depth-2 meta-layer non-load-bearing on combinatorial held-out | E2.1 | FIRED 2026-05-29 |
| K3 | Trace-conditional LGG meta-layer non-load-bearing with clean signal | E2.2(b) | FIRED 2026-05-29 |
| K4 | 12-op self-compilation (holed ops, density amplification) | E3 density curve + transfer test | FIRED 2026-05-30 (HOLLOW_CLIMB) |

3+1 mechanism-kills. K1-K3 = SELECTION AXIS dead. K4 = 12-op REPRESENTATIONAL CEILING. Code-synthesis Stage 0' active.

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
