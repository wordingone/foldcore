# Glossary — the-search

Terms used across issues, commits, the README, and [`STAGE_MAP.md`](STAGE_MAP.md). If a term in an issue or commit is unexplained, it should be here.

## Eras & numbering

- **Era I (Substrate / RHAE)** — ARC-AGI-3 interactive-game work; README findings 1–23; Steps ~1251–1378. Killed at the heuristic ceiling.
- **Era II (Visa reach ladder + four-faces)** — current; ARC-AGI-1 program synthesis on grids.
- **Stage `1a`–`1k`** — the reach ladder (Era II). Lowercase letter = one successive one-variable increment.
- **`E2.x`** — capability-isolation probes (frozen-LLM hill-climb; sparse-energy). E2.3's flat energy is the pivot into four-faces.
- **`F1`–`F4`** — the four-faces stages, each the cheapest falsifier for one CORE face. PASS/FAIL/PARTIAL verdicts.

## Metrics

- **RHAE** (Era I) — Relative Human Action Efficiency. `RHAE(try2) = mean(efficiency²)` across games, efficiency per level = (human_actions / agent_actions)². The Era-I metric. Best R2-compliant substrate ~1e-5..1e-4 (draw-variance-dominated).
- **Reach D** (Era II) — number of held tasks solved out of the 200-task held set. The reach ladder's metric. Current baseline **D=10/200**.
- **Spearman rho (F1)** — rank correlation between predicted energy and true distance-to-target. A good energy → rho near +1 (energy predicts distance; target = minimum). F1 gates on the **near-miss-only** rho (≥0.50 PASS, <0.20 FAIL), not the full sweep.
- **strict-minimal fraction** — fraction of held tasks where E_theta(true target) is strictly the lowest energy over the candidate sweep (exact-basin check).

## Four-faces architecture

- **CORE** — the frozen substrate: {grid-native tokenizer + BitNet ternary weights + LeCun EBM inference + program/transform output}. Built only after each face passes its F-stage falsifier.
- **META** — the self-compilation layer that modifies the system; R2/R3 self-modification lives here. **R6 gate: D_META > D_CORE** (the meta-layer must beat the frozen core, else it is deletable).
- **EBM / energy-based model** — inference framed as optimization: define an energy E_theta(candidate, examples) low at good candidates, and *infer = minimize energy* over the output space (LeCun's "inference is optimization"). Replaces a forward-pass classifier with a search on the energy landscape.
- **inference-by-optimization** — the EBM inference mode: instead of predicting an answer, search/optimize the output to minimize energy. F2 tests it in program/transform space.

## Constitution (R1–R6) — see [`CONSTITUTION.md`](CONSTITUTION.md)

- **R1** — no external loss/reward/metric drives learning (self-supervised only; training corpus disjoint from held — no answer leak).
- **R2** — the update signal IS the computation, not a separate optimizer (Adam/SGD = R2 violation; local Hebbian/LPL/MDL-compression = R2-compliant).
- **R3** — the system modifies itself AND the modification changes behavior (weight-drift > 0 is necessary, not sufficient).
- **R4** — modification tested against the prior state (second-exposure / wall-invariance).
- **R5** — one fixed ground truth (the task/held set).
- **R6** — no deletable component: deleting any part must degrade behavior, else it isn't earning its place. (For META: D_META > D_CORE.)
- For a pure reach/measurement stage with a frozen tool, **R1 + R5 are live; R2/R3/R4/R6 are N/A** (no self-modification).

## The Leo gate (4 layers) — see [`RESEARCH_WORKFLOW.md`](RESEARCH_WORKFLOW.md)

- **L1** — PR diff-review before the canonical run: ONLY the declared one-variable delta moved; fail-closed gate present with correct `EXPECTED_PREV_SPACE_HASH`.
- **L2** — TEMP-smoke: the fail-closed gate fires on each injected violation (one injection per field).
- **L3** — canonical result integer-recount: recompute the metric from raw per-unit records (never trust a reported scalar/boolean); assert every fail-closed field at the integer level, incl. **budget ENFORCED** (max per-task n_evals ≤ budget — config-asserted ≠ enforced).
- **L4** — reproducibility / set-diff + R1–R6 constitutional audit.

## Gate / harness terms

- **Fail-closed gate** — the harness asserts all pre-registered invariants *before writing the artifact* and aborts-without-write (loud) on any violation. A missing artifact is loud; a mislabeled one is silent — so never trust a boolean `matched=true`, verify the integer.
- **One variable** — exactly one thing changes per stage vs the prior; everything else (budget, depth, seeds, held split, grammar) held constant + asserted.
- **Held set / `eval_split_hash`** — the fixed 200 ARC-AGI-1 visa tasks reserved for evaluation; the hash pins the exact split (asserted == Stage-1d held hash). Training corpus must be disjoint (`held_in_train_count == 0`).
- **Per-task seeding / `arm-seed`** — each task's search RNG seeded from `(arm_seed, task_id)` so a stage-level change can't shift another task's stream (the Stage-1e RNG-coupling Class-C bug). `--arm-seed N` decouples the search RNG from the held-set seed → lets us test search-seed robustness on a *fixed* held set.
- **`space_hash` / `n_leaves`** — fingerprint + size of the program/transform search space; asserted equal to the prior stage's when only a non-space variable (budget/time) changes.
- **near-miss / random-far (F1 sweep)** — per held task: the true target (distance 0) + 12 near-miss candidates (target perturbed by k cells) + 5 random-far grids. The near-miss-only cut is the load-bearing regime (random-far is trivially rankable and inflates the full-sweep rho).

## Other

- **PRISM** — masked-prompt evaluation protocol (MBPP always present + random masked ARC games; no draws, no control) — see `RESEARCH_WORKFLOW.md` / `research-discipline`.
- **BitNet** — ternary-weight ({-1,0,+1}) network; the CORE's substrate (Face 2), tested only after the energy principle holds (F4).
- **PASS is necessary-not-sufficient** — a stage PASS licenses GO on the *next* stage only; it never validates the whole architecture.
