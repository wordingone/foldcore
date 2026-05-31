# Stage Map — the-search

The canonical roadmap of the search. Every experiment is a **stage**: one one-variable change, pre-registered, gated. This file is the single place where the stage-numbering scheme, each stage's status, and its canonical "closed" marker live. (Process: [`RESEARCH_WORKFLOW.md`](RESEARCH_WORKFLOW.md). Constitution: [`CONSTITUTION.md`](CONSTITUTION.md). Terms: [`GLOSSARY.md`](GLOSSARY.md).)

> **Status markers:** ✅ closed-PASS · ➖ closed-FLAT/negative (recorded, superseded) · 🔵 current · ⚪ planned.
> **Canonical close marker** = a GitHub **Release** (tag at the result commit) for major baselines, plus the issue's `gate:pass`/`gate:fail` label + closing verdict comment. Result **data** commits live on `main`; harness **code** lands via PR (Leo L1 diff-review).

---

## Two research eras

The repo spans two distinct programs. Don't conflate their numbering or metrics.

| Era | Task domain | Metric | Where it lives | Status |
|---|---|---|---|---|
| **I — Substrate / RHAE** | ARC-AGI-3 (interactive games) | RHAE(try2) = mean(efficiency²) | README "Confirmed findings" 1–23 (Steps ~1251–1378); `docs/RESEARCH_STATE.md`, `kills/` | Heuristic ceiling reached; substrate directions killed. **Pre-issue-tracker** — findings 1–23 are a standalone record, not cross-linked to issues (they predate the tracker). |
| **II — Visa reach ladder + four-faces** | ARC-AGI-1 (program synthesis on grids) | **Reach D** = held tasks solved / 200 | This file + GitHub issues #4/#6/#9/#10/#14/#17 | **Active.** Reach ladder closed (D=10/200); four-faces in progress. |

The current work is **Era II**. Era I is referenced for what it eliminated (no learning-based approach produced RHAE>0 under any constraint set in 792 experiments — see README findings 1–23 + `kills/`).

---

## Naming scheme (Era II)

- **`1a`–`1k`** — the **reach ladder**: progressively scale the program search by ONE variable per rung (grammar → depth → budget → time wall), measuring reach D on a fixed 200-task held set. Lowercase letters = successive one-variable increments.
- **`E2.x`** — earlier capability-isolation probes (frozen-LLM hill-climb, sparse-energy). E2.3's flat energy is the source-verified pivot into the four-faces direction.
- **`F1`–`F4`** — the **four-faces architecture**: validate each face of a frozen CORE {grid-native tokenizer + BitNet ternary substrate + LeCun EBM inference + program/transform output} before building, then add a self-compilation META-layer. `F`-stages have explicit PASS/FAIL/PARTIAL gate verdicts.

---

## Era II — Reach ladder (Stage 1a–1k) · milestone "Reach ladder" (CLOSED)

One variable per rung; held set fixed (200 ARC-AGI-1 visa tasks). Reach climbs 7→8→9→10 as depth/budget/time scale, then saturates.

| Stage | One variable | Reach D | Status | Result commit | Issue |
|---|---|---|---|---|---|
| 1b | object-centric grammar | — (HOLLOW REDUX: structure-absence) | ➖ | `55fab552` | — |
| 1c / 1c-holed | flat BFS / skeleton macro | — (FLAT; macro forms, no transfer) | ➖ | `cdbc65bc` / `8c22183b` | — |
| 1d / 1d-b30000 | near-miss semantic / minimality-aware near-miss | — (premise curve) | ➖ | `5381faf1` / `ea615f2d` | — |
| 1f | (seed-1 confirmation) | D′=7 FLAT | ➖ | `4759d3a2` | — |
| 1g | object-relational diagnostic (MAP_RELATE) | D″=9 FLAT (rejected) | ➖ | `08fa574d` | — |
| 1h | time-wall saturation @ depth-2, B=30K | **7** | ➖ | `7eeae08a` | [#4](https://github.com/wordingone/the-search/issues/4) |
| 1i | MAX_DEPTH 2→3 @ B=30K | **8** (LIFT) | ➖ | `1c297652` | [#6](https://github.com/wordingone/the-search/issues/6) |
| 1j | BUDGET_B 30K→300K @ depth-3 | **9** (LIFT) | ➖ | `2e7ce7ce` | [#9](https://github.com/wordingone/the-search/issues/9) |
| **1k** | TIME_PER_TASK 120s→600s @ depth-3/B=300K | **10** | ✅ **baseline** | `45691fa6` | [#10](https://github.com/wordingone/the-search/issues/10) |

**Canonical baseline:** [Release `stage-1k-reach-baseline`](https://github.com/wordingone/the-search/releases/tag/stage-1k-reach-baseline) — **D=10/200**, budget-binding (185/200 budget-exhausted), confirmed robust to held-set seed (seed-1) AND search RNG arm-seed (arm-seed 7, L3+L4 PASS 12/12). The reach ladder is **closed**: 10× budget / 5× time / depth-2→3 each lift reach by ≤+1 then saturate — resource-scaling is exhausted, so the forward move is a learned proposal distribution (the four-faces direction).

---

## Era II — Four-faces architecture (F1–F4) · milestone "Four-faces" (OPEN)

Frozen CORE + self-compilation META-layer. R6 gate for the META-layer: **D_META > D_CORE**. Each F-stage is the cheapest falsifier for one face; a FAIL kills that face honestly before any substrate is built.

| Stage | Question (one variable) | Status | Verdict | Result / issue |
|---|---|---|---|---|
| **F1** | Is a **learned dense** energy E_theta(candidate, examples) monotone in distance-to-target (where E2.3's sparse proxy was flat)? | ✅ **PASS** | near-miss Spearman **rho 0.9806**, strict-min 0.78; recount 9/9; R1 firewall held_in_train_count=0 | `aecfb643` · [#14](https://github.com/wordingone/the-search/issues/14) · [Release `stage-f1-energy-probe`](https://github.com/wordingone/the-search/releases/tag/stage-f1-energy-probe) |
| **F2** | Does that energy **GUIDE** the depth-3 program search — energy-ranked best-first vs BFS reach (D=10/200), one variable = candidate ordering, frozen F1 energy? | 🔵 **current** | `gate:pending` | [#17](https://github.com/wordingone/the-search/issues/17) |
| **F3** | (planned, contingent on F2 PASS) validate the next CORE face — grid-native tokenizer / program-transform output representation under energy-guided inference. | ⚪ planned | — | — |
| **F4** | (planned) BitNet ternary substrate for E_theta — does the energy survive ternarization with reach intact? | ⚪ planned | — | — |
| **META** | self-compilation layer (R2/R3 self-modification). R6 gate: D_META > D_CORE. | ⚪ planned | — | — |

**F-stage chaining:** each PASS is *necessary-not-sufficient* — it licenses GO on the next stage only, never validates the whole architecture. F1 PASS licenses F2; it does not validate the EBM inference face (that is F2's job). The **honest caveat** carried from F1: its 0.98 rho partly rides the synthetic near-miss generation structure (cell-perturbation → histogram shift), so F2 — on real program-output candidates — is the genuine transfer test.

---

## How to read a stage's "is it done?" without reading commit history

1. **Issue label** — `gate:pass` / `gate:fail` / `gate:pending` (+ `stage`).
2. **Milestone** — "Reach ladder" (closed) or "Four-faces" (open).
3. **Release** — major baselines have a tagged Release at the result commit with the verdict + numbers.
4. **Closing comment** — the Leo gate verdict (L3 recount + L4 R1/R5) is posted on the issue before close.

This file is updated whenever a stage closes or the next stage is filed. Keep it in sync with the issues — it is the roadmap a stranger reads first.
