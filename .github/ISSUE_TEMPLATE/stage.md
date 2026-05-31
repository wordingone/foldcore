---
name: Experiment stage
about: One one-variable experiment stage (1a→1k→F1→F2…). Binding pre-registration.
title: "Stage <N> — <one-line question>"
labels: ["stage", "gate:pending"]
---

## Why
<prior result + the confound/question motivating this stage. Cite the prior stage's result commit/issue.>

## One variable
<the SINGLE thing that changes vs the prior stage. Everything else is held constant + asserted.>

## Pre-registration (lock before any run)
- **Held eval split:** <held set; `eval_split_hash` asserted == Stage-1d held hash>
- **Held constants:** budget, depth, seeds, time wall, search algorithm, grammar/skeleton — list each
- **`EXPECTED_PREV_SPACE_HASH`:** <prior stage's `space_hash`>
- **Frozen artifacts (if any):** <e.g. frozen energy weights — assert hash, no retraining>

## Metric (pre-registered)
<exact measurement + aggregation. The primary gate metric, stated unambiguously.>

## Decision tree (written BEFORE running)
- **PASS** — <condition> → <next stage>
- **FAIL** — <condition> → <honest report + what face/direction it kills>
- **PARTIAL** — <condition> → <one follow-up variable; do NOT advance>

## Fail-closed gate (assert before artifact write; injection test must catch each)
- <field == expected value> (+ inject a violation → gate must fire)
- budget ENFORCED: max per-task n_evals ≤ budget (config-asserted ≠ enforced)
- held / disjointness: `held_in_train_count == 0`
- abort-without-write on any violation (loud)

## R1–R6
<which rules are live vs N/A + why. Pure reach/measurement with a frozen tool → R1+R5 live, R2/R3/R4/R6 N/A.>

## Acceptance
- [ ] Pre-reg file committed (constants + thresholds + hashes locked)
- [ ] Harness on branch + PR (Leo L1 diff-review: ONLY the one-variable delta + gate fields)
- [ ] TEMP-smoke: fail-closed gate fires on each injected violation (Leo L2)
- [ ] Canonical result JSON with raw per-unit records committed to `main`
- [ ] Leo integer-recount from raw records, not a reported scalar (L3)
- [ ] R1/R5 (+R2/R3/R4/R6 if self-modifying) audit (L4) → verdict → next stage or honest report

## Owner
<eli | leo>  •  Parent direction: #
