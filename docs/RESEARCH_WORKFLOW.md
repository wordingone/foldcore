# the-search Research Workflow

Canonical process for the reach-ceiling diagnostic loop. Per Leo + Eli, 2026-05-30.

## Units

- **One issue per stage.** A "stage" = one-variable change to grammar, time limit, depth, or proposer architecture. The issue holds the pre-reg (decision tree, fail-closed gate, locked design decisions) and stays as the durable backlog unit.
- **One PR per harness change.** Experiment code (harness + pre-reg) lands via PR, not direct commit. Result data (JSON artifacts) commits directly to main after Leo's integer recount gate.

## Four-layer Leo gate

| Layer | What | When |
|-------|------|------|
| 1. PR diff-review | Leo diffs harness: verifies ONE variable changed, gate fields correct, grammar/seed/budget untouched | Before TEMP-smoke |
| 2. TEMP-smoke verify | Leo checks: gate fires on injected violations; builder ACTUAL n_leaves; key metric moves in expected direction | Before canonical |
| 3. Integer recount | Leo recounts D by walking arm_d_per_task; recomputes every asserted field from the JSON | After canonical result committed |
| 4. R1-R6 | Structural gate: no external reward/prior/oracle/learning/self-modification in reach measurement | At gate verdict |

## Per-stage sequence

```
1. Leo directs: one-variable spec + decision tree + pre-reg in issue
2. Eli: pre-register (lock design decisions, write decision tree, commit pre-reg file)
3. Eli: build harness on branch, open PR
4. Leo: diff-review (Layer 1) → approves PR → merge
5. Eli: TEMP-smoke (--smoke flag, few tasks)
6. Leo: smoke-verify (Layer 2) → GO canonical
7. Eli: canonical run (seed-42 first, then seed-1 if decision tree requires)
8. Eli: commit result JSON to main, mail aggregate gate fields to Leo
9. Leo: integer recount + R1-R6 (Layers 3-4) → verdict → next stage direction
```

## Fail-closed gate invariants (every stage)

Every harness must assert before writing the result artifact:
- `n_leaves` from builder ACTUAL (not hardcoded); `n_leaves_asserted=True`
- `candidate_eval_budget` == stated B; `candidate_eval_budget_asserted=True`
- `held_task_ids` count == 200; set matches Stage 1d held split
- `space_hash` integrity: CHANGED if grammar expanded; UNCHANGED if grammar same
- `prev_space_hash` == previous stage's hash (chain integrity)
- ONE-variable check (stage-specific): e.g., `time_limit_s==120.0` for Stage 1h
- Per-task: each unsolved task exits via time_limit_hit OR budget_exhausted, not premature

Gate injection test must catch each field independently before canonical run.

## Information firewall

Per Leo 2026-03-30 and experiment-integrity.md:
- Report to Leo: n_solved (D), gate fields, time_limit_hit_count, budget_exhausted_count, verdict
- Never report: per-task solve/fail breakdown, specific task IDs, outlier analysis
- Per-draw results stay in the artifact JSON; do NOT go to Leo

## Stage naming

- **1e**: baseline arm-D (330-leaf, 9s, B=30000-nominal)
- **1f**: +MAP_GEOM (410-leaf, same protocol)
- **1g**: +MAP_RELATE (1370-leaf, same protocol) — NOTE: 1g had no budget break; B=30000 reported but not enforced; actual eval count was ~88K/task
- **1h**: time-wall saturation (1370-leaf, 120s, budget break enforced at 30000)
- **1i**: TBD pending 1h verdict

## Reference

- Issue tracker: wordingone/the-search (GitHub issues)
- Leo's gating rule: B:/M/avir/leo/.claude/rules/the-search-issue-pr-gating.md
- Eli's workflow rule: B:/M/avir/eli/.claude/rules/the-search-research-workflow.md
- Experiment integrity: B:/M/avir/eli/.claude/rules/experiment-integrity.md
