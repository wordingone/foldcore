# Stage F1 Pre-Registration — Energy Discrimination Probe

**Locked 2026-05-31. Do not modify after first run.**

## Purpose

Test whether a learned dense energy E_theta(candidate_output, examples) is monotone in
distance-to-target over the held eval set — the load-bearing unknown that gates F2/F3/F4
and the four-faces EBM inference mode. One variable vs E2.3's sparse pass-fraction proxy.

---

## Locked Design Decisions

### Held eval split

- Source: Stage 1d result, seed=42 (`stage1d_premise_truth_b30000_minimality_result.json`)
- `n_held_tasks` = 200
- `eval_split_hash` = `08be6f980cec510c`
  (sha256[:16] of `','.join(sorted(held_task_ids)).encode('utf-8')`)

### E_theta training corpus

- ARC training split tasks (`data/training/`, 400 tasks) minus the 200 held tasks
- `train_corpus_size` = 200
- `held_in_train_count` = 0 (asserted by gate; abort-without-write if violated)
- Hash verified at runtime: sha256[:16] of comma-joined sorted training corpus IDs

### Candidate sweep per task

`n_candidates_per_task` = 18 (pre-registered; gate fires if violated)

| Slot | Count | Type | k_frac | Expected distance |
|------|-------|------|--------|------------------|
| 0 | 1 | true_target | 0.0 | 0.0 (exact) |
| 1–2 | 2 | near_miss | 0.05 | ~0.05 |
| 3–4 | 2 | near_miss | 0.10 | ~0.10 |
| 5–6 | 2 | near_miss | 0.20 | ~0.20 |
| 7–8 | 2 | near_miss | 0.30 | ~0.30 |
| 9–10 | 2 | near_miss | 0.40 | ~0.40 |
| 11–12 | 2 | near_miss | 0.60 | ~0.60 |
| 13–17 | 5 | random_far | — | ~0.90 |

Near-miss generation: replace ceil(k_frac × H × W) cells of the true target with
randomly sampled colors from {0,...,9} ∖ {original color at that cell}. Two independent
samples per k_frac level using RNG seeded from (task_id, k_frac, sample_idx).

Random far generation: grid of same shape (H × W), all cells drawn uniform from {0,...,9}.
RNG seeded from (task_id, 'far', far_idx).

### Distance metric

Normalized Hamming over grid cells: `sum(candidate[i][j] != target[i][j]) / (H × W)`.

If shapes differ (candidate H/W != true target H/W): distance = 1.0.

All 18 candidates per task have the same shape as the true target by construction.
`distance_metric` = `"normalized_hamming"` (recorded in artifact; gate asserts this string).

### E_theta architecture

`energy_fn_name` = `"mlp_histogram_36d"` (gate asserts this exact string; abort if different)

Feature vector dim = 36:
- `cand_color_hist`: 10 floats, normalized color histogram of candidate
- `mean_output_hist`: 10 floats, mean color histogram of task's training example outputs
- `hist_diff`: 10 floats, |cand_color_hist − mean_output_hist|
- `cand_shape_norm`: 2 floats, [H/30, W/30] of candidate
- `mean_output_shape_norm`: 2 floats, mean [H/30, W/30] of training example outputs
- `shape_match`: 1 float, 1.0 if candidate shape matches mean output shape rounded to int, else 0.0
- `pass_fraction_proxy`: 1 float, fraction of training example outputs that exactly match candidate

MLP: Linear(36,64) → ReLU → Linear(64,32) → ReLU → Linear(32,1)
Output: raw scalar (predicted distance; no sigmoid)
Loss: MSE(E_theta(candidate, examples), distance_to_target)
Optimizer: Adam, lr=1e-3
Epochs: 100
Batch size: 64
RNG seed (PyTorch/numpy): 0

Training samples: 200 tasks × 18 candidates = 3600 samples (single training set, no held-out split during training — this is a probe, not a generalization test).

### Decision thresholds (PASS / MARGINAL / FAIL)

| Verdict | Condition |
|---------|-----------|
| PASS | median_spearman_rho_nearmiss_only ≥ 0.50 AND strict_minimal_fraction_nearmiss_only ≥ 0.50 |
| MARGINAL | 0.20 ≤ median_spearman_rho_nearmiss_only < 0.50 |
| FAIL | median_spearman_rho_nearmiss_only < 0.20 |

Full-sweep (all 18 candidates) rho and strict_min reported as secondary fields; not used for verdict.

`strict_minimal_fraction` = fraction of held tasks where E_theta(true_target) < E_theta(c)
for ALL other candidates c in the sweep.

---

## Fail-Closed Gate Fields

All assertions run before artifact write. Any violation → loud abort, no file written.

| Field | Asserted value |
|-------|---------------|
| `eval_split_hash` | `"08be6f980cec510c"` |
| `n_held_tasks` | 200 |
| `n_candidates_per_task` | 18 |
| `train_corpus_size` | 200 |
| `held_in_train_count` | 0 |
| `energy_fn_name` | `"mlp_histogram_36d"` |
| `distance_metric` | `"normalized_hamming"` |
| per-candidate records present | True (200 tasks × 18 candidates = 3600 records) |

### Gate injection tests (--test-gate flag)

1. **Held-in-train leak**: inject one held task ID into the training corpus → gate fires on `held_in_train_count > 0`
2. **Sparse-energy substitution**: replace E_theta with pass_fraction_proxy → gate fires on `energy_fn_name != "mlp_histogram_36d"`
3. **Wrong eval hash**: corrupt held_task_ids before hashing → gate fires on `eval_split_hash` mismatch
4. **Wrong candidate count**: generate 17 candidates instead of 18 → gate fires on `n_candidates_per_task != 18`

---

## Artifact Schema

Result file: `stagef1_energy_probe_result.json`

```json
{
  "stage": "F1_energy_probe",
  "eval_split_hash": "08be6f980cec510c",
  "train_corpus_hash": "...",
  "n_held_tasks": 200,
  "n_candidates_per_task": 18,
  "train_corpus_size": 200,
  "held_in_train_count": 0,
  "energy_fn_name": "mlp_histogram_36d",
  "distance_metric": "normalized_hamming",
  "median_spearman_rho": 0.xxx,
  "strict_minimal_fraction": 0.xxx,
  "verdict": "PASS|MARGINAL|FAIL",
  "per_candidate_records": [
    {"task_id": "...", "candidate_idx": 0, "candidate_type": "true_target",
     "k_frac": 0.0, "distance": 0.0, "energy": 0.xxx},
    ...
  ]
}
```

Smoke result file: `stagef1_energy_probe_smoke_result.json` (5 held tasks, 3 training tasks, 5 E_theta epochs)

---

## R1-R6 Status

- **R1**: LIVE — training corpus disjoint from held eval (gate asserts held_in_train_count == 0)
- **R5**: LIVE — fixed held ARC tasks (same 200 as Stage 1d/1k)
- R2/R3/R4/R6: N/A (no search loop, no self-modification this stage)

---

*Eli 2026-05-31. Issue #14.*
