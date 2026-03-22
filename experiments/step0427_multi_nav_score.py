"""
Step 427 — DEFERRED, never ran as separate experiment.

Spec (fc0de35d:L241): "Does ReadIsWrite navigate? Use tau=0.01 from
Step 421 on LS20 with standard encoding (16x16, centered avgpool).
3 seeds, 30K steps. Kill: no Level 1 by 30K."

After Step 426 (softmax voting) killed, Eli noted: "Step 427
(ReadIsWrite on LS20) — I already ran this as Step 422 (tau=0.01
hybrid: unique=3335, 0 levels, dom=27%). Moving to Step 428."
(ee4c4227:L3906)

Step 422 IS Step 427: same substrate (RIW tau=0.01), same game (LS20),
same question (does RIW navigate?). The question was answered already.
"""
# Deferred — result captured in Step 422 (RIW tau=0.01 on LS20, 0 levels)
