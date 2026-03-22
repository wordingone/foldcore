"""
Step 287 — SKIPPED by Leo directive (2026-03-15).

Context: Step 286 tested 4 encodings (raw, one-hot, binary, thermometer) for
a%b k-NN discoverability. No encoding achieved LOO >= 75%.
Thermometer best at 41.8%. Binary best OOD at 13%.

Leo mail (91a3a472:L595): "Step 286 confirms: modular arithmetic is
undiscoverable via encoding alone. No encoding makes a%b Lipschitz-continuous
enough for k-NN. Skip Step 287 (GCD iteration with bad step prediction is
pointless). Go directly to Step 288."

Step 287 was planned: "With the best encoding from 286, does iteration
produce correct GCD on OOD (a,b > training range)?" Without a working
step predictor, iteration is meaningless. This step was never run.
"""
# Intentionally skipped — see step0286_* and step0288_*
