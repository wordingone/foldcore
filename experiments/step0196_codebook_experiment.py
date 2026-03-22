"""
Step 196 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11009.
"""
print('=== UPDATED FROZEN FRAME AUDIT (Step 196) ===')
print()
print('ADAPTIVE (determined by data/LOO):')
print('  1. Stored exemplars (V): grows from data')
print('  2. Labels: from supervision or self-prediction (S1)')
print('  3. Feature weights (w): random, selected by LOO')
print('  4. Feature biases (b): random, selected by LOO')
print('  5. Number of features: determined by LOO convergence')
print('  6. Template TYPE per feature: selected by LOO (rho=0.90, Step 194)')
print('  7. Distance metric: selected by LOO (L2 vs cosine vs L1, Step 173)')
print('  8. Iteration count: determined by target matching (Step 195)')
print('  9. Self-supervised spawning during eval (S1, Step 106)')
print()
print('FROZEN (fixed by design):')
print('  1. LOO as scoring function (ground truth - constitutionally required)')
print('  2. Template FAMILY set: {cos, abs, sign, tanh, mod2}')
print('  3. Feature composition: f(w @ x + b) - single linear projection + nonlinearity')
print('  4. k=5 for top-k vote')
print('  5. L2 normalization of augmented vectors')
print('  6. Argmax + top-k-sum aggregation')
print()

adaptive = 9
frozen = 6
total = adaptive + frozen
print(f'SCORE: {adaptive}/{total} adaptive = {adaptive/total*100:.0f}% resolved')
print(f'       {frozen}/{total} frozen')
print()
print('COMPARISON across all substrates:')
print(f'  Living Seed:              2/8   = 25%')
print(f'  ANIMA:                    1/8   = 12.5%')
print(f'  Self-Improving (Step 176): 7/15  = 47%')
print(f'  Self-Improving (Step 196): {adaptive}/{total}  = {adaptive/total*100:.0f}%')
print()
print('BINDING frozen elements (removing hurts):')
print('  - LOO scoring: constitutionally frozen (Principle V)')
print('  - Template family: determines ceiling per task type (Step 193)')
print('  - f(w@x+b) form: limits to single-layer nonlinearities')
print()
print('NON-BINDING (removing doesnt hurt):')
print('  - k=5: Step 117 showed k not binding')
print('  - L2 normalization: convention')
print('  - Aggregation method: marginal differences')
" 2>&1
