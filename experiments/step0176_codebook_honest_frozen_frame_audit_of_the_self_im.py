"""
Step 176 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10636.
"""
# Step 176: Honest frozen frame audit of the self-improving substrate
# List EVERY element that is fixed by design, not by data

print('=== FROZEN FRAME AUDIT ===')
print()
print('ADAPTIVE (determined by data):')
print('  - Stored exemplars (V): grows from data')
print('  - Labels: from supervision')
print('  - Feature weights (w): random, selected by LOO')
print('  - Feature biases (b): random, selected by LOO')
print('  - Number of features: determined by LOO convergence')
print('  - Feature template TYPE: selected from menu by LOO (cos/abs/sign/tanh/mod2)')
print('  - Distance metric: selected from {cosine, L2, L1} by LOO')
print()
print('FROZEN (fixed by design):')
print('  1. k=5 (top-k parameter)')
print('  2. LOO as scoring function (the ground truth test)')
print('  3. L2 normalization of augmented vectors')
print('  4. Template MENU: {cos, abs, sign, tanh, mod2} — finite, fixed set')
print('  5. Feature composition: f(w @ x + b) — always a function of a linear projection')
print('  6. Argmax aggregation (per-class score -> prediction)')
print('  7. Top-k sum aggregation (not mean, not max, not weighted)')
print('  8. Training/eval split (LOO uses training labels)')
print()

frozen = 8
adaptive = 7
total = frozen + adaptive
print(f'SCORE: {adaptive}/{total} adaptive = {adaptive/total*100:.0f}% resolved')
print(f'       {frozen}/{total} frozen')
print()
print('COMPARISON:')
print('  Living Seed:    2/8 adaptive (25%)')
print('  ANIMA:          1/8 adaptive (12.5%)')  
print('  This substrate: 7/15 adaptive (47%)')
print()
print('BINDING CONSTRAINTS (removing hurts):')
print('  - Template menu: cos alone gets 84% on parity, multi-template gets 98.4%')
print('  - k=5: k=1 loses 5pp on P-MNIST')
print('  - LOO scoring: the ONLY evaluation signal')
print()
print('NON-BINDING (removing doesnt hurt):')
print('  - L2 normalization: could use unnormalized')
print('  - Top-k sum vs mean: marginal difference')
print('  - Argmax: standard, no alternative tested')
" 2>&1
