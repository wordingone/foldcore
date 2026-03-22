"""
Step 285 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 12537.
"""
import torch, math
device = 'cuda'

# Step 285: Full coverage one-hot — does it reach 100%?
n = 15  # smaller for full coverage

def one_hot_pair(a, b, n=15):
    v = torch.zeros(2*n, device=device)
    v[min(a,n-1)] = 1; v[n+min(b,n-1)] = 1
    return v

# FULL COVERAGE: all (a,b) pairs
X = []; y = []
for a in range(1, n):
    for b in range(1, n):
        for _ in range(3):  # 3 copies each
            X.append(one_hot_pair(a, b, n)); y.append(a % b)
X = torch.stack(X); y = torch.tensor(y, device=device, dtype=torch.long)

def predict_mod(a, b):
    query = one_hot_pair(a, b, n)
    sims = query @ X.T
    scores = torch.zeros(n, device=device)
    for c in range(n):
        m = y == c; cs = sims[m]
        if cs.shape[0] == 0: continue
        scores[c] = cs.topk(min(5, cs.shape[0])).values.sum()
    return scores.argmax().item()

correct = total = 0
for a in range(2, n):
    for b in range(1, a):
        ca, cb = a, b
        steps = 0
        while cb > 0 and steps < 50:
            pred = predict_mod(ca, cb)
            ca, cb = cb, pred
            steps += 1
        if ca == math.gcd(a, b): correct += 1
        total += 1

print(f'Step 285: Full coverage one-hot GCD (n={n})')
print(f'  Coverage: all {(n-1)**2} (a,b) pairs')
print(f'  GCD accuracy: {correct}/{total} ({correct/total*100:.1f}%)')
" 2>&1
