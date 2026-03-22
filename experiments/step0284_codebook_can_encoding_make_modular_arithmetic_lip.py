"""
Step 284 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 12534.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 284: Can ENCODING make modular arithmetic Lipschitz-continuous?
# Raw integers: a%b is discontinuous (7%3=1, 8%3=2, 9%3=0 — wraps)
# Binary: a%b requires carry chains (Step 242 showed feature discovery fails)
# ONE-HOT: every state is orthogonal (Step 185 showed this works for Fibonacci)
# Can one-hot make a%b k-NN-discoverable?

def one_hot_pair(a, b, n=20):
    v = torch.zeros(2*n, device=device)
    v[min(a, n-1)] = 1
    v[n + min(b, n-1)] = 1
    return v

n = 20  # support values 0-19
n_train = 800

# Training: one-hot encoded (a, b) -> a%b
X = []; y = []
for _ in range(n_train):
    a = torch.randint(1, n, (1,)).item()
    b = torch.randint(1, n, (1,)).item()
    X.append(one_hot_pair(a, b, n))
    y.append(a % b)
X = torch.stack(X); y = torch.tensor(y, device=device, dtype=torch.long)

# Test all pairs
Xte = []; yte = []
for a in range(1, n):
    for b in range(1, n):
        Xte.append(one_hot_pair(a, b, n))
        yte.append(a % b)
Xte = torch.stack(Xte); yte = torch.tensor(yte, device=device, dtype=torch.long)

n_cls = n

def knn(V, labels, te, yte, n_cls, k=5):
    sims = te @ V.T  # one-hot, no normalize needed (already binary)
    scores = torch.zeros(te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == yte).float().mean().item() * 100

acc = knn(X, y, Xte, yte, n_cls)

# Now: can we ITERATE one-hot k-NN for GCD?
import math

def predict_mod(a, b, X_db, y_db, n, k=5):
    query = one_hot_pair(a, b, n)
    sims = query @ X_db.T
    scores = torch.zeros(n, device=device)
    for c in range(n):
        m = y_db == c; cs = sims[m]
        if cs.shape[0] == 0: continue
        scores[c] = cs.topk(min(k, cs.shape[0])).values.sum()
    return scores.argmax().item()

# Iterate for GCD
correct = total = 0
for a in range(2, n):
    for b in range(1, a):
        ca, cb = a, b
        steps = 0
        while cb > 0 and steps < 50:
            pred_mod = predict_mod(ca, cb, X, y, n)
            ca, cb = cb, pred_mod
            steps += 1
        pred_gcd = ca
        true_gcd = math.gcd(a, b)
        if pred_gcd == true_gcd: correct += 1
        total += 1

print(f'Step 284: One-hot encoding for emergent GCD')
print(f'  a%b per-step accuracy: {acc:.1f}%')
print(f'  GCD via iteration: {correct}/{total} ({correct/total*100:.1f}%)')
print(f'  Does encoding solve the discontinuity? {\"YES\" if correct/total > 0.9 else \"PARTIALLY\" if correct/total > 0.5 else \"NO\"} ({correct/total*100:.1f}%)')
" 2>&1
