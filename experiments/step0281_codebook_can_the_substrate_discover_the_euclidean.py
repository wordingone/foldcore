"""
Step 281 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 12444.
"""
import torch, torch.nn.functional as F, math
device = 'cuda'

# Step 281: Can the substrate discover the Euclidean step (a,b) -> (b, a%b)?
# Training: pairs showing one step of the algorithm
# Test: does the discovered step ITERATE to produce correct GCD on OOD?

# ONE STEP of Euclid: (a, b) -> (b, a mod b)
# Train on small numbers
n_train = 500
X = torch.zeros(n_train, 2, device=device)
y_next_a = torch.zeros(n_train, device=device, dtype=torch.long)  # next a = b
y_next_b = torch.zeros(n_train, device=device, dtype=torch.long)  # next b = a % b

for i in range(n_train):
    a = torch.randint(1, 20, (1,)).item()
    b = torch.randint(1, a+1, (1,)).item() if a > 0 else 1
    X[i, 0] = a; X[i, 1] = b
    y_next_a[i] = b
    y_next_b[i] = a % b

# k-NN to predict EACH output component
def knn_predict(X_tr, y_tr, query, n_cls, k=5):
    sims = F.normalize(query.unsqueeze(0), dim=1) @ F.normalize(X_tr, dim=1).T
    scores = torch.zeros(1, n_cls, device=device)
    for c in range(n_cls):
        m = y_tr == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return scores.argmax(1).item()

# Test: iterate the predicted step to compute GCD
# OOD test: a,b up to 50 (trained on 1-20)
print('Step 281: Discover Euclidean step from I/O, iterate for GCD')

n_cls = 20  # max value
correct = total = ood_correct = ood_total = 0

for a in range(1, 50, 3):
    for b in range(1, a+1, 3):
        # Iterate predicted step
        ca, cb = a, b
        steps = 0
        while cb > 0 and steps < 50:
            query = torch.tensor([float(ca), float(cb)], device=device)
            pred_a = knn_predict(X, y_next_a, query, n_cls)
            pred_b = knn_predict(X, y_next_b, query, n_cls)
            ca, cb = pred_a, pred_b
            steps += 1
        
        pred_gcd = ca
        true_gcd = math.gcd(a, b)
        ok = pred_gcd == true_gcd
        total += 1; correct += int(ok)
        if a > 20 or b > 20:
            ood_total += 1; ood_correct += int(ok)

print(f'  In-distribution (1-20): {correct-ood_correct}/{total-ood_total} ({(correct-ood_correct)/(total-ood_total)*100:.1f}%)')
print(f'  OOD (21-50):            {ood_correct}/{ood_total} ({ood_correct/ood_total*100:.1f}%)')
print(f'  Overall:                {correct}/{total} ({correct/total*100:.1f}%)')
print(f'  True GCD(48,18)=6, predicted={math.gcd(48,18)}')
" 2>&1
