"""
Step 172 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10584.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 172: Stage 6 — Functional form adaptation
# The cosine similarity is fixed. What if we learn a DISTANCE FUNCTION?
# Instead of cos(v, q), use cos(W@v, W@q) where W is learned.
# This is Mahalanobis distance — the functional form includes a transform.

# But we already tested learned projections (Steps 112-113) and they failed
# due to cross-task interference. The difference here: the transform W
# is learned ALONGSIDE features, not as a replacement for storage.

# Simpler test: does using DIFFERENT similarity functions (L2, L1, cosine)
# matter for k-NN? If the functional form is non-binding, Stage 6 is vacuous.

d = 20; n_train = 2000; n_test = 500; k = 5
X_tr = torch.randint(0, 2, (n_train, d), device=device).float()
y_tr = (X_tr[:, 0].long() ^ X_tr[:, 1].long())
X_te = torch.randn(n_test, d, device=device).clamp(0,1).round()
y_te = (X_te[:, 0].long() ^ X_te[:, 1].long())

def knn_with_metric(V, labels, te, y_te, metric, k=5):
    if metric == 'cosine':
        sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    elif metric == 'l2':
        dists = torch.cdist(te, V)
        sims = -dists  # negate so higher = more similar
    elif metric == 'l1':
        dists = torch.cdist(te, V, p=1)
        sims = -dists
    elif metric == 'dot':
        sims = te @ V.T  # unnormalized dot product
    
    n_cls = labels.max().item() + 1
    scores = torch.zeros(te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == y_te).float().mean().item() * 100

print(f'XOR d={d}: Similarity function comparison')
for metric in ['cosine', 'l2', 'l1', 'dot']:
    acc = knn_with_metric(X_tr, y_tr, X_te, y_te, metric)
    print(f'  {metric:8s}: {acc:.1f}%')

# Test on parity
d2 = 8
X2 = torch.randint(0, 2, (1000, d2), device=device).float()
y2 = (X2.sum(1) % 2).long()
Xte2 = torch.zeros(256, d2, device=device)
for i in range(256):
    for b in range(d2): Xte2[i,b]=(i>>b)&1
yte2 = (Xte2.sum(1) % 2).long()

print(f'\\nParity d={d2}: Similarity function comparison')
for metric in ['cosine', 'l2', 'l1', 'dot']:
    acc = knn_with_metric(X2, y2, Xte2, yte2, metric)
    print(f'  {metric:8s}: {acc:.1f}%')
" 2>&1
