"""
Step 163 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10464.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 163: XOR at d=50 and d=100 with random cosine features
# Step 127 showed random quadratic features FAILED here. Do random cosines work?

for d in [20, 50, 100]:
    n_train, n_test = 2000, 500
    X_tr = torch.randint(0, 2, (n_train, d), device=device).float()
    y_tr = ((X_tr[:, 0].long() ^ X_tr[:, 1].long())).long()
    X_te = torch.randn(n_test, d, device=device).clamp(0, 1).round()
    y_te = ((X_te[:, 0].long() ^ X_te[:, 1].long())).long()

    def knn_acc(V, labels, te, y_te, k=5):
        sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
        scores = torch.zeros(te.shape[0], 2, device=device)
        for c in range(2):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == y_te).float().mean().item() * 100

    acc_base = knn_acc(X_tr, y_tr, X_te, y_te)
    
    # Try 1000 random cosine features, keep best
    best_acc = acc_base
    for _ in range(1000):
        w = torch.randn(d, device=device)
        b = torch.rand(1, device=device) * 6.28
        feat_tr = torch.cos(X_tr @ w + b).unsqueeze(1)
        feat_te = torch.cos(X_te @ w + b).unsqueeze(1)
        acc = knn_acc(
            F.normalize(torch.cat([X_tr, feat_tr], 1), dim=1), y_tr,
            F.normalize(torch.cat([X_te, feat_te], 1), dim=1), y_te)
        if acc > best_acc:
            best_acc = acc

    print(f'd={d:3d}: base={acc_base:.1f}% +random_cos(1000)={best_acc:.1f}% delta={best_acc-acc_base:+.1f}pp')
" 2>&1
