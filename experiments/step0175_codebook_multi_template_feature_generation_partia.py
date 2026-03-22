"""
Step 175 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10611.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 175: Multi-template feature generation — PARTIAL Stage 7
# Instead of just cos(w·x+b), offer multiple templates:
# - cos(w·x+b)   [nonlinear projection]
# - w·x mod 2    [modular arithmetic]
# - sign(w·x+b)  [step function]  
# - |w·x+b|      [absolute value]
# Let LOO select from ALL template types

d = 8
X = torch.randint(0, 2, (1000, d), device=device).float()
y = (X.sum(1) % 2).long()
Xte = torch.zeros(256, d, device=device)
for i in range(256):
    for b in range(d): Xte[i,b]=(i>>b)&1
yte = (Xte.sum(1) % 2).long()

def loo_accuracy(V, labels, k=5):
    V_n = F.normalize(V, dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(0)
    scores = torch.zeros(V.shape[0], 2, device=device)
    for c in range(2):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

def knn_acc(V, labels, te, yte, k=5):
    sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    scores = torch.zeros(te.shape[0], 2, device=device)
    for c in range(2):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == yte).float().mean().item() * 100

templates = {
    'cos':  lambda x, w, b: torch.cos(x @ w + b),
    'mod2': lambda x, w, b: ((x @ w.abs()).round() % 2).float(),
    'sign': lambda x, w, b: torch.sign(x @ w + b),
    'abs':  lambda x, w, b: torch.abs(x @ w + b),
    'tanh': lambda x, w, b: torch.tanh(x @ w + b),
}

V = X.clone()
acc_base = knn_acc(V, y, Xte, yte)
loo_base = loo_accuracy(F.normalize(V,dim=1), y)
print(f'Base: LOO={loo_base:.4f} test={acc_base:.1f}%')

# Search across ALL templates
best_template = None; best_w = None; best_b = None; best_loo = loo_base
for tname, tfn in templates.items():
    for _ in range(100):
        w = torch.randn(d, device=device) / (d**0.5)
        b = torch.rand(1, device=device) * 6.28
        try:
            feat = tfn(X, w, b).unsqueeze(1)
            aug = F.normalize(torch.cat([V, feat], 1), dim=1)
            loo = loo_accuracy(aug, y)
            if loo > best_loo:
                best_loo = loo; best_template = tname; best_w = w.clone(); best_b = b.clone()
        except:
            pass

if best_template:
    tfn = templates[best_template]
    feat_tr = tfn(X, best_w, best_b).unsqueeze(1)
    feat_te = tfn(Xte, best_w, best_b).unsqueeze(1)
    V_aug = F.normalize(torch.cat([V, feat_tr], 1), dim=1)
    te_aug = F.normalize(torch.cat([Xte, feat_te], 1), dim=1)
    acc = knn_acc(V_aug, y, te_aug, yte)
    print(f'Selected: {best_template} LOO={best_loo:.4f} test={acc:.1f}% (+{acc-acc_base:.1f}pp)')
    
    # Iterate
    for step in range(4):
        loo_base2 = loo_accuracy(F.normalize(V_aug, dim=1), y)
        best2 = None; best_loo2 = loo_base2
        for tname, tfn in templates.items():
            for _ in range(100):
                w = torch.randn(d, device=device) / (d**0.5)
                b = torch.rand(1, device=device) * 6.28
                try:
                    feat = tfn(X, w, b).unsqueeze(1)
                    aug = F.normalize(torch.cat([V_aug, feat], 1), dim=1)
                    loo = loo_accuracy(aug, y)
                    if loo > best_loo2:
                        best_loo2 = loo; best2 = (tname, w.clone(), b.clone())
                except: pass
        if best2 is None: break
        tname, w, b = best2
        feat_tr = templates[tname](X, w, b).unsqueeze(1)
        feat_te = templates[tname](Xte, w, b).unsqueeze(1)
        V_aug = torch.cat([V_aug, feat_tr], 1)
        te_aug = torch.cat([te_aug, feat_te], 1)
        acc = knn_acc(F.normalize(V_aug,dim=1), y, F.normalize(te_aug,dim=1), yte)
        print(f'  +{tname}: LOO={best_loo2:.4f} test={acc:.1f}%')
else:
    print('No improvement')
" 2>&1
