"""
Step 220 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11236.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 220: Addition with RAW integers — where the substrate should help
n_range = 10; n_out = 2*n_range - 1

X_tr = torch.randint(0, n_range, (800, 2), device=device).float()
y_tr = (X_tr[:,0] + X_tr[:,1]).long()

X_te = torch.zeros(100, 2, device=device)
y_te = torch.zeros(100, device=device, dtype=torch.long)
for i in range(100):
    a, b = i // 10, i % 10
    X_te[i] = torch.tensor([a, b], device=device, dtype=torch.float)
    y_te[i] = a + b

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def loo(V, labels, n_cls):
    V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

def knn(V, labels, te, yte, n_cls, k=5):
    sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    scores = torch.zeros(te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == yte).float().mean().item() * 100

base = knn(X_tr, y_tr, X_te, y_te, n_out)

V = X_tr.clone(); layers = []
for _ in range(10):
    cd = V.shape[1]; bl = loo(V, y_tr, n_out); best = None
    for tn, tf in templates.items():
        for _ in range(100):
            w = torch.randn(cd, device=device)/(cd**0.5); b = torch.rand(1,device=device)*n_out
            try:
                feat = tf(V, w, b).unsqueeze(1); aug = F.normalize(torch.cat([V,feat],1),dim=1)
                l = loo(aug, y_tr, n_out)
                if l > bl+0.001: bl=l; best=(tn,w.clone(),b.clone())
            except: pass
    if best is None: break
    tn,w,b = best; layers.append((tn,w,b))
    V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)

V_te = X_te.clone(); V_tr = X_tr.clone()
for tn,w,b in layers:
    V_tr = torch.cat([V_tr, templates[tn](V_tr,w,b).unsqueeze(1)], 1)
    V_te = torch.cat([V_te, templates[tn](V_te,w,b).unsqueeze(1)], 1)
sub = knn(F.normalize(V_tr,dim=1), y_tr, F.normalize(V_te,dim=1), y_te, n_out)

print(f'Addition (raw integers, 0-9 + 0-9):')
print(f'  Base k-NN: {base:.1f}%')
print(f'  Substrate: {sub:.1f}% ({len(layers)} layers, delta={sub-base:+.1f}pp)')
print(f'  Random: {100/n_out:.1f}%')
" 2>&1
