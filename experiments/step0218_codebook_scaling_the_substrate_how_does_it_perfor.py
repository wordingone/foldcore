"""
Step 218 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11210.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 218: SCALING the substrate — how does it perform as d increases?
# Parity on d = 8, 12, 16, 20, 30
# This tests whether the substrate scales to higher dimensions

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def test_parity_at_d(d, n_train=2000, n_test=500):
    X = torch.randint(0, 2, (n_train, d), device=device).float()
    y = (X.sum(1) % 2).long()
    Xte = torch.randint(0, 2, (n_test, d), device=device).float()
    yte = (Xte.sum(1) % 2).long()
    
    def loo(V, labels):
        V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
        scores = torch.zeros(V.shape[0], 2, device=device)
        for c in range(2):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == labels).float().mean().item()
    
    def knn(V, labels, te, yte):
        sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
        scores = torch.zeros(te.shape[0], 2, device=device)
        for c in range(2):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == yte).float().mean().item() * 100
    
    base = knn(X, y, Xte, yte)
    
    V = X.clone(); layers = []
    for _ in range(10):
        cd = V.shape[1]; bl = loo(V, y); best = None
        for tn, tf in templates.items():
            for _ in range(100):
                w = torch.randn(cd, device=device)/(cd**0.5); b = torch.rand(1,device=device)*6.28
                try:
                    feat = tf(V, w, b).unsqueeze(1); aug = F.normalize(torch.cat([V,feat],1),dim=1)
                    l = loo(aug, y)
                    if l > bl+0.001: bl=l; best=(tn,w.clone(),b.clone())
                except: pass
        if best is None: break
        tn,w,b = best; layers.append((tn,w,b))
        V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)
    
    V_te = Xte.clone(); V_tr = X.clone()
    for tn,w,b in layers:
        V_tr = torch.cat([V_tr, templates[tn](V_tr,w,b).unsqueeze(1)], 1)
        V_te = torch.cat([V_te, templates[tn](V_te,w,b).unsqueeze(1)], 1)
    sub = knn(F.normalize(V_tr,dim=1), y, F.normalize(V_te,dim=1), yte)
    return base, sub, len(layers)

print(f'{\"d\":>3s} | Base  | Sub   | Delta | #L')
print(f'----|-------|-------|-------|---')
for d in [8, 12, 16, 20, 30, 50]:
    b, s, n = test_parity_at_d(d)
    print(f'{d:3d} | {b:5.1f}% | {s:5.1f}% | {s-b:+.1f}pp | {n}')
" 2>&1
