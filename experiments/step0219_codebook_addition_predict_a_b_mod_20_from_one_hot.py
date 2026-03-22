"""
Step 219 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11229.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 219: Addition — predict a + b (mod 20) from one-hot encoded a, b
# One-hot makes k-NN exact. This tests whether the substrate can learn
# addition as a rule from examples.

n_range = 10  # numbers 0-9, sum 0-18 (19 possible outputs)
n_out = 2 * n_range - 1  # 0 through 18

# One-hot encode each number
def one_hot_pair(a, b, n=10):
    v = torch.zeros(2*n, device=device)
    v[a] = 1; v[n+b] = 1
    return v

n_train = 800  # of 100 possible pairs
X_tr = []; y_tr = []
for _ in range(n_train):
    a, b = torch.randint(0, n_range, (2,)).tolist()
    X_tr.append(one_hot_pair(a, b)); y_tr.append(a + b)
X_tr = torch.stack(X_tr); y_tr = torch.tensor(y_tr, device=device, dtype=torch.long)

# Test: ALL 100 possible pairs
X_te = []; y_te = []
for a in range(n_range):
    for b in range(n_range):
        X_te.append(one_hot_pair(a, b)); y_te.append(a + b)
X_te = torch.stack(X_te); y_te = torch.tensor(y_te, device=device, dtype=torch.long)

def knn(V, labels, te, yte, n_cls, k=5):
    sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    scores = torch.zeros(te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == yte).float().mean().item() * 100

acc_base = knn(X_tr, y_tr, X_te, y_te, n_out)
print(f'Addition (0-9 + 0-9, one-hot encoded):')
print(f'  k-NN base: {acc_base:.1f}%')
print(f'  Random: {100/n_out:.1f}%')

# Now test: can the substrate discover features that help?
templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def loo(V, labels, n_cls):
    V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

V = X_tr.clone(); layers = []
for _ in range(5):
    cd = V.shape[1]; bl = loo(V, y_tr, n_out); best = None
    for tn, tf in templates.items():
        for _ in range(50):
            w = torch.randn(cd, device=device)/(cd**0.5); b = torch.rand(1,device=device)*n_out
            try:
                feat = tf(V, w, b).unsqueeze(1); aug = F.normalize(torch.cat([V,feat],1),dim=1)
                l = loo(aug, y_tr, n_out)
                if l > bl+0.003: bl=l; best=(tn,w.clone(),b.clone())
            except: pass
    if best is None: break
    tn,w,b = best; layers.append((tn,w,b))
    V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)

V_te2 = X_te.clone(); V_tr2 = X_tr.clone()
for tn,w,b in layers:
    V_tr2 = torch.cat([V_tr2, templates[tn](V_tr2,w,b).unsqueeze(1)], 1)
    V_te2 = torch.cat([V_te2, templates[tn](V_te2,w,b).unsqueeze(1)], 1)
acc_sub = knn(F.normalize(V_tr2,dim=1), y_tr, F.normalize(V_te2,dim=1), y_te, n_out)

print(f'  Substrate: {acc_sub:.1f}% ({len(layers)} layers, delta={acc_sub-acc_base:+.1f}pp)')
" 2>&1
