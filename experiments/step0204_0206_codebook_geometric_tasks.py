"""
Steps 204-206 — Recovered from CC session 0606b161 (inline Bash execution).
foldcore k-NN / torch experiments, March 15 2026.
Source: line 11135.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 204: UCI Iris dataset — tabular, structured, 4 features, 3 classes
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = torch.tensor(iris.data, dtype=torch.float32, device=device), torch.tensor(iris.target, dtype=torch.long, device=device)

# 80/20 split
idx = torch.randperm(len(X))
n_tr = int(0.8 * len(X))
X_tr, y_tr = X[idx[:n_tr]], y[idx[:n_tr]]
X_te, y_te = X[idx[n_tr:]], y[idx[n_tr:]]
X_tr = F.normalize(X_tr, dim=1); X_te = F.normalize(X_te, dim=1)

d = 4

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b)}

def loo(V, labels, n_cls=3, k=5):
    V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

def knn_acc(V, labels, te, yte, n_cls=3, k=5):
    sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    scores = torch.zeros(te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == yte).float().mean().item() * 100

acc_base = knn_acc(X_tr, y_tr, X_te, y_te)

# Feature discovery
V = X_tr.clone(); layers = []
for step in range(5):
    cd = V.shape[1]; best_loo = loo(V, y_tr); best = None
    for tn, tf in templates.items():
        for _ in range(100):
            w = torch.randn(cd, device=device)/(cd**0.5)
            b = torch.rand(1, device=device)*6.28
            try:
                feat = tf(V, w, b).unsqueeze(1)
                aug = F.normalize(torch.cat([V,feat],1),dim=1)
                l = loo(aug, y_tr)
                if l > best_loo+0.005: best_loo=l; best=(tn,w.clone(),b.clone())
            except: pass
    if best is None: break
    tn,w,b = best; layers.append((tn,w,b))
    V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)

V_te = X_te.clone(); V_tr2 = X_tr.clone()
for tn,w,b in layers:
    V_tr2 = torch.cat([V_tr2, templates[tn](V_tr2,w,b).unsqueeze(1)], 1)
    V_te = torch.cat([V_te, templates[tn](V_te,w,b).unsqueeze(1)], 1)
acc_sub = knn_acc(F.normalize(V_tr2,dim=1), y_tr, F.normalize(V_te,dim=1), y_te)

print(f'UCI Iris (d=4, 3 classes):')
print(f'  k-NN base:  {acc_base:.1f}%')
print(f'  Substrate:  {acc_sub:.1f}% ({len(layers)} layers, delta={acc_sub-acc_base:+.1f}pp)')

# Step 205: Breast Cancer dataset — binary classification, 30 features
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
X2 = torch.tensor(bc.data, dtype=torch.float32, device=device)
y2 = torch.tensor(bc.target, dtype=torch.long, device=device)
idx2 = torch.randperm(len(X2)); n_tr2 = int(0.8*len(X2))
X_tr2, y_tr2 = F.normalize(X2[idx2[:n_tr2]], dim=1), y2[idx2[:n_tr2]]
X_te2, y_te2 = F.normalize(X2[idx2[n_tr2:]], dim=1), y2[idx2[n_tr2:]]

acc_bc_base = knn_acc(X_tr2, y_tr2, X_te2, y_te2, n_cls=2)

V = X_tr2.clone(); layers2 = []
for step in range(5):
    cd = V.shape[1]; best_loo = loo(V, y_tr2, n_cls=2); best = None
    for tn, tf in templates.items():
        for _ in range(50):
            w = torch.randn(cd, device=device)/(cd**0.5)
            b = torch.rand(1, device=device)*6.28
            try:
                feat = tf(V, w, b).unsqueeze(1)
                aug = F.normalize(torch.cat([V,feat],1),dim=1)
                l = loo(aug, y_tr2, n_cls=2)
                if l > best_loo+0.005: best_loo=l; best=(tn,w.clone(),b.clone())
            except: pass
    if best is None: break
    tn,w,b = best; layers2.append((tn,w,b))
    V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)

V_te3 = X_te2.clone(); V_tr3 = X_tr2.clone()
for tn,w,b in layers2:
    V_tr3 = torch.cat([V_tr3, templates[tn](V_tr3,w,b).unsqueeze(1)], 1)
    V_te3 = torch.cat([V_te3, templates[tn](V_te3,w,b).unsqueeze(1)], 1)
acc_bc_sub = knn_acc(F.normalize(V_tr3,dim=1), y_tr2, F.normalize(V_te3,dim=1), y_te2, n_cls=2)

print(f'\\nBreast Cancer (d=30, 2 classes):')
print(f'  k-NN base:  {acc_bc_base:.1f}%')
print(f'  Substrate:  {acc_bc_sub:.1f}% ({len(layers2)} layers, delta={acc_bc_sub-acc_bc_base:+.1f}pp)')

# Step 206: Wine dataset
from sklearn.datasets import load_wine
wine = load_wine()
X3 = torch.tensor(wine.data, dtype=torch.float32, device=device)
y3 = torch.tensor(wine.target, dtype=torch.long, device=device)
idx3 = torch.randperm(len(X3)); n_tr3 = int(0.8*len(X3))
X_tr3, y_tr3 = F.normalize(X3[idx3[:n_tr3]], dim=1), y3[idx3[:n_tr3]]
X_te3, y_te3 = F.normalize(X3[idx3[n_tr3:]], dim=1), y3[idx3[n_tr3:]]

acc_wine_base = knn_acc(X_tr3, y_tr3, X_te3, y_te3)

V = X_tr3.clone(); layers3 = []
for step in range(5):
    cd = V.shape[1]; best_loo = loo(V, y_tr3); best = None
    for tn, tf in templates.items():
        for _ in range(50):
            w = torch.randn(cd, device=device)/(cd**0.5)
            b = torch.rand(1, device=device)*6.28
            try:
                feat = tf(V, w, b).unsqueeze(1)
                aug = F.normalize(torch.cat([V,feat],1),dim=1)
                l = loo(aug, y_tr3)
                if l > best_loo+0.005: best_loo=l; best=(tn,w.clone(),b.clone())
            except: pass
    if best is None: break
    tn,w,b = best; layers3.append((tn,w,b))
    V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)

V_te4 = X_te3.clone(); V_tr4 = X_tr3.clone()
for tn,w,b in layers3:
    V_tr4 = torch.cat([V_tr4, templates[tn](V_tr4,w,b).unsqueeze(1)], 1)
    V_te4 = torch.cat([V_te4, templates[tn](V_te4,w,b).unsqueeze(1)], 1)
acc_wine_sub = knn_acc(F.normalize(V_tr4,dim=1), y_tr3, F.normalize(V_te4,dim=1), y_te3)

print(f'\\nWine (d=13, 3 classes):')
print(f'  k-NN base:  {acc_wine_base:.1f}%')
print(f'  Substrate:  {acc_wine_sub:.1f}% ({len(layers3)} layers, delta={acc_wine_sub-acc_wine_base:+.1f}pp)')
" 2>&1
