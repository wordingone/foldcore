"""
Step 203 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11132.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 203: Fashion-MNIST — does the substrate help on a DIFFERENT image dataset?
from torchvision import datasets, transforms

fmnist = datasets.FashionMNIST('C:/tmp/fmnist', train=True, download=True, transform=transforms.ToTensor())
fmnist_test = datasets.FashionMNIST('C:/tmp/fmnist', train=False, transform=transforms.ToTensor())

X_tr = F.normalize(fmnist.data[:6000].float().view(-1, 784).to(device), dim=1)
y_tr = fmnist.targets[:6000].to(device)
X_te = F.normalize(fmnist_test.data[:1000].float().view(-1, 784).to(device), dim=1)
y_te = fmnist_test.targets[:1000].to(device)

def knn_acc(V, labels, te, yte, k=5):
    sims = te @ V.T
    n_cls = labels.max().item()+1
    scores = torch.zeros(te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == yte).float().mean().item() * 100

# Base k-NN
acc_base = knn_acc(X_tr, y_tr, X_te, y_te)

# Substrate: layered feature discovery
templates = {
    'cos': lambda x,w,b: torch.cos(x@w+b),
    'abs': lambda x,w,b: torch.abs(x@w+b),
}

def loo(V, labels, k=5):
    V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
    n_cls = labels.max().item()+1
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

V = X_tr.clone(); layers = []
for step in range(5):
    cd = V.shape[1]; best_loo = loo(V, y_tr); best = None
    for tn, tf in templates.items():
        for _ in range(30):
            w = torch.randn(cd, device=device)/(cd**0.5)
            b = torch.rand(1, device=device)*6.28
            try:
                feat = tf(V, w, b).unsqueeze(1)
                aug = F.normalize(torch.cat([V,feat],1),dim=1)
                l = loo(aug, y_tr)
                if l > best_loo+0.001: best_loo=l; best=(tn,w.clone(),b.clone())
            except: pass
    if best is None: break
    tn,w,b = best; layers.append((tn,w,b))
    V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)

# Eval
V_te = X_te.clone()
V_tr = X_tr.clone()
for tn,w,b in layers:
    V_tr = torch.cat([V_tr, templates[tn](V_tr,w,b).unsqueeze(1)], 1)
    V_te = torch.cat([V_te, templates[tn](V_te,w,b).unsqueeze(1)], 1)
acc_sub = knn_acc(F.normalize(V_tr,dim=1), y_tr, F.normalize(V_te,dim=1), y_te)

print(f'Fashion-MNIST (6K train, 1K test):')
print(f'  k-NN base:  {acc_base:.1f}%')
print(f'  Substrate:  {acc_sub:.1f}% ({len(layers)} layers, delta={acc_sub-acc_base:+.1f}pp)')
" 2>&1
