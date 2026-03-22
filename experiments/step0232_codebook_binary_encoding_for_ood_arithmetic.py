"""
Step 232 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11370.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 232: Binary encoding for OOD arithmetic
# Train addition on 0-7 (3-bit), test on 8-15 (4-bit)
# Binary: 7 = [1,1,1], 8 = [1,0,0,0], 9 = [1,0,0,1]

n_bits = 4  # support up to 15

def to_binary(a, b, n_bits=4):
    v = torch.zeros(2*n_bits, device=device)
    for i in range(n_bits):
        v[i] = (a >> i) & 1
        v[n_bits + i] = (b >> i) & 1
    return v

def to_binary_result(s, n_bits=5):
    v = torch.zeros(n_bits, device=device)
    for i in range(n_bits):
        v[i] = (s >> i) & 1
    return v

# Train: a,b in 0-7
X_tr = []; y_tr = []
for a in range(8):
    for b in range(8):
        for _ in range(5):  # 5 copies each
            X_tr.append(to_binary(a, b))
            y_tr.append(a + b)
X_tr = torch.stack(X_tr); y_tr = torch.tensor(y_tr, device=device, dtype=torch.long)

# Test IN-DIST: a,b in 0-7
X_te_in = []; y_te_in = []
for a in range(8):
    for b in range(8):
        X_te_in.append(to_binary(a, b)); y_te_in.append(a + b)
X_te_in = torch.stack(X_te_in); y_te_in = torch.tensor(y_te_in, device=device, dtype=torch.long)

# Test OOD: at least one of a,b in 8-15
X_te_ood = []; y_te_ood = []
for a in range(16):
    for b in range(16):
        if a >= 8 or b >= 8:
            X_te_ood.append(to_binary(a, b)); y_te_ood.append(a + b)
X_te_ood = torch.stack(X_te_ood); y_te_ood = torch.tensor(y_te_ood, device=device, dtype=torch.long)

n_cls = 31  # max sum = 15+15=30

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def knn(V,labels,te,yte,n_cls,k=5):
    sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
    scores=torch.zeros(te.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(k,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==yte).float().mean().item()*100

def loo(V,labels,n_cls):
    V_n=F.normalize(V,dim=1);sims=V_n@V_n.T;sims.fill_diagonal_(-1e9)
    scores=torch.zeros(V.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==labels).float().mean().item()

# Base
in_base = knn(X_tr, y_tr, X_te_in, y_te_in, n_cls)
ood_base = knn(X_tr, y_tr, X_te_ood, y_te_ood, n_cls)

# Substrate
V=X_tr.clone();layers=[]
for _ in range(5):
    cd=V.shape[1];bl=loo(V,y_tr,n_cls);best=None
    for tn,tf in templates.items():
        for _ in range(50):
            w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*n_cls
            try:
                feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                l=loo(aug,y_tr,n_cls)
                if l>bl+0.002:bl=l;best=(tn,w.clone(),b.clone())
            except:pass
    if best is None:break
    tn,w,b=best;layers.append((tn,w,b))
    V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)

Vte_in=X_te_in.clone();Vte_ood=X_te_ood.clone();Vtr=X_tr.clone()
for tn,w,b in layers:
    Vtr=torch.cat([Vtr,templates[tn](Vtr,w,b).unsqueeze(1)],1)
    Vte_in=torch.cat([Vte_in,templates[tn](Vte_in,w,b).unsqueeze(1)],1)
    Vte_ood=torch.cat([Vte_ood,templates[tn](Vte_ood,w,b).unsqueeze(1)],1)
in_sub=knn(F.normalize(Vtr,dim=1),y_tr,F.normalize(Vte_in,dim=1),y_te_in,n_cls)
ood_sub=knn(F.normalize(Vtr,dim=1),y_tr,F.normalize(Vte_ood,dim=1),y_te_ood,n_cls)

print(f'Addition with BINARY encoding (train 0-7, test 0-15):')
print(f'  In-distribution:  base={in_base:.1f}% sub={in_sub:.1f}% delta={in_sub-in_base:+.1f}pp')
print(f'  OOD (8-15):       base={ood_base:.1f}% sub={ood_sub:.1f}% delta={ood_sub-ood_base:+.1f}pp')
print(f'  Features: {len(layers)}')
" 2>&1
