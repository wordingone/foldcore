"""
Step 242 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11620.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 242: Can the substrate discover ADDITION STRUCTURE from examples?
# Give it (a, b, sum) triples for small numbers.
# Can it learn to generalize to larger numbers WITHOUT being told ripple-carry?

# Key question: if we train on 0-7 + 0-7 = 0-14 in BINARY,
# and the substrate discovers features via LOO, does it learn
# per-bit operations that generalize to 8-15?

# Binary encoding: a (4 bits) + b (4 bits) = 8-dim input
n_bits = 4; d = 2 * n_bits

# Train: a, b in 0-7 (first 3 bits used)
X_tr = []; y_tr = []
for a in range(8):
    for b in range(8):
        ab = [(a>>i)&1 for i in range(n_bits)] + [(b>>i)&1 for i in range(n_bits)]
        for _ in range(5):
            X_tr.append(ab); y_tr.append(a + b)
X_tr = torch.tensor(X_tr, device=device, dtype=torch.float)
y_tr = torch.tensor(y_tr, device=device, dtype=torch.long)

# Test IN-DIST: 0-7 + 0-7
X_te_in = []; y_te_in = []
for a in range(8):
    for b in range(8):
        ab = [(a>>i)&1 for i in range(n_bits)] + [(b>>i)&1 for i in range(n_bits)]
        X_te_in.append(ab); y_te_in.append(a + b)
X_te_in = torch.tensor(X_te_in, device=device, dtype=torch.float)
y_te_in = torch.tensor(y_te_in, device=device, dtype=torch.long)

# Test OOD: includes 8-15
X_te_ood = []; y_te_ood = []
for a in range(16):
    for b in range(16):
        if a >= 8 or b >= 8:
            ab = [(a>>i)&1 for i in range(n_bits)] + [(b>>i)&1 for i in range(n_bits)]
            X_te_ood.append(ab); y_te_ood.append(a + b)
X_te_ood = torch.tensor(X_te_ood, device=device, dtype=torch.float)
y_te_ood = torch.tensor(y_te_ood, device=device, dtype=torch.long)

n_cls = 31  # max sum

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def loo(V,labels,n_cls):
    V_n=F.normalize(V,dim=1);sims=V_n@V_n.T;sims.fill_diagonal_(-1e9)
    scores=torch.zeros(V.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==labels).float().mean().item()

def knn(V,labels,te,yte,n_cls):
    sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
    scores=torch.zeros(te.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==yte).float().mean().item()*100

# Base
in_base = knn(X_tr, y_tr, X_te_in, y_te_in, n_cls)
ood_base = knn(X_tr, y_tr, X_te_ood, y_te_ood, n_cls)

# Feature discovery
V=X_tr.clone();layers=[]
for _ in range(10):
    cd=V.shape[1];bl=loo(V,y_tr,n_cls);best=None
    for tn,tf in templates.items():
        for _ in range(100):
            w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*n_cls
            try:
                feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                l=loo(aug,y_tr,n_cls)
                if l>bl+0.001:bl=l;best=(tn,w.clone(),b.clone())
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

print(f'Step 242: Can feature discovery learn addition structure?')
print(f'  In-dist:  base={in_base:.1f}% sub={in_sub:.1f}% delta={in_sub-in_base:+.1f}pp ({len(layers)} layers)')
print(f'  OOD:      base={ood_base:.1f}% sub={ood_sub:.1f}% delta={ood_sub-ood_base:+.1f}pp')
print(f'  Random OOD: {100/n_cls:.1f}%')
" 2>&1
