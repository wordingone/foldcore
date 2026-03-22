"""
Step 280 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 12408.
"""
import torch, torch.nn.functional as F, math
device = 'cuda'

# Step 280: Can the substrate discover GCD from I/O?
# Given: (a, b) -> gcd(a,b) examples
# Can it generalize to unseen pairs?

# NOT using the decomposed arithmetic engine (that's manual).
# Using the SUBSTRATE: k-NN + feature discovery.

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

# Training: random (a,b) pairs, label = gcd(a,b)
n_train = 2000
X = torch.randint(1, 30, (n_train, 2), device=device).float()
y = torch.tensor([math.gcd(int(X[i,0]), int(X[i,1])) for i in range(n_train)], device=device, dtype=torch.long)

# Test: different random pairs (some unseen)
n_test = 200
Xte = torch.randint(1, 30, (n_test, 2), device=device).float()
yte = torch.tensor([math.gcd(int(Xte[i,0]), int(Xte[i,1])) for i in range(n_test)], device=device, dtype=torch.long)

n_cls = max(y.max().item(), yte.max().item()) + 1

def loo(V, labels):
    V_n=F.normalize(V,dim=1);sims=V_n@V_n.T;sims.fill_diagonal_(-1e9)
    scores=torch.zeros(V.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==labels).float().mean().item()

def knn(V,labels,te,yte):
    sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
    scores=torch.zeros(te.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==yte).float().mean().item()*100

base = knn(X, y, Xte, yte)

# Feature discovery
V=X.clone();layers=[]
for _ in range(10):
    cd=V.shape[1];bl=loo(V,y);best=None
    for tn,tf in templates.items():
        for _ in range(100):
            w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*n_cls
            try:
                feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                l=loo(aug,y)
                if l>bl+0.002:bl=l;best=(tn,w.clone(),b.clone())
            except:pass
    if best is None:break
    tn,w,b=best;layers.append((tn,w,b))
    V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)

Vte=Xte.clone();Vtr=X.clone()
for tn,w,b in layers:
    Vtr=torch.cat([Vtr,templates[tn](Vtr,w,b).unsqueeze(1)],1)
    Vte=torch.cat([Vte,templates[tn](Vte,w,b).unsqueeze(1)],1)
sub=knn(F.normalize(Vtr,dim=1),y,F.normalize(Vte,dim=1),yte)

print(f'Step 280: Can the substrate DISCOVER GCD from I/O?')
print(f'  Base k-NN: {base:.1f}%')
print(f'  Substrate: {sub:.1f}% ({len(layers)} layers, delta={sub-base:+.1f}pp)')
print(f'  Random: {100/n_cls:.1f}%')
print(f'  Number of distinct GCD values: {len(y.unique())}')
print(f'  This is PATTERN MATCHING, not ALGORITHM DISCOVERY.')
print(f'  The honest question: is pattern matching ENOUGH for GCD?')
" 2>&1
