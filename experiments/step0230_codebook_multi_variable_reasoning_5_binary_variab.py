"""
Step 230 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11346.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 230: Multi-variable reasoning — 5 binary variables, complex rule
# Rule: output = (A AND B) OR (C XOR D) OR (NOT E AND A)
# d=5, 32 possible states

d = 5; n_train = 200; n_test = 32  # intentionally SMALL train set

X = torch.randint(0, 2, (n_train, d), device=device).float()
def rule(x):
    a,b,c,dd,e = int(x[0]),int(x[1]),int(x[2]),int(x[3]),int(x[4])
    return int((a and b) or (c ^ dd) or ((not e) and a))
y = torch.tensor([rule(X[i]) for i in range(n_train)], device=device, dtype=torch.long)

Xte = torch.zeros(32, d, device=device)
for i in range(32):
    for b in range(d): Xte[i,b] = (i>>b)&1
yte = torch.tensor([rule(Xte[i]) for i in range(32)], device=device, dtype=torch.long)

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def loo(V,labels):
    V_n=F.normalize(V,dim=1);sims=V_n@V_n.T;sims.fill_diagonal_(-1e9)
    scores=torch.zeros(V.shape[0],2,device=device)
    for c in range(2):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==labels).float().mean().item()

def knn(V,labels,te,yte):
    sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
    scores=torch.zeros(te.shape[0],2,device=device)
    for c in range(2):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==yte).float().mean().item()*100

base=knn(X,y,Xte,yte)

V=X.clone();layers=[]
for _ in range(5):
    cd=V.shape[1];bl=loo(V,y);best=None
    for tn,tf in templates.items():
        for _ in range(100):
            w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*2
            try:
                feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                l=loo(aug,y)
                if l>bl+0.005:bl=l;best=(tn,w.clone(),b.clone())
            except:pass
    if best is None:break
    tn,w,b=best;layers.append((tn,w,b))
    V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)

Vte=Xte.clone();Vtr=X.clone()
for tn,w,b in layers:
    Vtr=torch.cat([Vtr,templates[tn](Vtr,w,b).unsqueeze(1)],1)
    Vte=torch.cat([Vte,templates[tn](Vte,w,b).unsqueeze(1)],1)
sub=knn(F.normalize(Vtr,dim=1),y,F.normalize(Vte,dim=1),yte)

print(f'Complex multi-variable reasoning:')
print(f'  Rule: (A AND B) OR (C XOR D) OR (NOT E AND A)')
print(f'  Train: {n_train} of 32 possible states')
print(f'  Base k-NN: {base:.1f}%')
print(f'  Substrate: {sub:.1f}% ({len(layers)} layers, delta={sub-base:+.1f}pp)')

# How many of the 32 states are in training?
seen = set()
for i in range(n_train):
    seen.add(tuple(X[i].int().tolist()))
print(f'  Unique training states: {len(seen)}/32')
" 2>&1
