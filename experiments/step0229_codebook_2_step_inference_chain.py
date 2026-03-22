"""
Step 229 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11343.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 229: 2-step inference chain
# Given: A, rule1(A->B or not), rule2(B->C or not)
# Predict: C's value when A=1

# Encode: [A, rule1, rule2] where rules are 0=implies, 1=not-implies
# If A=1 and rule1=implies: B=1. If B=1 and rule2=implies: C=1.
# Chain: A=1, rule1=0, rule2=0 → C=1 (transitive)

d = 3; n_train = 1000

X = torch.zeros(n_train, d, device=device)
X[:, 0] = torch.randint(0, 2, (n_train,), device=device).float()  # A
X[:, 1] = torch.randint(0, 2, (n_train,), device=device).float()  # rule1: 0=implies, 1=not
X[:, 2] = torch.randint(0, 2, (n_train,), device=device).float()  # rule2: 0=implies, 1=not

def infer_C(a, r1, r2):
    b = a if r1 == 0 else 0  # A->B if rule1=implies
    c = b if r2 == 0 else 0  # B->C if rule2=implies
    return c

y = torch.tensor([infer_C(int(X[i,0]), int(X[i,1]), int(X[i,2])) for i in range(n_train)], device=device, dtype=torch.long)

# All 8 test cases
Xte = torch.tensor([[a,r1,r2] for a in range(2) for r1 in range(2) for r2 in range(2)], device=device, dtype=torch.float)
yte = torch.tensor([infer_C(a,r1,r2) for a in range(2) for r1 in range(2) for r2 in range(2)], device=device, dtype=torch.long)

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def knn(V,labels,te,yte,n_cls=2,k=5):
    sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
    scores=torch.zeros(te.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(k,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==yte).float().mean().item()*100

def loo(V,labels,n_cls=2):
    V_n=F.normalize(V,dim=1);sims=V_n@V_n.T;sims.fill_diagonal_(-1e9)
    scores=torch.zeros(V.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==labels).float().mean().item()

base = knn(X, y, Xte, yte)

V=X.clone();layers=[]
for _ in range(5):
    cd=V.shape[1];bl=loo(V,y);best=None
    for tn,tf in templates.items():
        for _ in range(50):
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

print(f'2-step inference chain (A->B->C):')
print(f'  Base k-NN: {base:.1f}%')
print(f'  Substrate: {sub:.1f}% ({len(layers)} layers, delta={sub-base:+.1f}pp)')

# Truth table
print(f'\\n  A r1 r2 | true C | Truth table')
for i in range(8):
    a,r1,r2 = int(Xte[i,0]),int(Xte[i,1]),int(Xte[i,2])
    print(f'  {a}  {r1}  {r2}  |   {yte[i].item()}    | A={\"T\" if a else \"F\"} {\"->\" if r1==0 else \"!>\"} B {\"->\" if r2==0 else \"!>\"} C = {\"T\" if yte[i] else \"F\"}')
" 2>&1
