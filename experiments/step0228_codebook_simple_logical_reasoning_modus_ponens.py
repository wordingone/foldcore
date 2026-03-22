"""
Step 228 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11320.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 228: Simple logical reasoning — modus ponens
# Premises: A=T/F, B=T/F, rule=(A implies B)
# Conclusion: given A=T, is B=T?
# This is a 1-step inference chain

# Encode: [A, B, rule_type] where rule_type encodes the relationship
# Rules: 0=A->B, 1=B->A, 2=A<->B, 3=independent
d = 3; n_train = 1000

X = torch.randint(0, 2, (n_train, 2), device=device).float()
rule = torch.randint(0, 4, (n_train,), device=device)
X_full = torch.cat([X, rule.float().unsqueeze(1)], 1)

# Label: is the state CONSISTENT with the rule?
def consistent(a, b, r):
    if r == 0: return not (a == 1 and b == 0)  # A->B: if A then B
    if r == 1: return not (b == 1 and a == 0)  # B->A: if B then A
    if r == 2: return a == b                     # A<->B
    return True                                   # independent

y = torch.tensor([int(consistent(int(X[i,0]), int(X[i,1]), rule[i].item())) for i in range(n_train)], device=device, dtype=torch.long)

# Test
Xte = []; yte = []
for a in range(2):
    for b in range(2):
        for r in range(4):
            Xte.append([float(a), float(b), float(r)])
            yte.append(int(consistent(a, b, r)))
Xte = torch.tensor(Xte, device=device); yte = torch.tensor(yte, device=device, dtype=torch.long)

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def loo(V,labels,n_cls=2):
    V_n=F.normalize(V,dim=1);sims=V_n@V_n.T;sims.fill_diagonal_(-1e9)
    scores=torch.zeros(V.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==labels).float().mean().item()
def knn(V,labels,te,yte,n_cls=2):
    sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
    scores=torch.zeros(te.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==yte).float().mean().item()*100

base = knn(X_full, y, Xte, yte)

V=X_full.clone();layers=[]
for _ in range(5):
    cd=V.shape[1];bl=loo(V,y);best=None
    for tn,tf in templates.items():
        for _ in range(50):
            w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*4
            try:
                feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                l=loo(aug,y)
                if l>bl+0.005:bl=l;best=(tn,w.clone(),b.clone())
            except:pass
    if best is None:break
    tn,w,b=best;layers.append((tn,w,b))
    V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)

Vte=Xte.clone();Vtr=X_full.clone()
for tn,w,b in layers:
    Vtr=torch.cat([Vtr,templates[tn](Vtr,w,b).unsqueeze(1)],1)
    Vte=torch.cat([Vte,templates[tn](Vte,w,b).unsqueeze(1)],1)
sub=knn(F.normalize(Vtr,dim=1),y,F.normalize(Vte,dim=1),yte)

print(f'Logical reasoning (modus ponens / consistency):')
print(f'  Base k-NN: {base:.1f}%')
print(f'  Substrate: {sub:.1f}% ({len(layers)} layers, delta={sub-base:+.1f}pp)')
print(f'  Total states: {len(Xte)} (all combinations)')
" 2>&1
