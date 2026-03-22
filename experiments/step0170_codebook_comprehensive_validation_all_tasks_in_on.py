"""
Step 170 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10553.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 170: COMPREHENSIVE VALIDATION — all tasks in one run

def loo_discover(X_tr, y_tr, X_te, y_te, n_rounds=3, n_cand=200, k=5):
    d = X_tr.shape[1]
    V = X_tr.clone()
    features = []
    
    def loo_acc(V, labels):
        V_n = F.normalize(V, dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(0)
        n_cls = labels.max().item() + 1
        scores = torch.zeros(V.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == labels).float().mean().item()
    
    def test_acc(V, labels, te, y_te):
        sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
        n_cls = labels.max().item() + 1
        scores = torch.zeros(te.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == y_te).float().mean().item() * 100
    
    acc_base = test_acc(F.normalize(V,dim=1), y_tr, F.normalize(X_te,dim=1), y_te)
    
    for _ in range(n_rounds):
        loo_base = loo_acc(F.normalize(V,dim=1), y_tr)
        best_w = None; best_b = None; best_loo = loo_base
        for _ in range(n_cand):
            w = torch.randn(d, device=device) / (d**0.5)
            b = torch.rand(1, device=device) * 6.28
            feat = torch.cos(X_tr @ w + b).unsqueeze(1)
            aug = F.normalize(torch.cat([V, feat], 1), dim=1)
            loo = loo_acc(aug, y_tr)
            if loo > best_loo:
                best_loo = loo; best_w = w.clone(); best_b = b.clone()
        if best_w is None: break
        features.append((best_w, best_b))
        V = torch.cat([V, torch.cos(X_tr @ best_w + best_b).unsqueeze(1)], 1)
    
    te_aug = X_te.clone()
    for w, b in features:
        te_aug = torch.cat([te_aug, torch.cos(X_te @ w + b).unsqueeze(1)], 1)
    acc_final = test_acc(F.normalize(V,dim=1), y_tr, F.normalize(te_aug,dim=1), y_te)
    
    return acc_base, acc_final, len(features)

print('=== Step 170: Comprehensive Validation ===')
print(f'{\"Task\":30s} | Base  | +Disco | Delta | #F')
print(f'{\"-\"*30}-|-------|-------|-------|---')

# 1. Parity
d=8; X=torch.randint(0,2,(1000,d),device=device).float()
y=(X.sum(1)%2).long()
Xte=torch.zeros(256,d,device=device)
for i in range(256):
    for b in range(d): Xte[i,b]=(i>>b)&1
yte=(Xte.sum(1)%2).long()
b,a,n=loo_discover(X,y,Xte,yte,n_rounds=5,n_cand=200)
print(f'{\"Parity d=8\":30s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}')

# 2. XOR d=20
d=20; X=torch.randint(0,2,(2000,d),device=device).float()
y=(X[:,0].long()^X[:,1].long()); Xte=torch.randn(500,d,device=device).clamp(0,1).round()
yte=(Xte[:,0].long()^Xte[:,1].long())
b,a,n=loo_discover(X,y,Xte,yte,n_rounds=3,n_cand=200)
print(f'{\"XOR d=20\":30s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}')

# 3. Multi-rule
d=10; X=torch.randint(0,2,(2000,d),device=device).float()
y=torch.full((2000,),3,device=device,dtype=torch.long)
for i in range(2000):
    xor=int(X[i,0])^int(X[i,1]); and_=int(X[i,2])&int(X[i,3])
    nor=1-(int(X[i,4])|int(X[i,5]))
    if xor and not and_ and not nor: y[i]=0
    elif and_ and not xor and not nor: y[i]=1
    elif nor and not xor and not and_: y[i]=2
Xte=torch.zeros(1024,d,device=device)
for i in range(1024):
    for bb in range(d): Xte[i,bb]=(i>>bb)&1
yte=torch.full((1024,),3,device=device,dtype=torch.long)
for i in range(1024):
    xor=int(Xte[i,0])^int(Xte[i,1]); and_=int(Xte[i,2])&int(Xte[i,3])
    nor=1-(int(Xte[i,4])|int(Xte[i,5]))
    if xor and not and_ and not nor: yte[i]=0
    elif and_ and not xor and not nor: yte[i]=1
    elif nor and not xor and not and_: yte[i]=2
b,a,n=loo_discover(X,y,Xte,yte,n_rounds=3,n_cand=200)
print(f'{\"Multi-rule (XOR+AND+NOR)\":30s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}')

# 4. CA Rule 90
d=3; rule={((i>>2)&1,(i>>1)&1,i&1):(90>>i)&1 for i in range(8)}
width=30; row=torch.zeros(width,dtype=torch.int); row[width//2]=1
Xca,yca=[],[]
for _ in range(100):
    nr=torch.zeros(width,dtype=torch.int)
    for i in range(1,width-1):
        nb=(row[i-1].item(),row[i].item(),row[i+1].item()); nr[i]=rule[nb]
        Xca.append([float(row[i-1]),float(row[i]),float(row[i+1])]); yca.append(nr[i].item())
    row=nr
Xca=torch.tensor(Xca,device=device,dtype=torch.float); yca=torch.tensor(yca,device=device,dtype=torch.long)
Xte_ca=torch.tensor([[i>>2&1,i>>1&1,i&1] for i in range(8)],dtype=torch.float,device=device)
yte_ca=torch.tensor([rule[tuple(Xte_ca[j].int().tolist())] for j in range(8)],dtype=torch.long,device=device)
b,a,n=loo_discover(Xca,yca,Xte_ca,yte_ca,n_rounds=3,n_cand=500)
print(f'{\"CA Rule 90 (XOR)\":30s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}')

# 5. CA Rule 110
rule110={((i>>2)&1,(i>>1)&1,i&1):(110>>i)&1 for i in range(8)}
row=torch.zeros(width,dtype=torch.int); row[width//2]=1
Xca2,yca2=[],[]
for _ in range(100):
    nr=torch.zeros(width,dtype=torch.int)
    for i in range(1,width-1):
        nb=(row[i-1].item(),row[i].item(),row[i+1].item()); nr[i]=rule110[nb]
        Xca2.append([float(row[i-1]),float(row[i]),float(row[i+1])]); yca2.append(nr[i].item())
    row=nr
Xca2=torch.tensor(Xca2,device=device,dtype=torch.float); yca2=torch.tensor(yca2,device=device,dtype=torch.long)
yte110=torch.tensor([rule110[tuple(Xte_ca[j].int().tolist())] for j in range(8)],dtype=torch.long,device=device)
b,a,n=loo_discover(Xca2,yca2,Xte_ca,yte110,n_rounds=3,n_cand=500)
print(f'{\"CA Rule 110 (Turing-complete)\":30s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}')
" 2>&1
