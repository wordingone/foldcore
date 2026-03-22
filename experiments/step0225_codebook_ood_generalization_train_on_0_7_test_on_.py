"""
Step 225 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11274.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 225: OOD generalization — train on 0-7, test on 8-9
# Can the substrate extrapolate arithmetic to unseen numbers?

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b)}

def substrate(X_tr, y_tr, X_te, y_te, n_cls, max_l=10, n_c=100):
    def loo(V, labels):
        V_n=F.normalize(V,dim=1); sims=V_n@V_n.T; sims.fill_diagonal_(-1e9)
        scores=torch.zeros(V.shape[0],n_cls,device=device)
        for c in range(n_cls):
            m=labels==c; cs=sims[:,m]
            if cs.shape[1]==0: continue
            scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
        return (scores.argmax(1)==labels).float().mean().item()
    def knn(V,labels,te,yte):
        sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
        scores=torch.zeros(te.shape[0],n_cls,device=device)
        for c in range(n_cls):
            m=labels==c; cs=sims[:,m]
            if cs.shape[1]==0: continue
            scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
        return (scores.argmax(1)==yte).float().mean().item()*100
    
    base=knn(X_tr,y_tr,X_te,y_te)
    V=X_tr.clone(); layers=[]
    for _ in range(max_l):
        cd=V.shape[1]; bl=loo(V,y_tr); best=None
        for tn,tf in templates.items():
            for _ in range(n_c//len(templates)):
                w=torch.randn(cd,device=device)/(cd**0.5); b=torch.rand(1,device=device)*n_cls
                try:
                    feat=tf(V,w,b).unsqueeze(1); aug=F.normalize(torch.cat([V,feat],1),dim=1)
                    l=loo(aug,y_tr)
                    if l>bl+0.001: bl=l; best=(tn,w.clone(),b.clone())
                except: pass
        if best is None: break
        tn,w,b=best; layers.append((tn,w,b))
        V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)
    
    Vte=X_te.clone(); Vtr=X_tr.clone()
    for tn,w,b in layers:
        Vtr=torch.cat([Vtr,templates[tn](Vtr,w,b).unsqueeze(1)],1)
        Vte=torch.cat([Vte,templates[tn](Vte,w,b).unsqueeze(1)],1)
    sub=knn(F.normalize(Vtr,dim=1),y_tr,F.normalize(Vte,dim=1),y_te)
    return base,sub,len(layers)

print('OOD Generalization: train on 0-7, test includes 8-9')
print(f'{\"Task\":20s} | {\"In-dist\":>7s} | {\"OOD\":>7s}')
print(f'{\"-\"*20}-|---------|-------')

for task, fn in [('Addition', lambda a,b: a+b), ('Max', lambda a,b: max(a,b))]:
    # In-distribution: train 0-7, test 0-7
    X_in = torch.randint(0,8,(500,2),device=device).float()
    y_in = torch.tensor([fn(int(x[0]),int(x[1])) for x in X_in],device=device,dtype=torch.long)
    Xte_in = torch.randint(0,8,(100,2),device=device).float()
    yte_in = torch.tensor([fn(int(x[0]),int(x[1])) for x in Xte_in],device=device,dtype=torch.long)
    
    # OOD: train 0-7, test includes pairs with 8 or 9
    Xte_ood = []; yte_ood = []
    for a in range(10):
        for b in range(10):
            if a >= 8 or b >= 8:  # at least one OOD number
                Xte_ood.append([float(a),float(b)]); yte_ood.append(fn(a,b))
    Xte_ood = torch.tensor(Xte_ood,device=device); yte_ood = torch.tensor(yte_ood,device=device,dtype=torch.long)
    
    n_cls = max(y_in.max().item(), yte_ood.max().item()) + 1
    
    _,sub_in,_ = substrate(X_in, y_in, Xte_in, yte_in, n_cls)
    _,sub_ood,_ = substrate(X_in, y_in, Xte_ood, yte_ood, n_cls)
    
    print(f'{task:20s} | {sub_in:5.1f}%  | {sub_ood:5.1f}%')
" 2>&1
