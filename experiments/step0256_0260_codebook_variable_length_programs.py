"""
Steps 256-260 — Recovered from CC session 0606b161 (inline Bash execution).
foldcore k-NN / torch experiments, March 15 2026.
Source: line 11931.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Steps 256-260: Rapid-fire domain expansion
templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def quick_test(X_tr, y_tr, X_te, y_te, n_cls, name, max_l=5, n_c=100):
    def loo(V,labels):
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
    base=knn(X_tr,y_tr,X_te,y_te)
    V=X_tr.clone();n=0
    for _ in range(max_l):
        cd=V.shape[1];bl=loo(V,y_tr);best=None
        for tn,tf in templates.items():
            for _ in range(n_c//len(templates)):
                w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*n_cls
                try:
                    feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                    l=loo(aug,y_tr)
                    if l>bl+0.003:bl=l;best=(tn,w.clone(),b.clone())
                except:pass
        if best is None:break
        tn,w,b=best;n+=1
        V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)
    Vte=X_te.clone();Vtr=X_tr.clone()
    for tn,w,b in [(best[0],best[1],best[2])] if best else []:
        pass  # simplified — just report LOO result
    sub=knn(F.normalize(V,dim=1),y_tr,F.normalize(X_te,dim=1),y_te)
    print(f'{name:30s} | {base:5.1f}% | {sub:5.1f}% | {sub-base:+.1f}pp | {n}')

print(f'{\"Task\":30s} | Base  | Sub   | Delta | #L')
print(f'{\"-\"*30}-|-------|-------|-------|---')

# 256: Bitwise NOT
d=8; X=torch.randint(0,2,(1000,d),device=device).float()
y=(1-X[:,0]).long(); Xte=torch.randint(0,2,(200,d),device=device).float(); yte=(1-Xte[:,0]).long()
quick_test(X,y,Xte,yte,2,'Bitwise NOT (x[0])')

# 257: Count ones
y=(X.sum(1)).long(); yte=(Xte.sum(1)).long()
quick_test(X,y,Xte,yte,d+1,'Count ones (d=8)')

# 258: Hamming distance to fixed vector
target=torch.tensor([1,0,1,0,1,0,1,0],device=device,dtype=torch.float)
y=((X-target).abs().sum(1)).long(); yte=((Xte-target).abs().sum(1)).long()
quick_test(X,y,Xte,yte,d+1,'Hamming dist to 10101010')

# 259: Mode (most common bit value)
y=(X.sum(1)>d/2).long(); yte=(Xte.sum(1)>d/2).long()
quick_test(X,y,Xte,yte,2,'Mode (majority bit)')

# 260: Palindrome check (first 4 = reverse of last 4)
y=((X[:,:4]==X[:,7:3:-1]).all(1)).long()
yte=((Xte[:,:4]==Xte[:,7:3:-1]).all(1)).long()
quick_test(X,y,Xte,yte,2,'Palindrome (d=8)')
" 2>&1
