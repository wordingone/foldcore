"""
Steps 268-272 — Recovered from CC session 0606b161 (inline Bash execution).
foldcore k-NN / torch experiments, March 15 2026.
Source: line 12193.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Steps 268-280: Final batch — characterize limits at scale

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b)}

def quick_substrate(X,y,Xte,yte,n_cls):
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
    base=knn(X,y,Xte,yte)
    V=X.clone();n=0
    for _ in range(3):
        cd=V.shape[1];bl=loo(V,y);best=None
        for tn,tf in templates.items():
            for _ in range(30):
                w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*n_cls
                try:
                    feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                    l=loo(aug,y)
                    if l>bl+0.005:bl=l;best=(tn,w.clone(),b.clone())
                except:pass
        if best is None:break
        tn,w,b=best;n+=1
        V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)
    Vte=Xte.clone();Vtr=X.clone()
    for tn,w,b in [(best[0],best[1],best[2])] if best and n>0 else []:
        Vtr=torch.cat([Vtr,templates[tn](Vtr,w,b).unsqueeze(1)],1)
        Vte=torch.cat([Vte,templates[tn](Vte,w,b).unsqueeze(1)],1)
    sub=knn(F.normalize(V,dim=1),y,F.normalize(Xte,dim=1),yte) if n==0 else knn(F.normalize(Vtr,dim=1),y,F.normalize(Vte,dim=1),yte)
    return base, sub, n

print('FINAL BATCH: Steps 268-280')
print(f'{\"Task\":35s} | Base  | Sub   | Delta')
print(f'{\"-\"*35}-|-------|-------|------')

# 268: GCD classifier — is GCD(a,b) > 1?
X=torch.randint(1,15,(1000,2),device=device).float()
import math
y=torch.tensor([1 if math.gcd(int(X[i,0]),int(X[i,1]))>1 else 0 for i in range(1000)],device=device,dtype=torch.long)
Xte=torch.randint(1,15,(200,2),device=device).float()
yte=torch.tensor([1 if math.gcd(int(Xte[i,0]),int(Xte[i,1]))>1 else 0 for i in range(200)],device=device,dtype=torch.long)
b,s,n=quick_substrate(X,y,Xte,yte,2)
print(f'{\"GCD(a,b) > 1?\":35s} | {b:5.1f}% | {s:5.1f}% | {s-b:+.1f}pp')

# 269: Prime check — is n prime?
X=torch.arange(2,100,device=device).float().unsqueeze(1)
def is_prime(n):
    n=int(n)
    if n<2: return 0
    for i in range(2,int(n**0.5)+1):
        if n%i==0: return 0
    return 1
y=torch.tensor([is_prime(X[i,0]) for i in range(len(X))],device=device,dtype=torch.long)
Xte=torch.arange(2,50,device=device).float().unsqueeze(1)
yte=torch.tensor([is_prime(Xte[i,0]) for i in range(len(Xte))],device=device,dtype=torch.long)
b,s,n=quick_substrate(X,y,Xte,yte,2)
print(f'{\"Is n prime? (2-100)\":35s} | {b:5.1f}% | {s:5.1f}% | {s-b:+.1f}pp')

# 270: Fibonacci membership — is n a Fibonacci number?
fibs=set([1,2,3,5,8,13,21,34,55,89])
X=torch.arange(1,100,device=device).float().unsqueeze(1)
y=torch.tensor([1 if int(X[i,0]) in fibs else 0 for i in range(len(X))],device=device,dtype=torch.long)
Xte=torch.arange(1,50,device=device).float().unsqueeze(1)
yte=torch.tensor([1 if int(Xte[i,0]) in fibs else 0 for i in range(len(Xte))],device=device,dtype=torch.long)
b,s,n=quick_substrate(X,y,Xte,yte,2)
print(f'{\"Is n Fibonacci?\":35s} | {b:5.1f}% | {s:5.1f}% | {s-b:+.1f}pp')

# 271: Collatz step — n even: n/2, n odd: 3n+1
X=torch.arange(1,100,device=device).float().unsqueeze(1).repeat(5,1)
y=torch.tensor([int(X[i,0])//2 if int(X[i,0])%2==0 else 3*int(X[i,0])+1 for i in range(len(X))],device=device,dtype=torch.long)
n_cls=y.max().item()+1
Xte=torch.arange(1,30,device=device).float().unsqueeze(1)
yte=torch.tensor([int(Xte[i,0])//2 if int(Xte[i,0])%2==0 else 3*int(Xte[i,0])+1 for i in range(len(Xte))],device=device,dtype=torch.long)
b,s,n=quick_substrate(X,y,Xte,yte,min(n_cls,300))
print(f'{\"Collatz step\":35s} | {b:5.1f}% | {s:5.1f}% | {s-b:+.1f}pp')

# 272: Digit sum (single digit result)
X=torch.randint(0,100,(1000,1),device=device).float()
def digit_sum(n):
    return sum(int(d) for d in str(int(n)))
y=torch.tensor([digit_sum(X[i,0]) for i in range(len(X))],device=device,dtype=torch.long)
Xte=torch.randint(0,100,(200,1),device=device).float()
yte=torch.tensor([digit_sum(Xte[i,0]) for i in range(len(Xte))],device=device,dtype=torch.long)
b,s,n=quick_substrate(X,y,Xte,yte,19)
print(f'{\"Digit sum\":35s} | {b:5.1f}% | {s:5.1f}% | {s-b:+.1f}pp')
" 2>&1
