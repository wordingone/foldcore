"""
Step 274 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 12244.
"""
import torch, torch.nn.functional as F, math
device = 'cuda'

# Step 274: Euclid's GCD algorithm via decomposed arithmetic
# GCD(a,b) = while b>0: a,b = b, a%b. Return a.

# Proven arithmetic engine
X_fa=[]; y_s=[]; y_c=[]
for a in range(2):
    for b in range(2):
        for cin in range(2):
            s=a+b+cin
            for _ in range(50):
                X_fa.append([float(a),float(b),float(cin)]); y_s.append(s%2); y_c.append(s//2)
X_fa=torch.tensor(X_fa,device=device); y_s=torch.tensor(y_s,device=device,dtype=torch.long); y_c=torch.tensor(y_c,device=device,dtype=torch.long)

def fa(a,b,cin):
    q=torch.tensor([float(a),float(b),float(cin)],device=device)
    sims=F.normalize(q.unsqueeze(0),dim=1)@F.normalize(X_fa,dim=1).T
    return y_s[sims[0].topk(5).indices].mode().values.item(), y_c[sims[0].topk(5).indices].mode().values.item()

def add_bin(a,b,nb=12):
    ab=[(a>>i)&1 for i in range(nb)];bb=[(b>>i)&1 for i in range(nb)];carry=0;r=[]
    for i in range(nb):s,carry=fa(ab[i],bb[i],carry);r.append(s)
    return sum(bit*(2**i) for i,bit in enumerate(r+[carry]))

def div_mod(a,b):
    if b==0: return 0,a
    q=0;r=a
    while r>=b and q<1000:
        r=r-b; q=add_bin(q,1)
    return q,r

def gcd_euclid(a,b):
    steps=0
    while b>0 and steps<100:
        _,r = div_mod(a,b)
        a,b = b,r
        steps+=1
    return a

# Test
print('Step 274: GCD via Euclid\\'s algorithm (decomposed arithmetic)')
correct=total=0
for a in range(1,30,2):
    for b in range(1,30,3):
        pred = gcd_euclid(a,b)
        true = math.gcd(a,b)
        ok = pred==true
        total+=1; correct+=int(ok)

print(f'  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)')
print(f'  GCD(48,18) = {gcd_euclid(48,18)} (true: {math.gcd(48,18)})')
print(f'  GCD(100,75) = {gcd_euclid(100,75)} (true: {math.gcd(100,75)})')
print(f'  GCD(17,13) = {gcd_euclid(17,13)} (true: {math.gcd(17,13)})')
print(f'  GCD(144,89) = {gcd_euclid(144,89)} (true: {math.gcd(144,89)})')
" 2>&1
