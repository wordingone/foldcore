"""
Step 238 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11535.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 238: Composed arithmetic — (a + b) * c from one truth table

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
    tk=sims[0].topk(5)
    return y_s[tk.indices].mode().values.item(), y_c[tk.indices].mode().values.item()

def to_bits(n,nb): return [(n>>i)&1 for i in range(nb)]
def from_bits(bits): return sum(b*(2**i) for i,b in enumerate(bits))

def add_bin(a,b,nb):
    ab=to_bits(a,nb); bb=to_bits(b,nb); carry=0; r=[]
    for i in range(nb):
        s,carry=fa(ab[i] if i<len(ab) else 0, bb[i] if i<len(bb) else 0, carry); r.append(s)
    r.append(carry)
    return from_bits(r)

def mul_bin(a,b,nb=8):
    result=0; ab=to_bits(a,nb); bb=to_bits(b,nb)
    for i in range(nb):
        if bb[i]==1:
            result = add_bin(result, a*(2**i), 2*nb)
    return result

# Test: (a + b) * c
print('Step 238: Composed arithmetic (a+b)*c')
correct = total = 0
for a in range(0, 15, 2):
    for b in range(0, 15, 2):
        for c in range(0, 10, 2):
            sum_ab = add_bin(a, b, 8)
            prod = mul_bin(sum_ab, c, 8)
            true = (a + b) * c
            ok = prod == true
            total += 1; correct += int(ok)

print(f'  Overall: {correct}/{total} ({correct/total*100:.1f}%)')
print(f'  (7+8)*5 = {mul_bin(add_bin(7,8,8), 5, 8)} (true: {(7+8)*5})')
print(f'  (13+14)*9 = {mul_bin(add_bin(13,14,8), 9, 8)} (true: {(13+14)*9})')
" 2>&1
