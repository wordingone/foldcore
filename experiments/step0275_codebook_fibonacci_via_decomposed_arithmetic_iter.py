"""
Step 275 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 12265.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 275: Fibonacci via decomposed arithmetic iteration
# fib(n) = iterate n times: (a,b) -> (b, a+b), starting from (0,1)

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

def add_bin(a,b,nb=16):
    ab=[(a>>i)&1 for i in range(nb)]; bb=[(b>>i)&1 for i in range(nb)]; carry=0; r=[]
    for i in range(nb): s,carry=fa(ab[i],bb[i],carry); r.append(s)
    return sum(bit*(2**i) for i,bit in enumerate(r+[carry]))

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, add_bin(a, b)
    return a

# True Fibonacci
def true_fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print('Step 275: Fibonacci via decomposed arithmetic')
correct = 0; total = 0
for n in range(20):
    pred = fibonacci(n)
    true = true_fib(n)
    ok = pred == true
    total += 1; correct += int(ok)
    if n <= 15 or not ok:
        print(f'  fib({n:2d}) = {pred:5d} (true: {true:5d}) {\"OK\" if ok else \"FAIL\"}')" 2>&1
