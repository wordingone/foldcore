"""
Step 249 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11797.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 249: Longer programs via staged composition
# Test programs of length 5, 10, 20

# Full adder (proven)
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
    ab=[(a>>i)&1 for i in range(nb)]; bb=[(b>>i)&1 for i in range(nb)]; carry=0; r=[]
    for i in range(nb): s,carry=fa(ab[i],bb[i],carry); r.append(s)
    return sum(b*(2**i) for i,b in enumerate(r+[carry]))

# Operation classifier
op_templates = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}
X_op=[]; y_op=[]
for _ in range(300):
    op=torch.randint(0,3,(1,)).item()
    noisy=[op_templates[op][i]+torch.randn(1).item()*0.3 for i in range(3)]
    X_op.append(noisy); y_op.append(op)
X_op=torch.tensor(X_op,device=device); y_op=torch.tensor(y_op,device=device,dtype=torch.long)

def classify_op(op_vec):
    sims=F.normalize(op_vec.unsqueeze(0),dim=1)@F.normalize(X_op,dim=1).T
    scores=torch.zeros(1,3,device=device)
    for c in range(3):
        m=y_op==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return scores.argmax(1).item()

def compute(op,acc,operand):
    if op==0: return add_bin(acc,operand)
    elif op==1: return max(0,acc-operand)
    else: return add_bin(acc,acc)

# Test longer programs
import random
random.seed(42)

print('Step 249: Longer programs via staged composition')
for prog_len in [5, 10, 20, 50]:
    correct = 0; n_test = 20
    for _ in range(n_test):
        prog = [(random.randint(0,2), random.randint(0,5)) for _ in range(prog_len)]
        
        acc = 0; true_acc = 0
        for op, operand in prog:
            op_vec = torch.tensor([op_templates[op][i]+random.gauss(0,0.2) for i in range(3)], device=device)
            op_pred = classify_op(op_vec)
            acc = compute(op_pred, acc, operand)
            if op==0: true_acc+=operand
            elif op==1: true_acc=max(0,true_acc-operand)
            else: true_acc*=2
            acc = acc % 4096; true_acc = true_acc % 4096  # prevent overflow
        
        if acc == true_acc: correct += 1
    
    print(f'  len={prog_len:2d}: {correct}/{n_test} ({correct/n_test*100:.0f}%)')
" 2>&1
