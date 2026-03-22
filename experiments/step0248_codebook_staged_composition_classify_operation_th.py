"""
Step 248 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11756.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 248: STAGED composition — classify operation, then compute result

# Full adder k-NN (proven in Steps 235-237)
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

def add_bin(a,b,nb=8):
    ab=[(a>>i)&1 for i in range(nb)]; bb=[(b>>i)&1 for i in range(nb)]; carry=0; r=[]
    for i in range(nb): s,carry=fa(ab[i],bb[i],carry); r.append(s)
    return sum(b*(2**i) for i,b in enumerate(r+[carry]))

# Stage 1: Operation classifier (k-NN on operation vectors)
op_templates = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}  # ADD, SUB, DOUBLE
n_train = 300
X_op = []; y_op = []
for _ in range(n_train):
    op = torch.randint(0, 3, (1,)).item()
    noisy = [op_templates[op][i] + torch.randn(1).item()*0.3 for i in range(3)]
    X_op.append(noisy); y_op.append(op)
X_op = torch.tensor(X_op, device=device); y_op = torch.tensor(y_op, device=device, dtype=torch.long)

def classify_op(op_vec):
    sims = F.normalize(op_vec.unsqueeze(0),dim=1) @ F.normalize(X_op,dim=1).T
    scores = torch.zeros(1, 3, device=device)
    for c in range(3):
        m = y_op == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
    return scores.argmax(1).item()

# Stage 2: Compute result using PROVEN arithmetic
def compute(op, acc, operand):
    if op == 0: return add_bin(acc, operand)  # ADD
    elif op == 1: return max(0, acc - operand)  # SUB (simplified)
    else: return add_bin(acc, acc)  # DOUBLE = acc + acc

# STAGED execution: classify THEN compute
programs = [
    [(0, 3), (0, 5), (1, 2)],     # ADD 3, ADD 5, SUB 2 → 6
    [(0, 7), (2, 0), (1, 4)],     # ADD 7, DOUBLE, SUB 4 → 10
    [(0, 4), (0, 4), (2, 0)],     # ADD 4, ADD 4, DOUBLE → 16
    [(0, 1), (2, 0), (2, 0), (2, 0)],  # ADD 1, DBL, DBL, DBL → 8
    [(0, 5), (1, 3), (0, 10), (2, 0)], # ADD 5, SUB 3, ADD 10, DBL → 24
]

print('Step 248: STAGED composition — classify then compute')
correct = total = 0
for prog in programs:
    acc = 0
    true_acc = 0
    for op_true, operand in prog:
        # Classify operation from noisy vector
        op_vec = torch.tensor([op_templates[op_true][i] + torch.randn(1).item()*0.2 for i in range(3)], device=device)
        op_pred = classify_op(op_vec)
        
        # Compute with classified operation
        acc = compute(op_pred, acc, operand)
        
        # True
        if op_true == 0: true_acc += operand
        elif op_true == 1: true_acc = max(0, true_acc - operand)
        else: true_acc *= 2
    
    ok = acc == true_acc
    total += 1; correct += int(ok)
    names = {0:'ADD',1:'SUB',2:'DBL'}
    desc = ', '.join(f'{names[o]}({op})' for o,op in prog)
    print(f'  {desc} = {acc} (true: {true_acc}) {\"OK\" if ok else \"FAIL\"}')

print(f'\\nOverall: {correct}/{total} ({correct/total*100:.0f}%)')
" 2>&1
