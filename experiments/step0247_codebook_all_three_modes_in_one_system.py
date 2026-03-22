"""
Step 247 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11741.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 247: ALL THREE MODES in one system
# Task: given a stream of (operation, operand) pairs, compute the result
# The system must:
# 1. DISCOVER features to classify the operation type (feature discovery)
# 2. SYNTHESIZE the circuit for each operation (program synthesis)
# 3. ITERATE to process the stream (iterated computation)

# Operations encoded as noisy continuous features (not clean integers)
# The system must first CLASSIFY, then COMPUTE

# Encode operation type as noisy 4-dim vector
op_templates = {
    0: torch.tensor([1.0, 0.0, 0.0, 0.0]),  # ADD
    1: torch.tensor([0.0, 1.0, 0.0, 0.0]),  # SUB
    2: torch.tensor([0.0, 0.0, 1.0, 0.0]),  # DOUBLE (multiply by 2)
}

n_train = 500
# Training data: (op_noisy, operand) -> result
X_tr = []; y_tr = []
for _ in range(n_train):
    op = torch.randint(0, 3, (1,)).item()
    operand = torch.randint(0, 8, (1,)).item()
    acc = torch.randint(0, 8, (1,)).item()
    op_vec = op_templates[op] + torch.randn(4) * 0.2
    feat = torch.cat([op_vec, torch.tensor([float(acc), float(operand)])])
    
    if op == 0: result = acc + operand  # ADD
    elif op == 1: result = max(0, acc - operand)  # SUB
    else: result = acc * 2  # DOUBLE
    
    X_tr.append(feat); y_tr.append(result)

X_tr = torch.stack(X_tr).to(device)
y_tr = torch.tensor(y_tr, device=device, dtype=torch.long)

# Mode 1: Feature discovery to improve operation classification
templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b)}

def loo(V, labels, n_cls, k=5):
    V_n=F.normalize(V,dim=1);sims=V_n@V_n.T;sims.fill_diagonal_(-1e9)
    scores=torch.zeros(V.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(k,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==labels).float().mean().item()

n_cls = y_tr.max().item() + 1
V = X_tr.clone(); layers = []
for _ in range(3):
    cd=V.shape[1]; bl=loo(V,y_tr,n_cls); best=None
    for tn,tf in templates.items():
        for _ in range(30):
            w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*n_cls
            try:
                feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                l=loo(aug,y_tr,n_cls)
                if l>bl+0.003:bl=l;best=(tn,w.clone(),b.clone())
            except:pass
    if best is None:break
    tn,w,b=best;layers.append((tn,w,b))
    V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)

# Mode 3: Execute a multi-step program using k-NN classification
def predict_result(op_vec, acc, operand, V_db, y_db):
    feat = torch.cat([op_vec, torch.tensor([float(acc), float(operand)], device=device)])
    # Apply discovered features
    aug = feat.unsqueeze(0)
    for tn,w,b in layers:
        aug = torch.cat([aug, templates[tn](aug,w,b).unsqueeze(1)],1)
    
    sims = F.normalize(aug,dim=1) @ F.normalize(V_db,dim=1).T
    scores = torch.zeros(1, n_cls, device=device)
    for c in range(n_cls):
        m=y_db==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return scores.argmax(1).item()

# Test: execute a multi-step program
V_db = V  # augmented training codebook
programs = [
    [(0, 3), (0, 5), (1, 2)],          # ADD 3, ADD 5, SUB 2 → 6
    [(0, 7), (2, 0), (1, 4)],          # ADD 7, DOUBLE, SUB 4 → 10
    [(0, 4), (0, 4), (2, 0)],          # ADD 4, ADD 4, DOUBLE → 16
]

print('Step 247: Three modes combined — stream processing')
correct = 0; total = 0
for prog in programs:
    acc = 0
    for op, operand in prog:
        op_vec = (op_templates[op] + torch.randn(4)*0.1).to(device)
        acc = predict_result(op_vec, acc, operand, V_db, y_tr)
    
    # True result
    true_acc = 0
    for op, operand in prog:
        if op == 0: true_acc += operand
        elif op == 1: true_acc = max(0, true_acc - operand)
        else: true_acc *= 2
    
    names = {0:'ADD',1:'SUB',2:'DBL'}
    desc = ', '.join(f'{names[o]}({op})' for o,op in prog)
    ok = acc == true_acc
    total += 1; correct += int(ok)
    print(f'  {desc} = {acc} (true: {true_acc}) {\"OK\" if ok else \"FAIL\"}')

print(f'\\nOverall: {correct}/{total}')
print(f'Features discovered: {len(layers)}')
" 2>&1
