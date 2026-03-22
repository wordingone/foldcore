"""
Step 240 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11579.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 240: Programs with conditionals
# Instructions: ADD1, ADD2, SUB1, NOP, IF_POS (skip next if acc>0), IF_ZERO (skip next if acc==0)
# This is a conditional branch — the substrate must learn control flow

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

def to_bits(n,nb): return [(n>>i)&1 for i in range(nb)]
def from_bits(bits): return sum(b*(2**i) for i,b in enumerate(bits))

def add_bin(a,b,nb=8):
    ab=to_bits(a,nb);bb=to_bits(b,nb);carry=0;r=[]
    for i in range(nb):
        s,carry=fa(ab[i],bb[i] if i<len(bb) else 0,carry);r.append(s)
    r.append(carry)
    return from_bits(r)

# Execute with conditionals
# 0=ADD1, 1=ADD2, 2=SUB1, 3=NOP, 4=IF_POS(skip next if acc>0), 5=IF_ZERO(skip next if acc==0)
def execute(program, nb=8):
    acc = 0; ip = 0
    while ip < len(program):
        inst = program[ip]
        if inst == 0: acc = add_bin(acc, 1, nb)
        elif inst == 1: acc = add_bin(acc, 2, nb)
        elif inst == 2: acc = max(0, acc - 1)
        elif inst == 3: pass  # NOP
        elif inst == 4:  # IF_POS: skip next if acc > 0
            if acc > 0: ip += 1
        elif inst == 5:  # IF_ZERO: skip next if acc == 0
            if acc == 0: ip += 1
        ip += 1
    return acc

def true_execute(program):
    acc = 0; ip = 0
    while ip < len(program):
        inst = program[ip]
        if inst == 0: acc += 1
        elif inst == 1: acc += 2
        elif inst == 2: acc = max(0, acc - 1)
        elif inst == 4:
            if acc > 0: ip += 1
        elif inst == 5:
            if acc == 0: ip += 1
        ip += 1
    return acc

# Test
correct = total = 0
for _ in range(300):
    prog_len = torch.randint(2, 10, (1,)).item()
    prog = torch.randint(0, 6, (prog_len,)).tolist()
    pred = execute(prog)
    true = true_execute(prog)
    ok = pred == true
    total += 1; correct += int(ok)

print(f'Step 240: Programs with conditionals (IF_POS, IF_ZERO)')
print(f'  Overall: {correct}/{total} ({correct/total*100:.1f}%)')

# Specific programs
programs = [
    [0, 0, 4, 1, 0],      # ADD1, ADD1, IF_POS(skip ADD2), ADD1 → should skip ADD2
    [5, 1, 0],             # IF_ZERO(skip ADD2), ADD1 → acc=0 so skip, result=1
    [0, 5, 1, 0],          # ADD1, IF_ZERO(skip ADD2), ADD1 → acc=1≠0, don't skip, result=4
    [2, 2, 4, 1, 0],       # SUB1, SUB1, IF_POS(skip ADD2), ADD1 → acc=0, don't skip, result=3
]
for prog in programs:
    pred = execute(prog)
    true = true_execute(prog)
    names = {0:'ADD1',1:'ADD2',2:'SUB1',3:'NOP',4:'IF+',5:'IF0'}
    s = ','.join(names[i] for i in prog)
    print(f'  {s} = {pred} (true: {true}) {\"OK\" if pred==true else \"FAIL\"}')
" 2>&1
