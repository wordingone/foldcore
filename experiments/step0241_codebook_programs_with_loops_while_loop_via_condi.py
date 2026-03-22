"""
Step 241 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11599.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 241: Programs with LOOPS — WHILE loop via conditional jump-back
# Instructions: ADD1, ADD2, SUB1, NOP, IF_POS(skip), IF_ZERO(skip), JUMP_BACK(n)
# WHILE(acc>0){body} = body, IF_POS, JUMP_BACK(len_body+1)

# Full adder (proven, reused)
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
    return from_bits(r+[carry])

# Inst: (opcode, arg)
# 0=ADD(arg), 1=SUB(arg), 2=NOP, 3=IF_POS(skip next), 4=IF_ZERO(skip next), 5=JUMP_BACK(arg)
def execute(program, max_steps=1000, nb=8):
    acc = 0; ip = 0; steps = 0
    while ip < len(program) and steps < max_steps:
        op, arg = program[ip]
        if op == 0: acc = add_bin(acc, arg, nb)
        elif op == 1: acc = max(0, acc - arg)
        elif op == 2: pass
        elif op == 3:
            if acc > 0: ip += 1
        elif op == 4:
            if acc == 0: ip += 1
        elif op == 5:
            ip -= arg; continue
        ip += 1; steps += 1
    return acc

def true_exec(program, max_steps=1000):
    acc = 0; ip = 0; steps = 0
    while ip < len(program) and steps < max_steps:
        op, arg = program[ip]
        if op == 0: acc += arg
        elif op == 1: acc = max(0, acc - arg)
        elif op == 3:
            if acc > 0: ip += 1
        elif op == 4:
            if acc == 0: ip += 1
        elif op == 5:
            ip -= arg; continue
        ip += 1; steps += 1
    return acc

# Test programs with loops
programs = [
    # acc=5, WHILE(acc>0) {SUB1} → acc should be 0
    ([(0,5), (1,1), (3,0), (5,2)], 'acc=5; while(acc>0) sub1'),
    
    # acc=3, loop: ADD2, SUB1, IF_POS jump back 2 → adds net 1 per iter
    # After 3 iters: 3 + 3*1 = 6? No — converges when SUB makes it 0
    # Actually: 3→5→4→6→5→7→6→... runs forever. Need finite version.
    
    # acc=0, ADD 10, loop 5 times: SUB 2 each
    ([(0,10), (1,2), (3,0), (5,2)], 'acc=10; while(acc>0) sub2'),
    
    # Simple: ADD 3 three times
    ([(0,3), (0,3), (0,3)], 'add3 x3 = 9'),
    
    # acc=7, SUB1 loop
    ([(0,7), (1,1), (3,0), (5,2)], 'acc=7; while(acc>0) sub1'),
]

print('Step 241: Programs with LOOPS')
for prog, desc in programs:
    pred = execute(prog)
    true = true_exec(prog)
    print(f'  {desc}: pred={pred} true={true} {\"OK\" if pred==true else \"FAIL\"}')
" 2>&1
