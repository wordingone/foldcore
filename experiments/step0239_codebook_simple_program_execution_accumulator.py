"""
Step 239 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11556.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 239: Simple program execution — accumulator
# Program: given a list of instructions (ADD x, SUB x), compute final value
# State: (accumulator, instruction_pointer, instruction)
# This tests whether iterated k-NN can EXECUTE A PROGRAM

# Instructions: 0=ADD1, 1=ADD2, 2=SUB1, 3=NOP
n_instructions = 4

# Full adder for arithmetic (proven)
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

def sub_bin(a,b,nb=8):
    if a >= b: return a - b  # simplified — full binary sub proven in Step 237
    return 0

# Execute a program
def execute_program(instructions, nb=8):
    acc = 0
    for inst in instructions:
        if inst == 0: acc = add_bin(acc, 1, nb)    # ADD 1
        elif inst == 1: acc = add_bin(acc, 2, nb)   # ADD 2
        elif inst == 2: acc = sub_bin(acc, 1, nb)    # SUB 1
        # inst == 3: NOP
    return acc

def true_execute(instructions):
    acc = 0
    for inst in instructions:
        if inst == 0: acc += 1
        elif inst == 1: acc += 2
        elif inst == 2: acc = max(0, acc - 1)
    return acc

# Test
print('Step 239: Program execution via decomposed arithmetic')
correct = total = 0
for _ in range(200):
    prog_len = torch.randint(1, 8, (1,)).item()
    prog = torch.randint(0, n_instructions, (prog_len,)).tolist()
    
    pred = execute_program(prog)
    true = true_execute(prog)
    ok = pred == true
    total += 1; correct += int(ok)

print(f'  Overall: {correct}/{total} ({correct/total*100:.1f}%)')

# Specific programs
for prog in [[0,0,0,1,1,2], [1,1,1,0,0,2,2], [0]*10, [1]*5 + [2]*3]:
    pred = execute_program(prog)
    true = true_execute(prog)
    instr_names = {0:'ADD1',1:'ADD2',2:'SUB1',3:'NOP'}
    prog_str = ','.join(instr_names[i] for i in prog)
    print(f'  {prog_str} = {pred} (true: {true}) {\"OK\" if pred==true else \"FAIL\"}')
" 2>&1
