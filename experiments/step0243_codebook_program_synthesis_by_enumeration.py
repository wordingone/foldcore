"""
Step 243 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11639.
"""
import torch
device = 'cuda'

# Step 243: Program synthesis by enumeration
# Given I/O examples of addition, can we FIND the ripple-carry program
# by enumerating small programs and testing them?

# Primitive operations available to programs:
# AND(a,b), OR(a,b), XOR(a,b), NOT(a)
# These are the building blocks of a full adder

# Target: full adder (a, b, cin) -> (sum, carry_out)
# sum = a XOR b XOR cin
# carry = (a AND b) OR (cin AND (a XOR b))

# Generate I/O examples
io_pairs = []
for a in range(2):
    for b in range(2):
        for cin in range(2):
            s = (a + b + cin) % 2
            cout = (a + b + cin) // 2
            io_pairs.append(((a, b, cin), (s, cout)))

print('Step 243: Program synthesis — discover full adder from I/O')
print('Target truth table:')
for (a,b,c), (s,co) in io_pairs:
    print(f'  ({a},{b},{c}) -> sum={s}, carry={co}')

# Enumerate all possible 2-gate circuits with 3 inputs
# Gate types: AND, OR, XOR
# Circuit: gate1(inp1, inp2) = wire4, gate2(inp3, inp4) = output
# Inputs: 0=a, 1=b, 2=cin, 3=gate1_output

gate_ops = {
    'AND': lambda x,y: x&y,
    'OR': lambda x,y: x|y,
    'XOR': lambda x,y: x^y,
}

# Search for SUM circuit (1 or 2 gates)
print('\\nSearching for SUM circuit...')
inputs = ['a', 'b', 'cin']

# 1-gate circuits
for gname, gfn in gate_ops.items():
    for i in range(3):
        for j in range(3):
            correct = all(gfn(io[0][i], io[0][j]) == io[1][0] for io in io_pairs)
            if correct:
                print(f'  FOUND 1-gate: {gname}({inputs[i]}, {inputs[j]}) = sum')

# 2-gate circuits: gate1(i,j) then gate2(k, gate1_out)
for g1name, g1fn in gate_ops.items():
    for i in range(3):
        for j in range(3):
            for g2name, g2fn in gate_ops.items():
                for k in range(3):
                    correct = True
                    for (a,b,c), (s,co) in io_pairs:
                        vals = [a, b, c]
                        w1 = g1fn(vals[i], vals[j])
                        out = g2fn(vals[k], w1)
                        if out != s: correct = False; break
                    if correct:
                        print(f'  FOUND 2-gate: {g2name}({inputs[k]}, {g1name}({inputs[i]}, {inputs[j]})) = sum')

# Search for CARRY circuit
print('\\nSearching for CARRY circuit...')
# 2-gate circuits for carry
found_carry = False
for g1name, g1fn in gate_ops.items():
    for i in range(3):
        for j in range(3):
            for g2name, g2fn in gate_ops.items():
                for k in range(3):
                    correct = True
                    for (a,b,c), (s,co) in io_pairs:
                        vals = [a, b, c]
                        w1 = g1fn(vals[i], vals[j])
                        out = g2fn(vals[k], w1)
                        if out != co: correct = False; break
                    if correct and not found_carry:
                        print(f'  FOUND 2-gate: {g2name}({inputs[k]}, {g1name}({inputs[i]}, {inputs[j]})) = carry')
                        found_carry = True

if not found_carry:
    # 3-gate circuits for carry: g1(i,j), g2(k,l), g3(g1,g2)
    print('  Trying 3-gate circuits...')
    for g1n, g1f in gate_ops.items():
        for i in range(3):
            for j in range(i,3):
                for g2n, g2f in gate_ops.items():
                    for k in range(3):
                        for l in range(k,3):
                            for g3n, g3f in gate_ops.items():
                                correct = True
                                for (a,b,c), (s,co) in io_pairs:
                                    vals = [a,b,c]
                                    w1 = g1f(vals[i], vals[j])
                                    w2 = g2f(vals[k], vals[l])
                                    out = g3f(w1, w2)
                                    if out != co: correct = False; break
                                if correct:
                                    print(f'  FOUND 3-gate: {g3n}({g1n}({inputs[i]},{inputs[j]}), {g2n}({inputs[k]},{inputs[l]})) = carry')
                                    found_carry = True
                                    break
                            if found_carry: break
                        if found_carry: break
                    if found_carry: break
                if found_carry: break
            if found_carry: break
        if found_carry: break
" 2>&1
