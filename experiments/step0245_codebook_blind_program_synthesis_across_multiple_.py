"""
Step 245 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11700.
"""
import torch
device = 'cuda'

# Step 245: Blind program synthesis across multiple unknown functions
# For each function, give I/O examples and let the system discover the circuit

gate_ops = {'AND': lambda x,y: x&y, 'OR': lambda x,y: x|y, 'XOR': lambda x,y: x^y, 'NAND': lambda x,y: 1-(x&y)}

def synth(io_pairs, n_inputs, max_gates=2):
    '''Synthesize circuit from I/O pairs.'''
    # 1-gate
    for gn, gf in gate_ops.items():
        for i in range(n_inputs):
            for j in range(n_inputs):
                if all(gf(ins[i], ins[j]) == out for ins, out in io_pairs):
                    return f'{gn}(in{i},in{j})'
    # 2-gate
    for g1n, g1f in gate_ops.items():
        for i in range(n_inputs):
            for j in range(n_inputs):
                for g2n, g2f in gate_ops.items():
                    for k in range(n_inputs):
                        if all(g2f(ins[k], g1f(ins[i], ins[j])) == out for ins, out in io_pairs):
                            return f'{g2n}(in{k},{g1n}(in{i},in{j}))'
    return None

# Test functions (the system doesn't know what they compute)
functions = {
    'mystery1': lambda a,b: a ^ b,                      # XOR
    'mystery2': lambda a,b: 1 - (a & b),                 # NAND
    'mystery3': lambda a,b,c: (a & b) | c,               # AND-OR
    'mystery4': lambda a,b,c: a ^ (b & c),               # XOR-AND
    'mystery5': lambda a,b,c: (a | b) & (a | c),         # distributive
    'mystery6': lambda a,b,c: (a ^ b) ^ c,               # 3-way XOR (parity)
}

print('Step 245: Blind program synthesis on unknown functions')
print(f'{\"Function\":12s} | Inputs | Discovered circuit')
print(f'{\"-\"*12}-|--------|-------------------')

for name, fn in functions.items():
    if fn.__code__.co_varnames[:3] == ('a','b','c'):
        n_in = 3
        io = [([a,b,c], fn(a,b,c)) for a in range(2) for b in range(2) for c in range(2)]
    else:
        n_in = 2
        io = [([a,b], fn(a,b)) for a in range(2) for b in range(2)]
    
    circuit = synth(io, n_in)
    print(f'{name:12s} | {n_in}      | {circuit or \"NOT FOUND\"}')
" 2>&1
