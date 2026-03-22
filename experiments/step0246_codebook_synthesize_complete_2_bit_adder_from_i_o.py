"""
Step 246 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11720.
"""
import torch

# Step 246: Synthesize COMPLETE 2-bit adder from I/O examples
# Input: a1,a0,b1,b0 (4 bits). Output: s2,s1,s0 (3 bits for sum 0-6)
# The system must discover ALL output bit circuits independently.

gate_ops = {'AND': lambda x,y: x&y, 'OR': lambda x,y: x|y, 'XOR': lambda x,y: x^y}

def synth_bit(io_pairs, n_inputs, max_depth=3):
    '''Synthesize circuit for one output bit.'''
    # Depth 1
    for gn, gf in gate_ops.items():
        for i in range(n_inputs):
            for j in range(n_inputs):
                if all(gf(ins[i], ins[j]) == out for ins, out in io_pairs):
                    return f'{gn}({i},{j})', lambda ins, gf=gf, i=i, j=j: gf(ins[i], ins[j])

    # Depth 2
    for g1n, g1f in gate_ops.items():
        for i in range(n_inputs):
            for j in range(n_inputs):
                for g2n, g2f in gate_ops.items():
                    for k in range(n_inputs):
                        if all(g2f(ins[k], g1f(ins[i], ins[j])) == out for ins, out in io_pairs):
                            return f'{g2n}({k},{g1n}({i},{j}))', lambda ins, g2f=g2f, g1f=g1f, i=i, j=j, k=k: g2f(ins[k], g1f(ins[i], ins[j]))

    # Depth 3 with intermediate wire reuse
    for g1n, g1f in gate_ops.items():
        for i in range(n_inputs):
            for j in range(n_inputs):
                for g2n, g2f in gate_ops.items():
                    for k in range(n_inputs):
                        for g3n, g3f in gate_ops.items():
                            for l in range(n_inputs):
                                # g3(l, g2(k, g1(i,j)))
                                if all(g3f(ins[l], g2f(ins[k], g1f(ins[i], ins[j]))) == out for ins, out in io_pairs):
                                    return f'{g3n}({l},{g2n}({k},{g1n}({i},{j})))', None
                            # g3(g1(i,j), g2(k,l))
                            for l in range(n_inputs):
                                if all(g3f(g1f(ins[i], ins[j]), g2f(ins[k], ins[l])) == out for ins, out in io_pairs):
                                    return f'{g3n}({g1n}({i},{j}),{g2n}({k},{l}))', None
    return None, None

# 2-bit addition: a (2 bits) + b (2 bits) = sum (3 bits)
# Inputs: [a0, a1, b0, b1]
# Outputs: [s0, s1, s2]
n_inputs = 4
io_all = []
for a in range(4):
    for b in range(4):
        ins = [(a>>0)&1, (a>>1)&1, (b>>0)&1, (b>>1)&1]
        s = a + b
        outs = [(s>>0)&1, (s>>1)&1, (s>>2)&1]
        io_all.append((ins, outs))

print('Step 246: Synthesize 2-bit adder from I/O examples')
print(f'Input: a0,a1,b0,b1 -> Output: s0,s1,s2')
print(f'{len(io_all)} I/O pairs (all 16 combinations)')
print()

for bit in range(3):
    io_bit = [(ins, outs[bit]) for ins, outs in io_all]
    circuit, _ = synth_bit(io_bit, n_inputs)
    print(f'  Output bit {bit}: {circuit or \"NOT FOUND (needs >3 gates)\"}')

# Verify: do the discovered circuits produce correct addition?
print(f'\\nNote: input mapping: 0=a0, 1=a1, 2=b0, 3=b1')
" 2>&1
