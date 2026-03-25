"""
Diagnostic: trace chain execution for seed 1 to find LS20 L1 failure.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
os.environ['PYTHONUTF8'] = '1'

import numpy as np

# Load substrate module
import importlib.util
spec = importlib.util.spec_from_file_location('sub', 'B:/M/the-search/experiments/step1018b_ls20_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from substrates.chain import ArcGameWrapper, SplitCIFAR100Wrapper, ChainRunner, make_prism, _substrate_n_actions, _substrate_reset

# Simulate seed 1's chain order: CIFAR, FT09, CIFAR, VC33, LS20
# (based on log: phases 1,3 are CIFAR; phases 2,4 are FT09/VC33; phase 5 is LS20)

# Create substrate for seed 1
sub = mod.Ls20SolverSubstrate(seed=0)
print(f"Initial: _is_ls20={sub._is_ls20}, _level_idx={sub._level_idx}, _step_in_seq={sub._step_in_seq}")

# Run fake CIFAR (just simulate on_level_transition calls)
print("\n--- Simulating CIFAR-before ---")
print(f"Before: _is_ls20={sub._is_ls20}, _level_idx={sub._level_idx}, _step_in_seq={sub._step_in_seq}")
for i in range(20):
    sub.on_level_transition()
print(f"After: _is_ls20={sub._is_ls20}, _level_idx={sub._level_idx}, _step_in_seq={sub._step_in_seq}")

# Simulate FT09 (calls set_game(68) then runs, calling on_level_transition on done)
print("\n--- Simulating FT09 ---")
sub.set_game(68)
print(f"After set_game(68): _is_ls20={sub._is_ls20}, _level_idx={sub._level_idx}, _step_in_seq={sub._step_in_seq}")
# FT09 might call on_level_transition on done
sub.on_level_transition()  # simulated done
print(f"After on_level_transition: _is_ls20={sub._is_ls20}, _level_idx={sub._level_idx}, _step_in_seq={sub._step_in_seq}")

# Simulate CIFAR-after
print("\n--- Simulating CIFAR-after ---")
print(f"Before: _is_ls20={sub._is_ls20}, _level_idx={sub._level_idx}, _step_in_seq={sub._step_in_seq}")
for i in range(20):
    sub.on_level_transition()
print(f"After: _is_ls20={sub._is_ls20}, _level_idx={sub._level_idx}, _step_in_seq={sub._step_in_seq}")

# Simulate VC33
print("\n--- Simulating VC33 ---")
sub.set_game(68)
print(f"After set_game(68): _is_ls20={sub._is_ls20}, _level_idx={sub._level_idx}, _step_in_seq={sub._step_in_seq}")
sub.on_level_transition()
print(f"After on_level_transition: _is_ls20={sub._is_ls20}, _level_idx={sub._level_idx}, _step_in_seq={sub._step_in_seq}")

# Now LS20
print("\n--- LS20 ---")
sub.set_game(4)
print(f"After set_game(4): _is_ls20={sub._is_ls20}, _level_idx={sub._level_idx}, _step_in_seq={sub._step_in_seq}")

# Run LS20 L1 BFS
try:
    import arcagi3
    env = arcagi3.make('LS20')
except:
    import util_arcagi3
    env = util_arcagi3.make('LS20')

seed = 1
_substrate_reset(sub, seed)
obs = env.reset(seed=seed)
level = 0
l1_step = None
steps = 0
fresh_episode = True

SOL_L1 = mod._SOLUTIONS[0]
print(f"Playing L1 BFS ({len(SOL_L1)} steps)...")

# Just play enough steps to check L1
for i in range(20):
    action = sub.process(np.array([obs] if not hasattr(obs, 'shape') else obs, dtype=np.float32) if not isinstance(obs, np.ndarray) else obs.astype(np.float32))
    obs, reward, done, info = env.step(action % 4)
    steps += 1

    if fresh_episode:
        fresh_episode = False
        print(f"  Step {steps}: FRESH EPISODE (action={action}), cl={info.get('level',0)}, done={done}")
        continue

    cl = info.get('level', 0) if isinstance(info, dict) else 0
    print(f"  Step {steps}: action={action}, cl={cl}, done={done}, _step_in_seq={sub._step_in_seq}")

    if cl > level:
        print(f"  *** LEVEL ADVANCED: {level} -> {cl} ***")
        if cl == 1 and l1_step is None:
            l1_step = steps
        level = cl
        sub.on_level_transition()
        print(f"  After on_level_transition: _level_idx={sub._level_idx}, _step_in_seq={sub._step_in_seq}")

    if done:
        print(f"  *** DONE at step {steps} ***")
        break

    if steps >= 20:
        break

print(f"\nResult: l1_step={l1_step}, level={level}")
