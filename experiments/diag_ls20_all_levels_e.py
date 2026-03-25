"""
Verify 1018e BFS solutions work in practice for all 7 levels.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
os.environ['PYTHONUTF8'] = '1'

import importlib.util

spec = importlib.util.spec_from_file_location('sub', 'B:/M/the-search/experiments/step1018e_ls20_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SOLUTIONS = mod._SOLUTIONS
ACTION_NAMES = mod.ACTION_NAMES

import arcagi3
env = arcagi3.make('LS20')

for seed in range(3):
    print(f"\n=== Seed {seed} ===")
    obs = env.reset(seed=seed)
    current_level = 0
    total_steps = 0

    while current_level < 7:
        sol = SOLUTIONS[current_level]
        level_name = f"L{current_level + 1}"

        if sol is None:
            print(f"  {level_name}: NO SOLUTION")
            break

        print(f"  {level_name}: playing {len(sol)}-step solution...", end=' ', flush=True)
        completed = False

        for step_i, action in enumerate(sol):
            obs, reward, done, info = env.step(action)
            total_steps += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0

            if cl > current_level:
                print(f"DONE at step {step_i+1} (total {total_steps})")
                completed = True
                current_level = cl
                break

            if done and cl <= current_level:
                print(f"DIED at step {step_i+1}!")
                obs = env.reset(seed=seed)
                break

        if not completed:
            # Try 5 more steps
            found = False
            for extra_i in range(5):
                obs, reward, done, info = env.step(0)
                total_steps += 1
                cl = info.get('level', 0) if isinstance(info, dict) else 0
                if cl > current_level:
                    print(f"DONE with extra step {extra_i+1}")
                    current_level = cl
                    found = True
                    break
                if done:
                    obs = env.reset(seed=seed)
                    break
            if not found:
                print(f"FAILED after {len(sol)} steps")
                break

    if current_level >= 7:
        print(f"  ALL 7 LEVELS! total_steps={total_steps}")
    else:
        print(f"  Reached L{current_level+1}/7, steps={total_steps}")

print("\nSolution lengths:", [len(s) if s else 0 for s in SOLUTIONS])
