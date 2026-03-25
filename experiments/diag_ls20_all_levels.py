"""
Diagnostic: verify BFS solutions for LS20 levels 1-7 in practice.
Plays through all 7 levels sequentially using BFS solutions.
Reports which levels complete, which fail.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
os.environ['PYTHONUTF8'] = '1'

import importlib.util

# Load substrate module (computes BFS)
spec = importlib.util.spec_from_file_location('sub', 'B:/M/the-search/experiments/step1018b_ls20_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SOLUTIONS = mod._SOLUTIONS
ACTION_NAMES = mod.ACTION_NAMES

print("BFS solutions:")
for i, sol in enumerate(SOLUTIONS):
    if sol:
        print(f"  L{i+1}: {len(sol)} steps: {''.join(ACTION_NAMES[a][0] for a in sol)}")
    else:
        print(f"  L{i+1}: FAIL")

# Create LS20 env (using util_arcagi3 for correct reset behavior)
try:
    import arcagi3
    env = arcagi3.make('LS20')
    print("\nUsing arcagi3 (updated)")
except ImportError:
    import util_arcagi3
    env = util_arcagi3.make('LS20')
    print("\nUsing util_arcagi3")

# Run 3 seeds, play through all levels
for seed in range(3):
    print(f"\n=== Seed {seed} ===")
    obs = env.reset(seed=seed)

    current_level = 0  # 0-indexed: 0=L1, 1=L2, etc.
    total_steps = 0
    game_done = False

    while current_level < 7 and not game_done:
        sol = SOLUTIONS[current_level]
        level_name = f"L{current_level + 1}"

        if sol is None:
            print(f"  {level_name}: NO BFS SOLUTION — skipping")
            break

        print(f"  {level_name}: playing {len(sol)}-step solution...")

        completed = False
        died = False

        for step_i, action in enumerate(sol):
            obs, reward, done, info = env.step(action)
            total_steps += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0

            if cl > current_level:
                print(f"    ✓ Completed at step {step_i+1} (total {total_steps}), cl={cl}")
                completed = True
                current_level = cl
                break

            if done:
                if cl > current_level:
                    print(f"    ✓ Completed+done at step {step_i+1}, cl={cl}")
                    completed = True
                    current_level = cl
                else:
                    print(f"    ✗ DIED at step {step_i+1} (cl={cl}, expected L{current_level+1})")
                    died = True
                # Reset after done
                obs = env.reset(seed=seed)
                break

        if not completed and not died:
            # Played all solution steps but level not completed
            print(f"    ✗ SOLUTION EXHAUSTED after {len(sol)} steps, cl still {current_level}")
            # Try a few more steps to see if level completes
            extra_done = False
            for extra_i in range(10):
                obs, reward, done, info = env.step(0)  # UP
                total_steps += 1
                cl = info.get('level', 0) if isinstance(info, dict) else 0
                if cl > current_level:
                    print(f"    ✓ Completed with extra step {extra_i+1}, cl={cl}")
                    current_level = cl
                    extra_done = True
                    break
                if done:
                    obs = env.reset(seed=seed)
                    break
            if not extra_done:
                print(f"    ✗ FAILED — stopping")
                break
        elif died:
            print(f"    Restarting L{current_level+1} after death...")
            # Try once more after death
            if sol:
                completed2 = False
                for step_i, action in enumerate(sol):
                    obs, reward, done, info = env.step(action)
                    total_steps += 1
                    cl = info.get('level', 0) if isinstance(info, dict) else 0
                    if cl > current_level:
                        print(f"    ✓ Completed on retry at step {step_i+1}, cl={cl}")
                        completed2 = True
                        current_level = cl
                        break
                    if done:
                        obs = env.reset(seed=seed)
                        break
                if not completed2:
                    print(f"    ✗ FAILED on retry — stopping")
                    break

    if current_level >= 7:
        print(f"  ALL 7 LEVELS COMPLETED! total_steps={total_steps}")
    else:
        print(f"  Reached L{current_level+1} (of 7), stopped at total_steps={total_steps}")
