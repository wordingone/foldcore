"""
Diagnostic: trace LS20 Level 1 BFS solution step by step.
Checks what the game reports at each step.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

os.environ['PYTHONUTF8'] = '1'

# Load the substrate (computes BFS)
import importlib.util
spec = importlib.util.spec_from_file_location('sub', 'B:/M/the-search/experiments/step1018b_ls20_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SOLUTIONS = mod._SOLUTIONS
SOL_L1 = SOLUTIONS[0]
ACTION_NAMES = mod.ACTION_NAMES
print(f"L1 solution ({len(SOL_L1)} steps): {''.join(ACTION_NAMES[a][0] for a in SOL_L1)}")

# Create LS20 game env
try:
    import arcagi3
    env = arcagi3.make('LS20')
except ImportError:
    import util_arcagi3
    env = util_arcagi3.make('LS20')

# Run L1 solution for seed 0
for seed in range(3):
    print(f"\n=== Seed {seed} ===")
    obs = env.reset(seed=seed)
    print(f"  After reset: obs shape={obs.shape if hasattr(obs,'shape') else 'N/A'}")

    level = 0
    for i, action in enumerate(SOL_L1):
        obs, reward, done, info = env.step(action)
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        print(f"  Step {i+1:2d} ({ACTION_NAMES[action][0]}): cl={cl}, done={done}, reward={reward}")
        if cl > level:
            level = cl
            print(f"  *** LEVEL ADVANCED to {level} ***")
        if done:
            print(f"  *** DONE at step {i+1} ***")
            break

    print(f"  Final: level={level}")
    if level == 0:
        # Play extra steps to see if level advances
        print("  L1 not completed — playing 5 more steps...")
        for i in range(5):
            obs, reward, done, info = env.step(0)  # UP
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            print(f"  Extra step {i+1} (U): cl={cl}, done={done}")
            if done:
                break
