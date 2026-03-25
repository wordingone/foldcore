"""
Trace L3 BFS solution step by step.
Shows: step counter (sleft), position implied by action, trigger/goal events.
Goal: find why 52-step BFS solution fails in practice.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
os.environ['PYTHONUTF8'] = '1'

import importlib.util, numpy as np

spec = importlib.util.spec_from_file_location('sub', 'B:/M/the-search/experiments/step1018b_ls20_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SOLUTIONS = mod._SOLUTIONS
ACTION_NAMES = mod.ACTION_NAMES
SOL_L3 = SOLUTIONS[2]  # 0-indexed: L3 = index 2
print(f"L3 BFS solution ({len(SOL_L3)} steps): {''.join(ACTION_NAMES[a][0] for a in SOL_L3)}")

# Create LS20 env
import arcagi3
env = arcagi3.make('LS20')

# Play L1 and L2 first to get to L3
print("\nPlaying L1...")
obs = env.reset(seed=0)
for action in SOLUTIONS[0]:
    obs, reward, done, info = env.step(action)
print(f"  After L1: cl={info.get('level',0)}")

print("Playing L2...")
for action in SOLUTIONS[1]:
    obs, reward, done, info = env.step(action)
print(f"  After L2: cl={info.get('level',0)}")

# Now trace L3
print(f"\n=== L3 trace (moves_per_life=21) ===")
print(f"Step  Action  cl  done  notes")

# We need to observe the step counter. We can read it from the game's internal state.
# The env wraps arcengine which has the game object. Let's try to access it.
game = env._env._game if hasattr(env._env, '_game') else None
if game is None:
    # Try to find game object
    try:
        game = env._env._env._game
    except:
        pass

def get_step_counter(env):
    """Try to read step counter from game internals."""
    try:
        g = env._env._game
        return g._step_counter_ui.osgviligwp
    except:
        try:
            g = env._env
            return g._game._step_counter_ui.osgviligwp
        except:
            return None

# Actually let's just trace actions and cl/done
cl_before = 2
for i, action in enumerate(SOL_L3):
    obs, reward, done, info = env.step(action)
    cl = info.get('level', 0) if isinstance(info, dict) else 0

    notes = []
    if cl > cl_before:
        notes.append(f"LEVEL ADVANCE {cl_before}->{cl}")
        cl_before = cl
    if done:
        notes.append("DONE")
        # After done, reset for fresh start
        obs = env.reset(seed=0)
        notes.append("RESET")

    print(f"  {i+1:3d}  {ACTION_NAMES[action][0]:5s}   cl={cl}  done={done}  {' '.join(notes)}")

    if done and cl <= 2:
        print(f"\n  *** DIED at step {i+1} — L3 not completed ***")
        print(f"  Remaining solution: {''.join(ACTION_NAMES[a][0] for a in SOL_L3[i+1:])}")
        break

    if cl > 2:
        print(f"\n  *** L3 COMPLETED at step {i+1} ***")
        break

print("\n=== L3 BFS level data (from mock parse) ===")
# Parse the level data the BFS used
prev = mod._install_mock_arcengine()
import importlib
ls20_path = None
import glob
matches = glob.glob('B:/M/the-search/environment_files/ls20/**/ls20.py', recursive=True)
if matches:
    ls20_path = matches[0]
    print(f"ls20.py: {ls20_path}")

    ls20_spec = importlib.util.spec_from_file_location('ls20_game', ls20_path)
    ls20_mod = importlib.util.module_from_spec(ls20_spec)
    ls20_spec.loader.exec_module(ls20_mod)

    game_obj = ls20_mod.Ls20Game()
    levels = game_obj._levels
    print(f"  Total levels: {len(levels)}")

    # L3 = index 2
    ld = mod._extract_level_data(game_obj, 2)
    print(f"\nL3 level data:")
    print(f"  player: {ld['px0']},{ld['py0']}")
    print(f"  goals: {ld['goals']}")
    print(f"  n_goals: {ld['n_goals']}")
    print(f"  moves_per_life: {ld['moves_per_life']}")
    print(f"  rot_triggers: {ld['rot_triggers']}")
    print(f"  col_triggers: {ld['color_triggers']}")
    print(f"  shp_triggers: {ld['shape_triggers']}")
    print(f"  collectibles: {ld['collectibles']}")
    print(f"  walls ({len(ld['walls'])} total): {ld['walls'][:10]}...")
    print(f"  start: shape={ld['start_shape']}, color={ld['start_color']}, rot={ld['start_rot']}")
