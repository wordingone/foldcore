"""
Step 1036 — Atomic Action-Influence Substrate (Debate: Prosecution Experiment)

DEBATE WIN CONDITION: Whichever side produces a working substrate with RHAE data wins.
SPIRIT CLAUSE: One system, one config, discovers from interaction. No hardcoded pipelines.

REVISED SPEC (Leo mail #3066): Action-influence maps handle ALL game types.
The substrate discovers which actions affect which pixels from interaction.

Mechanism:
  change_map: 64x64 float -- per-pixel running change frequency (WHERE)
  target_map: 64x64 float -- per-pixel running mean (goal inference)
  influence: n_actions x 64x64 float -- per-action pixel influence (HOW)
  Per step:
    1. diff = |obs - prev_obs|
    2. change_map = alpha_c * change_map + (1-alpha_c) * diff
    3. influence[prev_action] = alpha_i * influence[prev_action] + (1-alpha_i) * diff
    4. target_map = alpha_t * target_map + (1-alpha_t) * obs
    5. mismatch = |obs - target_map|
    6. score[a] = sum(influence[a] * change_map * mismatch) for each action
    7. action = argmax(score) + epsilon exploration

R3 HYPOTHESIS: Running statistics are l_0 (data changes, operations don't).
  Falsified if: influence maps never converge to meaningful action-pixel mappings.

Defense concerns:
  1. target_map (running mean) != goal state
  2. Influence maps for 71 actions need many trials to stabilize
  3. score = influence * change_map * mismatch may be noisy

Action space (game-agnostic, one config):
  0-63:  ACTION6 click at 8x8 grid positions
  64-70: ACTION1 through ACTION7 (keyboard/other)
  Total: 71 actions. Substrate discovers which are useful from interaction.

All 3 games, randomized order per seed, 5 seeds, 10K steps/game.
KILL: No L2+ on any game.
SUCCESS: L2+ on any game with one config.
"""
import os, sys, time
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import arc_agi
from arcengine import GameAction, GameState

# --- Config (ONE config, all games) ---
ALPHA_CHANGE = 0.99
ALPHA_TARGET = 0.999
ALPHA_INFLUENCE = 0.1
KERNEL = 5
EPSILON_WARMUP = 0.3    # exploration rate during warmup
EPSILON_FINAL = 0.05    # exploration rate after warmup
WARMUP_STEPS = 1000     # steps before reducing epsilon
MAX_STEPS = 10_000
TIME_CAP = 120  # seconds per game

# --- Action space (game-agnostic) ---
GAME_ACTIONS = list(GameAction)
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]

# Actions 0-63: ACTION6 click at grid positions
# Actions 64-70: ACTION1 through ACTION7 (keyboard)
N_ACTIONS = 64 + 7  # = 71

def execute_action(env, action_idx):
    """Execute action by index. Returns obs."""
    if action_idx < 64:
        # Click action
        cx, cy = CLICK_GRID[action_idx]
        return env.step(GAME_ACTIONS[6], data={"x": cx, "y": cy})
    else:
        # Keyboard action (ACTION1-ACTION7)
        ga_idx = action_idx - 64 + 1  # 1-7
        return env.step(GAME_ACTIONS[ga_idx], data={})


# --- Atomic Substrate ---
class AtomicInfluenceSubstrate:
    """
    Single update rule per step. No phases, no clustering, no game-specific logic.
    Discovers WHERE (change_map), HOW (influence), WHAT (target_map) from interaction.
    """

    def __init__(self, rng):
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.target_map = None
        self.prev_obs = None
        self.prev_action = None
        self.influence = np.zeros((N_ACTIONS, 64, 64), dtype=np.float32)
        self.rng = rng
        self.step_count = 0

    def step(self, frame):
        """Process one frame, return action index (0..N_ACTIONS-1)."""
        arr = np.array(frame[0], dtype=np.float32)
        self.step_count += 1

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self.target_map = arr.copy()
            # Random first action
            action = self.rng.randint(N_ACTIONS)
            self.prev_action = action
            return action

        # 1. Per-pixel absolute change
        diff = np.abs(arr - self.prev_obs)

        # 2. WHERE: running change frequency
        self.change_map = ALPHA_CHANGE * self.change_map + (1.0 - ALPHA_CHANGE) * diff

        # 3. HOW: per-action influence map (what did prev_action change?)
        if self.prev_action is not None:
            self.influence[self.prev_action] = (
                (1.0 - ALPHA_INFLUENCE) * self.influence[self.prev_action]
                + ALPHA_INFLUENCE * diff
            )

        # 4. WHAT: running mean of observations
        self.target_map = ALPHA_TARGET * self.target_map + (1.0 - ALPHA_TARGET) * arr

        # 5. Mismatch: deviation from running mean
        mismatch = np.abs(arr - self.target_map)

        # 6. Score each action: how much does it affect wrong interactive regions?
        cm_mismatch = self.change_map * mismatch  # interactive AND wrong
        cm_smooth = uniform_filter(cm_mismatch, size=KERNEL)

        scores = np.zeros(N_ACTIONS, dtype=np.float32)
        for a in range(N_ACTIONS):
            scores[a] = np.sum(self.influence[a] * cm_smooth)

        # 7. Epsilon-greedy action selection
        eps = EPSILON_WARMUP if self.step_count < WARMUP_STEPS else EPSILON_FINAL
        if self.rng.random() < eps or np.max(scores) == 0:
            action = self.rng.randint(N_ACTIONS)
        else:
            action = int(np.argmax(scores))

        # Update prev
        self.prev_obs = arr.copy()
        self.prev_action = action

        return action


# --- Run one game ---
def run_game(arc, game_id, seed, substrate):
    np.random.seed(seed)
    env = arc.make(game_id)

    obs = env.reset()
    t0 = time.time()
    max_levels = 0
    level_steps = {}
    total_steps = 0
    action_counts = np.zeros(N_ACTIONS, dtype=int)

    for step_i in range(MAX_STEPS):
        if time.time() - t0 > TIME_CAP:
            break

        if obs is None or obs.state == GameState.GAME_OVER:
            obs = env.reset()
            continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset()
            continue

        action_idx = substrate.step(obs.frame)
        action_counts[action_idx] += 1
        total_steps += 1

        levels_before = obs.levels_completed
        obs = execute_action(env, action_idx)

        if obs and obs.levels_completed > levels_before:
            if obs.levels_completed > max_levels:
                max_levels = obs.levels_completed
                level_steps[max_levels] = step_i + 1

    elapsed = time.time() - t0

    # Action type breakdown
    n_click = int(action_counts[:64].sum())
    n_keyboard = int(action_counts[64:].sum())

    return {
        'max_levels': max_levels,
        'level_steps': level_steps,
        'steps': total_steps,
        'elapsed': round(elapsed, 1),
        'n_click': n_click,
        'n_keyboard': n_keyboard,
    }


# --- Main ---
def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()

    # Deduplicate games
    games = {}
    for e in envs:
        gname = e.game_id.split('-')[0]
        if gname not in games:
            games[gname] = e.game_id
    game_names = sorted(games.keys())

    print("=== Step 1036: Atomic Action-Influence Substrate (Debate Prosecution) ===")
    print("One process() function. One config. All games.")
    print(f"Config: alpha_change={ALPHA_CHANGE}, alpha_target={ALPHA_TARGET}, "
          f"alpha_influence={ALPHA_INFLUENCE}, kernel={KERNEL}")
    print(f"Actions: {N_ACTIONS} (64 click + 7 keyboard)")
    print(f"Games: {', '.join(f'{g} ({games[g]})' for g in game_names)}")
    print(f"Per game: {MAX_STEPS} steps, {TIME_CAP}s cap, 5 seeds")
    print()

    all_results = {g: [] for g in game_names}

    for seed in range(5):
        rng = np.random.RandomState(seed)
        order = list(game_names)
        rng.shuffle(order)
        print(f"--- Seed {seed} (order: {' > '.join(order)}) ---")

        substrate = AtomicInfluenceSubstrate(np.random.RandomState(seed))

        for gname in order:
            r = run_game(arc, games[gname], seed, substrate)
            lvl_str = ""
            if r['level_steps']:
                lvl_str = "  " + "  ".join(
                    f"L{l}@{s}" for l, s in sorted(r['level_steps'].items()))
            print(f"  {gname}: L={r['max_levels']}  steps={r['steps']}  "
                  f"click={r['n_click']} kb={r['n_keyboard']}  "
                  f"{r['elapsed']}s{lvl_str}")
            all_results[gname].append(r)

    # Summary
    print("\n=== Summary ===")
    any_l2 = False
    for gname in sorted(all_results.keys()):
        results = all_results[gname]
        counts = {}
        for lvl in range(1, 6):
            c = sum(1 for r in results if r['max_levels'] >= lvl)
            if c > 0:
                counts[lvl] = c
        max_l = max(r['max_levels'] for r in results)
        parts = [f"L{l}: {c}/5" for l, c in sorted(counts.items())]
        avg_click = np.mean([r['n_click'] for r in results])
        avg_kb = np.mean([r['n_keyboard'] for r in results])
        print(f"  {gname}: {', '.join(parts) if parts else 'L0 only'}  "
              f"(max={max_l}, avg click={avg_click:.0f}, avg kb={avg_kb:.0f})")
        if max_l >= 2:
            any_l2 = True

    if any_l2:
        print("\n  SIGNAL: Atomic substrate produces L2+!")
        print("  Prosecution substrate has RHAE data.")
    else:
        print("\n  FAIL: No L2+ on any game.")
        print("  Prosecution substrate does not produce RHAE data.")

    print("\nStep 1036 DONE")


if __name__ == "__main__":
    main()
