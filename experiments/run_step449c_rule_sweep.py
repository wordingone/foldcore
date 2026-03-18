#!/usr/bin/env python3
"""
Step 449c — CA-Graph rule sweep: Rules 30, 90, 110, 184.
Tests Wolfram classification prediction:
- Rule 30 (Class III chaotic): too sensitive, ratio ~1.0
- Rule 90 (Class II periodic): too insensitive or too ordered
- Rule 110 (Class IV edge-of-chaos): sweet spot, navigates
- Rule 184 (Class II traffic rule): more structured, may cluster

Same graph config as 449. 10K steps, 3 seeds per rule.
"""

import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


def make_rule_lookup(rule_num):
    return np.array([(rule_num >> i) & 1 for i in range(8)], dtype=np.uint8)


def ca_step(state, lookup):
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    return lookup[(left * 4 + state * 2 + right)]


def obs_to_cell(pooled, lookup, t_steps=10):
    state = (pooled > np.median(pooled)).astype(np.uint8)
    for _ in range(t_steps):
        state = ca_step(state, lookup)
    return hash(state.tobytes())


class CAGraph:
    def __init__(self, rule_lookup, n_actions=4, ca_steps=10):
        self.rule_lookup = rule_lookup
        self.n_actions = n_actions
        self.ca_steps = ca_steps
        self.edges = {}
        self.cells = set()
        self.prev_cell = None
        self.prev_action = None
        self.step_count = 0

    def step(self, obs):
        cell_id = obs_to_cell(obs, self.rule_lookup, self.ca_steps)
        self.cells.add(cell_id)
        self.step_count += 1

        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            if key not in self.edges:
                self.edges[key] = {}
            self.edges[key][cell_id] = self.edges[key].get(cell_id, 0) + 1

        visit_counts = [
            sum(self.edges.get((cell_id, a), {}).values())
            for a in range(self.n_actions)
        ]
        min_count = min(visit_counts)
        candidates = [a for a, c in enumerate(visit_counts) if c == min_count]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell_id
        self.prev_action = action
        return action


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def run_seed(arc, game_id, seed, rule_lookup, max_steps=10000):
    from arcengine import GameState
    np.random.seed(seed)
    g = CAGraph(rule_lookup=rule_lookup, n_actions=4, ca_steps=10)
    env = arc.make(game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    action_counts = [0] * na
    level_step = None
    t0 = time.time()

    while ts < max_steps:
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue

        pooled = avgpool16(obs.frame)
        action_idx = g.step(pooled)

        action_counts[action_idx % na] += 1
        action = env.action_space[action_idx % na]
        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1
        if obs is None:
            break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None:
                level_step = ts

        if time.time() - t0 > 280:
            break

    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    ratio = len(g.cells) / max(g.step_count, 1)
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells), 'ratio': ratio, 'dom': dom,
    }


def main():
    import arc_agi
    print("Step 449c: CA-Graph rule sweep — Rules 30, 90, 110, 184", flush=True)
    print("Wolfram prediction: Class IV (110) works, III (30) too chaotic, II too ordered.",
          flush=True)
    print(flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: LS20 not found"); return

    # Rules and their Wolfram classifications
    rules = [
        (30,  "Class III chaotic"),
        (90,  "Class II periodic"),
        (110, "Class IV edge-of-chaos"),
        (184, "Class II traffic"),
    ]

    t_total = time.time()
    rule_results = {}

    for rule_num, rule_desc in rules:
        print(f"\n--- Rule {rule_num} ({rule_desc}) ---", flush=True)
        lookup = make_rule_lookup(rule_num)
        results = []
        for seed in [0, 1, 2]:
            r = run_seed(arc, ls20.game_id, seed=seed, rule_lookup=lookup, max_steps=10000)
            status = f"LEVEL 1 at step {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:26s}  unique={r['unique_cells']:5d}"
                  f"  ratio={r['ratio']:.3f}  dom={r['dom']:.0f}%", flush=True)
            results.append(r)

        wins = [r for r in results if r['levels'] > 0]
        avg_ratio = sum(r['ratio'] for r in results) / len(results)
        avg_cells = sum(r['unique_cells'] for r in results) / len(results)
        rule_results[rule_num] = {
            'wins': len(wins), 'avg_ratio': avg_ratio, 'avg_cells': avg_cells,
            'level_steps': sorted([r['level_step'] for r in wins]),
        }
        print(f"  -> {len(wins)}/3  ratio={avg_ratio:.3f}  avg_cells={avg_cells:.0f}",
              flush=True)

    print(f"\n{'='*60}", flush=True)
    print("STEP 449c RULE SWEEP RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Rule':<6} {'Class':<28} {'Wins':<8} {'Ratio':<8} {'Cells':<8} {'Steps'}", flush=True)
    for rule_num, rule_desc in rules:
        rr = rule_results[rule_num]
        print(f"  {rule_num:<4}  {rule_desc:<28}  {rr['wins']}/3    "
              f"{rr['avg_ratio']:.3f}   {rr['avg_cells']:<8.0f} {rr['level_steps']}",
              flush=True)
    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)
    print(flush=True)

    # Verdict
    for rule_num, rule_desc in rules:
        rr = rule_results[rule_num]
        if rr['avg_ratio'] > 0.9:
            verdict = "TOO CHAOTIC (ratio > 0.9)"
        elif rr['avg_cells'] < 50:
            verdict = "TOO ORDERED (< 50 cells)"
        elif rr['wins'] > 0:
            verdict = f"NAVIGATES ({rr['wins']}/3)"
        else:
            verdict = f"healthy dynamics, no navigation at 10K (ratio={rr['avg_ratio']:.3f})"
        print(f"  Rule {rule_num} ({rule_desc}): {verdict}", flush=True)


if __name__ == '__main__':
    main()
