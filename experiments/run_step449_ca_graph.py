#!/usr/bin/env python3
"""
Step 449 — CA-Graph Substrate on LS20.
4th architecture family: cellular automaton + graph edges.

Architecture:
- Input: 16x16 avgpool -> 256 values, quantized to {0,1} at median
- CA: 1D 256 cells, Rule 110 (Class IV, edge of chaos), T=10 steps
- Cell ID: hash of final 256-bit CA state
- Graph: (cell, action) -> {next_cell: count}, action = least-visited

Rule 110 binary: 01101110
- 111->0, 110->1, 101->1, 100->0, 011->1, 010->1, 001->1, 000->0

Ban checklist:
1. Cosine/attract? NO — binary CA rules + hash
2. LVQ? NO — cellular automaton
3. Codebook + X? NO — completely different computation model
4. Shared spatial engine? NO — local rules, no match->update->grow

Key diagnostic: unique_cells/steps ratio
- >0.9: too sensitive (chaotic, kill)
- <0.1: too insensitive (ordered, kill)
- 0.1-0.5: sweet spot, meaningful compression

Kill: ratio >0.9 OR unique_cells <50 at 10K.
LS20, 10K steps, 3 seeds.
"""

import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)

# Rule 110 lookup table: index = (left<<2 | center<<1 | right), value = new state
RULE110 = np.array([(110 >> i) & 1 for i in range(8)], dtype=np.uint8)


def ca_step(state):
    """One step of 1D CA with Rule 110. Periodic boundaries."""
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    idx = (left * 4 + state * 2 + right)
    return RULE110[idx]


def obs_to_cell(pooled, t_steps=10):
    """256D pooled obs -> 10 CA steps -> hash cell ID."""
    # Binarize at median
    state = (pooled > np.median(pooled)).astype(np.uint8)
    # Evolve CA for t_steps
    for _ in range(t_steps):
        state = ca_step(state)
    return hash(state.tobytes())


class CAGraph:
    def __init__(self, n_actions=4, ca_steps=10):
        self.n_actions = n_actions
        self.ca_steps = ca_steps
        self.edges = {}
        self.cells = set()
        self.prev_cell = None
        self.prev_action = None
        self.step_count = 0

    def step(self, obs):
        """obs: flat float32 (256,). Returns action index."""
        cell_id = obs_to_cell(obs, self.ca_steps)
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


def run_structural_test():
    """Structural check < 30s."""
    print("Structural test...", flush=True)
    t0 = time.time()

    g = CAGraph(n_actions=4, ca_steps=10)
    rng = np.random.RandomState(0)

    actions = []
    for i in range(1000):
        obs = rng.randn(256).astype(np.float32)
        action = g.step(obs)
        actions.append(action)

    from collections import Counter
    counts = Counter(actions)
    dom = max(counts.values()) / len(actions)
    ratio = len(g.cells) / g.step_count

    assert len(counts) == 4, f"R2: {len(counts)}/4 actions"
    assert dom < 0.5, f"R3: dom={dom:.0%}"
    assert len(g.cells) > 0, "R4: no cells"
    assert len(g.edges) > 0, "R5: no edges"
    assert not hasattr(g, 'nodes'), "R6: codebook DNA"

    elapsed = time.time() - t0
    print(f"  PASS: cells={len(g.cells)}  edges={len(g.edges)}  dom={dom:.0%}"
          f"  ratio={ratio:.3f}  {elapsed:.1f}s", flush=True)

    if ratio > 0.9:
        print(f"  WARNING: ratio={ratio:.2f} — random inputs fully unique (expected)", flush=True)
    return True


def run_seed(arc, game_id, seed, max_steps=10000):
    from arcengine import GameState
    np.random.seed(seed)
    g = CAGraph(n_actions=4, ca_steps=10)
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
            print(f"  LEVEL {lvls} at step {ts}  cells={len(g.cells)}  edges={len(g.edges)}",
                  flush=True)

        if time.time() - t0 > 280:
            break

    elapsed = time.time() - t0
    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    ratio = len(g.cells) / max(g.step_count, 1)
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells), 'edges': len(g.edges),
        'steps': g.step_count, 'ratio': ratio,
        'dom': dom, 'elapsed': elapsed,
    }


def main():
    import arc_agi
    print("Step 449: CA-Graph Substrate on LS20 (Rule 110, T=10)", flush=True)
    print("4th architecture family: cellular automaton + graph edges.", flush=True)
    print("Key diagnostic: unique_cells/steps ratio.", flush=True)
    print("Kill: ratio >0.9 OR unique_cells <50 at 10K.", flush=True)
    print(flush=True)

    if not run_structural_test():
        print("STRUCTURAL FAIL — stopping.", flush=True)
        return
    print(flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: LS20 not found"); return

    t_total = time.time()
    results = []
    for seed in [0, 1, 2]:
        print(f"--- Seed {seed} ---", flush=True)
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=10000)
        status = f"LEVEL 1 at step {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status:26s}  unique_cells={r['unique_cells']:5d}"
              f"  ratio={r['ratio']:.3f}  edges={r['edges']:5d}"
              f"  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)

        if r['ratio'] > 0.9:
            print(f"  KILL: ratio={r['ratio']:.2f} — CA too sensitive (chaotic)", flush=True)
        elif r['unique_cells'] < 50:
            print(f"  KILL: unique_cells={r['unique_cells']} — CA too insensitive (ordered)",
                  flush=True)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("STEP 449 FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    wins = [r for r in results if r['levels'] > 0]
    avg_ratio = sum(r['ratio'] for r in results) / len(results)
    avg_cells = sum(r['unique_cells'] for r in results) / len(results)
    avg_dom = sum(r['dom'] for r in results) / len(results)

    print(f"Reliability: {len(wins)}/3", flush=True)
    if wins:
        print(f"Step-to-level: {sorted([r['level_step'] for r in wins])}", flush=True)
    print(f"Avg unique_cells: {avg_cells:.0f}", flush=True)
    print(f"Avg unique/steps ratio: {avg_ratio:.3f}", flush=True)
    print(f"Avg dom: {avg_dom:.0f}%", flush=True)
    print(f"Total elapsed: {time.time() - t_total:.0f}s", flush=True)
    print(flush=True)

    if avg_ratio > 0.9:
        print(f"CA TOO SENSITIVE (ratio={avg_ratio:.2f}): Rule 110 at T=10 is chaotic.", flush=True)
        print("Try smaller T or different rule (less sensitive).", flush=True)
    elif avg_cells < 50:
        print(f"CA TOO INSENSITIVE: only {avg_cells:.0f} unique cells.", flush=True)
        print("Try larger T or more sensitive rule.", flush=True)
    elif wins:
        print(f"CA NAVIGATES: cellular automaton provides discriminative compression!", flush=True)
    else:
        print(f"Sweet spot dynamics (ratio={avg_ratio:.2f}) but no navigation at 10K.", flush=True)
        print("Request 30K if healthy.", flush=True)

    print(flush=True)
    print("Reference:", flush=True)
    print(f"  Reservoir 448: ratio=0.942 (too sensitive — temporal inconsistency)", flush=True)
    print(f"  Random grid 446b: ratio=1.0, cells=1869 (static, fails navigation)", flush=True)
    print(f"  Cosine graph 442b: ratio~0.07, nodes=1984 (navigates at 30K)", flush=True)


if __name__ == '__main__':
    main()
