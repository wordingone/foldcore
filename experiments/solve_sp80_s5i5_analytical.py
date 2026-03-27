"""
Analytical solver for SP80 (L3-L6) and S5I5 (L3-L8).
Uses set_level() to inspect game state, then computes solutions from source code analysis.

For SP80: liquid flow game with bars, receptacles, spill mechanic.
For S5I5: arm rotation/extension game with panels, arms, pieces, targets.

Strategy: Use game internals to discover clickable regions and positions,
then analytically determine the correct sequence. For complex levels,
use targeted DFS (not BFS) with game-state-aware pruning.
"""
import sys
import os
import json
import time
import hashlib
import importlib
import numpy as np
from collections import deque
from arcengine import GameAction, ActionInput, GameState

# --- Shared utilities ---

def encode_click(x, y):
    return 7 + y * 64 + x

def decode_action(a):
    if a < 7: return f"KB{a}"
    ci = a - 7
    return f"CL({ci%64},{ci//64})"

def frame_hash(arr):
    if arr is None: return ''
    return hashlib.md5(arr.astype(np.uint8).tobytes()).hexdigest()

ARCAGI3_TO_GA = {
    0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3,
    3: GameAction.ACTION4, 4: GameAction.ACTION5, 5: GameAction.ACTION6,
    6: GameAction.ACTION7,
}

def game_step(game, action):
    """Execute arcagi3 action. Return (frame_2d, levels_completed, state)."""
    if action < 7:
        ga = ARCAGI3_TO_GA[action]
        if ga == GameAction.ACTION6:
            ai = ActionInput(id=GameAction.ACTION6, data={'x': 0, 'y': 0})
        else:
            ai = ActionInput(id=ga, data={})
    else:
        ci = action - 7
        ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})
    try:
        r = game.perform_action(ai, raw=True)
    except Exception as e:
        return None, 0, None
    if r is None:
        return None, 0, None
    f = np.array(r.frame, dtype=np.uint8)
    if f.ndim == 3: f = f[-1]
    return f, r.levels_completed, r.state

def make_at_level(game_cls, level_idx):
    g = game_cls()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)
    return g

def replay_at_level(game_cls, level_idx, actions):
    g = make_at_level(game_cls, level_idx)
    frame = None
    levels = 0
    state = None
    for a in actions:
        frame, levels, state = game_step(g, a)
        if state in (GameState.GAME_OVER, GameState.WIN) or levels > 0:
            break
    return frame, levels, state, g


# --- Action discovery ---

def find_unique_actions_at_level(game_cls, level_idx, kb_actions=[], click_step=1):
    """Find actions that change the frame at a given level."""
    g = make_at_level(game_cls, level_idx)
    # Get base frame by doing a no-op click outside
    base_f, _, _, _ = replay_at_level(game_cls, level_idx, [encode_click(0, 0)])
    base_h = frame_hash(base_f)

    groups = {}

    # KB actions
    for kb in kb_actions:
        f, lev, st, _ = replay_at_level(game_cls, level_idx, [kb])
        if f is None: continue
        if lev > 0:
            return [kb], True  # instant solve
        h = frame_hash(f)
        if h != base_h and h not in groups:
            groups[h] = kb

    # Click grid
    for y in range(0, 64, click_step):
        for x in range(0, 64, click_step):
            a = encode_click(x, y)
            f, lev, st, _ = replay_at_level(game_cls, level_idx, [a])
            if f is None: continue
            if lev > 0:
                return [a], True
            h = frame_hash(f)
            if h != base_h and h not in groups:
                groups[h] = a

    return list(groups.values()), False


# --- DFS solver with depth limit ---

def dfs_solve_level(game_cls, level_idx, actions, max_depth=30, time_limit=120):
    """Iterative deepening DFS with frame hashing."""
    t0 = time.time()

    print(f"    DFS with {len(actions)} actions: {[decode_action(a) for a in actions]}")

    # Get initial hash
    base_f, _, _, _ = replay_at_level(game_cls, level_idx, [encode_click(0, 0)])
    init_h = frame_hash(base_f)

    for target_depth in range(1, max_depth + 1):
        if time.time() - t0 > time_limit:
            print(f"    TIMEOUT at depth {target_depth}")
            return None

        visited = {init_h}
        stack = [([], init_h)]
        explored = 0

        while stack:
            if time.time() - t0 > time_limit:
                print(f"    TIMEOUT at depth {target_depth} ({explored} explored)")
                return None

            seq, cur_h = stack.pop()

            if len(seq) >= target_depth:
                continue

            for a in actions:
                explored += 1
                new_seq = seq + [a]
                f, lev, st, _ = replay_at_level(game_cls, level_idx, new_seq)

                if lev > 0:
                    elapsed = time.time() - t0
                    print(f"    SOLVED at depth {len(new_seq)}! ({explored} explored, {elapsed:.1f}s)")
                    return new_seq

                if f is None or st == GameState.GAME_OVER:
                    continue

                h = frame_hash(f)
                if h not in visited:
                    visited.add(h)
                    stack.append((new_seq, h))

        elapsed = time.time() - t0
        print(f"      d={target_depth}: {explored} explored, {len(visited)} visited, {elapsed:.1f}s")

    return None


# --- BFS solver (for smaller action spaces) ---

def bfs_solve_level(game_cls, level_idx, actions, max_depth=50, max_states=500000, time_limit=300):
    """BFS solve level using set_level + unique actions."""
    t0 = time.time()

    base_f, _, _, _ = replay_at_level(game_cls, level_idx, [encode_click(0, 0)])
    init_h = frame_hash(base_f)

    print(f"    BFS with {len(actions)} actions: {[decode_action(a) for a in actions[:10]]}{'...' if len(actions)>10 else ''}")

    queue = deque([()])
    visited = {init_h}
    explored = 0
    depth = 0

    while queue:
        if time.time() - t0 > time_limit:
            print(f"    TIMEOUT ({time_limit}s, e={explored}, d={depth})")
            return None

        seq = queue.popleft()

        if len(seq) > depth:
            depth = len(seq)
            el = time.time() - t0
            rate = explored / max(el, 0.1)
            print(f"      d={depth} v={len(visited)} q={len(queue)} e={explored} t={el:.0f}s ({rate:.0f}/s)")

        if len(seq) >= max_depth:
            continue

        for action in actions:
            new_actions = list(seq) + [action]
            frame, levels, state, _ = replay_at_level(game_cls, level_idx, new_actions)
            explored += 1

            if levels > 0:
                print(f"    SOLVED! {len(new_actions)} actions, {explored} states, {time.time()-t0:.1f}s")
                return new_actions

            if state == GameState.GAME_OVER or frame is None:
                continue

            h = frame_hash(frame)
            if h not in visited:
                visited.add(h)
                queue.append(tuple(new_actions))

            if explored >= max_states:
                print(f"    LIMIT ({max_states}, d={depth})")
                return None

    print(f"    EXHAUSTED ({explored})")
    return None


# --- SP80 specific analysis ---

def analyze_sp80_level(game_cls, level_idx):
    """Analyze SP80 level to find bar positions, sources, receptacles."""
    g = make_at_level(game_cls, level_idx)
    game = g
    level = game.current_level

    print(f"\n  SP80 Level {level_idx + 1} analysis:")
    print(f"    Grid size: {level.grid_size}")
    print(f"    Rotation: {game.sywpxxgfq * 90} degrees")
    print(f"    Steps: {game.tunzhnhfa}")
    print(f"    Spill count: {game.zzocrmvox}")

    # Bars (clickable, tag ksmzdcblcz)
    bars = game.rxjmwfcjyw()
    print(f"    Bars ({len(bars)}):")
    for b in bars:
        print(f"      {b.name} at ({b.x},{b.y}) size {b.width}x{b.height} tags={b.tags}")

    # Sources (tag nkrtlkykwe)
    sources = game.mdtzyuabwe()
    print(f"    Sources ({len(sources)}):")
    for s in sources:
        print(f"      at ({s.x},{s.y})")

    # Receptacles (tag xsrqllccpx)
    recs = game.mldlhgjtqi()
    print(f"    Receptacles ({len(recs)}):")
    for r in recs:
        print(f"      at ({r.x},{r.y}) size {r.width}x{r.height} rot={r.rotation}")

    # Drain (tag uzunfxpwmd)
    drains = game.cycphutjqn()
    print(f"    Drains ({len(drains)}):")
    for d in drains:
        print(f"      at ({d.x},{d.y}) size {d.width}x{d.height} rot={d.rotation}")

    # Source markers (tag syaipsfndp)
    markers = list(level.get_sprites_by_tag("syaipsfndp"))
    print(f"    Source markers ({len(markers)}):")
    for m in markers:
        print(f"      at ({m.x},{m.y})")

    # Selected bar
    if game.dpkgglmdup:
        print(f"    Selected: {game.dpkgglmdup.name} at ({game.dpkgglmdup.x},{game.dpkgglmdup.y})")

    return g


def solve_sp80():
    """Solve SP80 levels 3-6 analytically."""
    sp80_path = 'B:/M/the-search/environment_files/sp80/0ee2d095'
    if sp80_path not in sys.path:
        sys.path.insert(0, sp80_path)
    from sp80 import Sp80

    # Load existing solution
    with open('B:/M/the-search/experiments/results/prescriptions/sp80_fullchain.json') as f:
        existing = json.load(f)

    full_sequence = list(existing['sequence'])

    # Verify L1-L2
    g = Sp80()
    g.full_reset()
    levels = 0
    for a in full_sequence:
        _, levels, st = game_step(g, a)
        if st in (GameState.GAME_OVER, GameState.WIN):
            break
    print(f"  L1-L2 verification: levels_completed={levels}")

    if levels < 2:
        print("  L1-L2 FAILED!")
        return None

    results = {
        'game': 'sp80',
        'levels': {
            'L1': {'status': 'SOLVED', 'method': 'existing'},
            'L2': {'status': 'SOLVED', 'method': 'existing'},
        },
        'full_sequence': list(full_sequence),
    }

    current_max = 2

    for lnum in range(3, 7):
        print(f"\n{'='*60}")
        print(f"SP80 Level {lnum}")
        print(f"{'='*60}")

        t0 = time.time()

        # Analyze level
        analyze_sp80_level(Sp80, lnum - 1)

        # Find unique actions
        # SP80 has kb actions 0-5 (ACTION1-6)
        # ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right, ACTION5=spill, ACTION6=click
        kb_actions = [0, 1, 2, 3, 4]  # don't include 5 (ACTION6/click) as kb since we scan clicks
        unique, instant = find_unique_actions_at_level(Sp80, lnum - 1, kb_actions, click_step=2)

        if instant:
            sol = unique
        else:
            print(f"  {len(unique)} unique actions found")
            # For SP80, use BFS with reasonable limits
            sol = bfs_solve_level(Sp80, lnum - 1, unique, max_depth=40, max_states=300000, time_limit=240)

        elapsed = time.time() - t0

        if sol is not None:
            # Verify in chain
            test_seq = results['full_sequence'] + sol
            g = Sp80()
            g.full_reset()
            chain_levels = 0
            for a in test_seq:
                _, chain_levels, st = game_step(g, a)
                if st in (GameState.GAME_OVER, GameState.WIN):
                    break

            print(f"  Chain verification: levels={chain_levels}")

            if chain_levels >= lnum:
                results['levels'][f'L{lnum}'] = {
                    'status': 'SOLVED',
                    'actions': sol,
                    'length': len(sol),
                    'time': round(elapsed, 2),
                    'decoded': [decode_action(a) for a in sol],
                }
                results['full_sequence'].extend(sol)
                current_max = lnum
                print(f"  SOLVED L{lnum}: {len(sol)} actions, {elapsed:.1f}s")
            else:
                print(f"  Chain verification FAILED (got {chain_levels}, need {lnum})")
                results['levels'][f'L{lnum}'] = {'status': 'CHAIN_FAIL', 'time': round(elapsed, 2)}
                break
        else:
            print(f"  UNSOLVED ({elapsed:.1f}s)")
            results['levels'][f'L{lnum}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
            break

    results['max_level'] = current_max
    results['total_actions'] = len(results['full_sequence'])
    return results


# --- S5I5 specific analysis ---

def analyze_s5i5_level(game_cls, level_idx):
    """Analyze S5I5 level to find panels, arms, pieces, targets."""
    g = make_at_level(game_cls, level_idx)
    game = g
    level = game.current_level

    print(f"\n  S5I5 Level {level_idx + 1} analysis:")
    print(f"    Grid size: {level.grid_size}")
    print(f"    Steps: {game.okmxyzxpez.jjosiqawcv}")

    # Panels (tag gdgcpukdrl)
    panels = list(level.get_sprites_by_tag("gdgcpukdrl"))
    print(f"    Panels ({len(panels)}):")
    for p in panels:
        color = p.pixels[p.height//2, p.width//2] if p.pixels.size > 0 else '?'
        print(f"      {p.name} at ({p.x},{p.y}) size {p.width}x{p.height} color={color}")

    # Arms (tag agujdcrunq)
    arms = list(level.get_sprites_by_tag("agujdcrunq"))
    print(f"    Arms ({len(arms)}):")
    for a in arms:
        arm_color = a.pixels[1, 1] if a.pixels.shape[0] > 1 and a.pixels.shape[1] > 1 else '?'
        orientation = game.fhkoulsvoi(a)
        arm_len = max(a.width, a.height) // 3  # bjntvocxdv = 3
        print(f"      {a.name} at ({a.x},{a.y}) size {a.width}x{a.height} color={arm_color} orient={orientation} len={arm_len}")

    # Pieces/moveable (tag zylvdxoiuq)
    pieces = list(level.get_sprites_by_tag("zylvdxoiuq"))
    print(f"    Pieces ({len(pieces)}):")
    for p in pieces:
        print(f"      {p.name} at ({p.x},{p.y}) size {p.width}x{p.height}")

    # Targets (tag cpdhnkdobh)
    targets = list(level.get_sprites_by_tag("cpdhnkdobh"))
    print(f"    Targets ({len(targets)}):")
    for t in targets:
        print(f"      {t.name} at ({t.x},{t.y}) size {t.width}x{t.height}")

    # Rotation buttons (tag myzmclysbl)
    rotbtns = list(level.get_sprites_by_tag("myzmclysbl"))
    print(f"    Rotation buttons ({len(rotbtns)}):")
    for r in rotbtns:
        btn_color = r.pixels[r.height//2, r.height//2] if r.pixels.size > 0 else '?'
        print(f"      {r.name} at ({r.x},{r.y}) size {r.width}x{r.height} color={btn_color}")

    # Children (parent-child arm relationships)
    children_data = level.get_data("Children")
    if children_data:
        print(f"    Children relationships: {children_data}")

    # Panel->arm mapping
    print(f"    Panel->Arm mapping:")
    for panel in panels:
        if panel in game.dfyrdkjdcj:
            arm_names = [a.name for a in game.dfyrdkjdcj[panel]]
            print(f"      {panel.name} -> {arm_names}")

    # Arm->children mapping
    print(f"    Arm->Children mapping:")
    for arm in arms:
        if arm in game.enplxxgoja:
            child_names = [c.name for c in game.enplxxgoja[arm]]
            print(f"      {arm.name} -> {child_names}")

    return g


def solve_s5i5():
    """Solve S5I5 levels 3-8 analytically."""
    s5i5_path = 'B:/M/the-search/environment_files/s5i5/a48e4b1d'
    if s5i5_path not in sys.path:
        sys.path.insert(0, s5i5_path)
    from s5i5 import S5i5

    # Load existing solution
    with open('B:/M/the-search/experiments/results/prescriptions/s5i5_fullchain.json') as f:
        existing = json.load(f)

    l12_seq = existing['full_sequence']

    # Verify L1-L2
    g = S5i5()
    g.full_reset()
    levels = 0
    for a in l12_seq:
        _, levels, st = game_step(g, a)
        if st in (GameState.GAME_OVER, GameState.WIN):
            break
    print(f"  L1-L2 verification: levels_completed={levels}")

    if levels < 2:
        print("  L1-L2 FAILED!")
        return None

    results = {
        'game': 's5i5',
        'version': 'a48e4b1d',
        'levels': {
            'L1': {'status': 'SOLVED', 'method': 'existing'},
            'L2': {'status': 'SOLVED', 'method': 'existing'},
        },
        'full_sequence': list(l12_seq),
    }

    current_max = 2

    for lnum in range(3, 9):
        print(f"\n{'='*60}")
        print(f"S5I5 Level {lnum}")
        print(f"{'='*60}")

        t0 = time.time()

        # Analyze level
        analyze_s5i5_level(S5i5, lnum - 1)

        # Find unique actions (S5I5 is click-only, action 6)
        unique, instant = find_unique_actions_at_level(S5i5, lnum - 1, [], click_step=2)

        if instant:
            sol = unique
        else:
            # Refine with step=1 if few actions found
            if len(unique) <= 4:
                unique2, instant2 = find_unique_actions_at_level(S5i5, lnum - 1, [], click_step=1)
                if instant2:
                    sol = unique2
                    elapsed = time.time() - t0
                    results['levels'][f'L{lnum}'] = {
                        'status': 'SOLVED', 'actions': sol, 'length': len(sol),
                        'time': round(elapsed, 2),
                    }
                    results['full_sequence'].extend(sol)
                    current_max = lnum
                    print(f"  SOLVED L{lnum}: {len(sol)} actions")
                    continue
                unique = unique2

            print(f"  {len(unique)} unique actions found")

            # Use BFS for small action spaces, DFS for larger
            if len(unique) <= 6:
                sol = bfs_solve_level(S5i5, lnum - 1, unique, max_depth=40, max_states=500000, time_limit=240)
            else:
                sol = dfs_solve_level(S5i5, lnum - 1, unique, max_depth=25, time_limit=240)

        elapsed = time.time() - t0

        if sol is not None:
            # Verify in chain
            test_seq = results['full_sequence'] + sol
            g = S5i5()
            g.full_reset()
            chain_levels = 0
            for a in test_seq:
                _, chain_levels, st = game_step(g, a)
                if st in (GameState.GAME_OVER, GameState.WIN):
                    break

            print(f"  Chain verification: levels={chain_levels}")

            if chain_levels >= lnum:
                results['levels'][f'L{lnum}'] = {
                    'status': 'SOLVED',
                    'actions': sol,
                    'length': len(sol),
                    'time': round(elapsed, 2),
                    'decoded': [decode_action(a) for a in sol],
                }
                results['full_sequence'].extend(sol)
                current_max = lnum
                print(f"  SOLVED L{lnum}: {len(sol)} actions, {elapsed:.1f}s")
            else:
                print(f"  Chain verification FAILED (got {chain_levels}, need {lnum})")
                results['levels'][f'L{lnum}'] = {'status': 'CHAIN_FAIL', 'time': round(elapsed, 2)}
                break
        else:
            print(f"  UNSOLVED ({elapsed:.1f}s)")
            results['levels'][f'L{lnum}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
            break

    results['max_level_solved'] = current_max
    results['total_actions'] = len(results['full_sequence'])
    return results


if __name__ == '__main__':
    os.chdir('B:/M/the-search')

    games = sys.argv[1:] if len(sys.argv) > 1 else ['sp80', 's5i5']

    for game_id in games:
        if game_id == 'sp80':
            print(f"\n{'='*60}")
            print(f"SOLVING SP80 (L3-L6)")
            print(f"{'='*60}")
            result = solve_sp80()
            if result:
                out_path = 'B:/M/the-search/experiments/results/prescriptions/sp80_fullchain.json'
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nSaved: {out_path}")
                print(f"Max level: {result.get('max_level', 0)}/6, Total actions: {result.get('total_actions', 0)}")

        elif game_id == 's5i5':
            print(f"\n{'='*60}")
            print(f"SOLVING S5I5 (L3-L8)")
            print(f"{'='*60}")
            result = solve_s5i5()
            if result:
                out_path = 'B:/M/the-search/experiments/results/prescriptions/s5i5_fullchain.json'
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nSaved: {out_path}")
                print(f"Max level: {result.get('max_level_solved', 0)}/8, Total actions: {result.get('total_actions', 0)}")
