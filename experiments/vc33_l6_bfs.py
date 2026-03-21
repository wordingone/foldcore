"""
VC33 Level 7 (LEVELS[6]) BFS solver.
Uses deepcopy checkpoint approach: saves env state at level 7 start,
restores from checkpoint for each BFS node expansion.

Level 7: 48x48 grid, vertical mode (TiD=[0,-2]).
"""
import sys
sys.path.insert(0, 'B:/M/the-search/experiments/environment_files/vc33/9851e02b')
sys.path.insert(0, 'B:/M/the-search/experiments')
import logging
logging.getLogger().setLevel(logging.WARNING)

import copy, time
from collections import deque

GA=(0,27); GB=(0,33); GC=(24,33); GD=(24,27)
S1=(6,30); S2=(30,30); NOP_L5=(60,60)

SOLUTIONS_PREV = {
    0: [(62,34),(62,34),(62,34)],
    1: [(0,24),(0,24),(0,44),(0,44),(0,44),(0,44),(0,44)],
    2: [(12,56),(24,56),(12,56),(24,56),(12,56),(34,56),(24,56),(12,56),(34,56),(24,56),(12,56),(34,56),(24,56),(12,56),
        (46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56)],
    3: [(15,61),(15,61),(12,43),(32,32),(15,61),(15,61),(15,61),
        (39,61),(39,61),(51,61),(39,61),(27,34),(32,32),
        (51,61),(39,61),(51,61),(39,61),(51,61),(39,61),(51,61),(39,61),(51,61),(39,61)],
    4: [(61,17),(61,17),(61,17),(61,17),(61,35),(61,35),(61,35),(61,35),(61,35),(61,52),(61,52),(25,49),(32,32),
        (61,29),(61,29),(61,29),(61,52),(61,52),(40,32),(32,32),
        (61,17),(61,17),(61,17),(61,17),(28,14),(32,32),
        (61,11),(61,11),(61,11),(61,11),(40,32),(32,32),
        (61,11),(61,35),(61,35),(61,35),(61,46),(61,46),(25,49),(32,32),
        (61,29),(61,11),(61,52),(61,52),(61,52),(61,52),(61,52),(61,52),(61,52)],
    5: [GA,GD,GD,GD,S1,NOP_L5,GB,GB,GC,GC,GC,GC,GC,GC,S2,NOP_L5,GD,GD,GD,GD,GD,GD],
}


def advance_to_level(env, action6, target_level):
    obs = env.reset()
    for lvl in range(target_level):
        for cx, cy in SOLUTIONS_PREV[lvl]:
            obs = env.step(action6, data={"x": cx, "y": cy})
    return obs


def get_game(env):
    return getattr(env, 'game', None) or getattr(env, '_game', None)


def get_state_hash(game):
    rdn = tuple(sorted(
        (sp.name, sp.x, sp.y, sp.width, sp.height)
        for sp in game.current_level.get_sprites_by_tag("rDn")
    ))
    hqb = tuple(sorted(
        (sp.name, sp.x, sp.y)
        for sp in game.current_level.get_sprites_by_tag("HQB")
    ))
    return (rdn, hqb)


def inspect_level(game):
    rdn = game.current_level.get_sprites_by_tag("rDn")
    hqb = game.current_level.get_sprites_by_tag("HQB")
    zgd = game.current_level.get_sprites_by_tag("ZGd")
    zhk = game.current_level.get_sprites_by_tag("zHk")
    fzk = game.current_level.get_sprites_by_tag("fZK")
    uxg = game.current_level.get_sprites_by_tag("UXg")
    oro = game.oro

    print(f"oro={oro}, lia()={game.lia()}, grid={game.current_level.grid_size}")
    print(f"\nrDn ({len(rdn)}):")
    for sp in sorted(rdn, key=lambda s: s.name):
        mud = game.mud(sp)
        print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}, "
              f"utq={sp.y+sp.height}, mud={mud}")
        suo = game.suo(sp)
        print(f"    suo={[s.name for s in suo]}")
    print(f"\nHQB ({len(hqb)}):")
    for sp in sorted(hqb, key=lambda s: s.name):
        print(f"  {sp.name}: pos=({sp.x},{sp.y}), tags={list(sp.tags) if hasattr(sp,'tags') else '?'}")
    print(f"\nfZK ({len(fzk)}):")
    for sp in sorted(fzk, key=lambda s: (sp.x, sp.y)):
        print(f"  {sp.name}: pos=({sp.x},{sp.y})")
    print(f"\nUXg ({len(uxg)}):")
    for sp in sorted(uxg, key=lambda s: s.name):
        print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}")
    print(f"\nZGd ({len(zgd)}):")
    for sp in sorted(zgd, key=lambda s: (s.x, s.y)):
        # Find gel mapping
        gel_str = ""
        for zg_sp, (pmj, chd) in game.dzy.items():
            if zg_sp.name == sp.name:
                gel_str = f" gel({pmj.name}->{chd.name})"
        print(f"  {sp.name}: pos=({sp.x},{sp.y}){gel_str}")
    print(f"\nzHk ({len(zhk)}):")
    for sp in sorted(zhk, key=lambda s: (s.x, s.y)):
        print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}")
        print(f"    krt={game.krt(sp)}")
    print(f"\ngug()={game.gug()}")


def bfs_solve(env_checkpoint, action6, actions, max_depth=40):
    from arcengine import GameState

    game_ckpt = get_game(env_checkpoint)
    init_hash = get_state_hash(game_ckpt)

    # Time a deepcopy
    t0 = time.time()
    env_test = copy.deepcopy(env_checkpoint)
    deepcopy_ms = (time.time() - t0) * 1000
    print(f"deepcopy time: {deepcopy_ms:.1f}ms")

    # BFS
    visited = {init_hash: []}
    queue = deque([(init_hash, [])])

    t_start = time.time()
    expansions = 0
    nodes_expanded = 0

    while queue:
        curr_hash, curr_seq = queue.popleft()

        if len(curr_seq) >= max_depth:
            continue

        nodes_expanded += 1

        # Restore to current state from checkpoint
        env_curr = copy.deepcopy(env_checkpoint)
        for pcx, pcy in curr_seq:
            env_curr.step(action6, data={"x": pcx, "y": pcy})

        # Try each action
        for cx, cy in actions:
            env_next = copy.deepcopy(env_curr)
            obs = env_next.step(action6, data={"x": cx, "y": cy})
            expansions += 1

            new_seq = curr_seq + [(cx, cy)]

            if obs.state == GameState.WIN or obs.levels_completed > 6:
                elapsed = time.time() - t_start
                print(f"WIN at depth {len(new_seq)}! "
                      f"({nodes_expanded} nodes, {expansions} exp, {elapsed:.1f}s)")
                return new_seq

            game_next = get_game(env_next)
            h = get_state_hash(game_next)

            if h not in visited:
                visited[h] = new_seq
                queue.append((h, new_seq))

        if nodes_expanded % 50 == 0:
            elapsed = time.time() - t_start
            est_deepcopy = nodes_expanded * deepcopy_ms / 1000
            print(f"  {len(visited)} states, depth={len(curr_seq)}, "
                  f"queue={len(queue)}, {elapsed:.1f}s elapsed")

    elapsed = time.time() - t_start
    print(f"No solution found ({nodes_expanded} nodes, {elapsed:.1f}s)")
    return None


def main():
    import arc_agi

    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    if vc33 is None:
        print("VC33 not found"); return

    env = arc.make(vc33.game_id)
    action6 = env.action_space[0]

    # Advance to level 7
    t0 = time.time()
    obs = advance_to_level(env, action6, 6)
    print(f"Advanced to Level 7 in {time.time()-t0:.1f}s: levels_completed={obs.levels_completed}")
    game = get_game(env)

    print("\n=== Level 7 State ===")
    inspect_level(game)

    if obs.levels_completed != 6:
        print(f"ERROR: expected levels_completed=6, got {obs.levels_completed}")
        return

    # Save checkpoint
    print("\nSaving checkpoint...")
    t0 = time.time()
    env_checkpoint = copy.deepcopy(env)
    print(f"Checkpoint saved in {(time.time()-t0)*1000:.1f}ms")

    # Get actions from checkpoint
    game_ckpt = get_game(env_checkpoint)
    zgd = game_ckpt.current_level.get_sprites_by_tag("ZGd")
    zhk = game_ckpt.current_level.get_sprites_by_tag("zHk")
    actions = [(sp.x, sp.y) for sp in sorted(zgd, key=lambda s: (s.x, s.y))]
    actions += [(sp.x, sp.y) for sp in sorted(zhk, key=lambda s: (s.x, s.y))]
    gs = game_ckpt.current_level.grid_size
    actions.append((gs - 1, gs - 1))  # NOP: far corner
    print(f"Actions ({len(actions)}): {actions}")

    # BFS
    print("\n=== BFS Solve ===")
    solution = bfs_solve(env_checkpoint, action6, actions, max_depth=40)

    if solution:
        print(f"\nSOLUTION ({len(solution)} clicks):")
        print(solution)

        # Verify
        print("\nVerifying...")
        from arcengine import GameState
        obs = advance_to_level(env, action6, 6)
        for i, (cx, cy) in enumerate(solution):
            obs = env.step(action6, data={"x": cx, "y": cy})
            if obs.state == GameState.WIN or obs.levels_completed > 6:
                print(f"VERIFIED: level advanced at click {i} (total={i+1} clicks)")
                break
        else:
            print("VERIFICATION FAILED: no advance after all clicks")
    else:
        print("No solution found — try increasing max_depth or check level 7 analysis")


if __name__ == "__main__":
    main()
