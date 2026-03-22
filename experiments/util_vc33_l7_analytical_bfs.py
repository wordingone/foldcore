"""
VC33 Level 7 (LEVELS[6]) analytical BFS.

Level state: track heights + (train_track, train_y) for each of 3 trains.
No game engine calls — pure Python dict BFS. Fast.

Win conditions (from pixel color mapping):
  ChX (color=11) -> EtZ at (32,18): ChX.y=18, keB in suo(track)
  PPS (color=14) -> cCW at (14,10): PPS.y=10, khL in suo(track)
  VAJ (color=15) -> xIX at (14,42): VAJ.y=42, khL in suo(track)

Track layout (Level 7, 48x48 grid):
  HMp: x=0,  y=0,  w=14, h_init=20, mud=22
  RmM: x=16, y=0,  w=16, h_init=8,  mud=48  (bridge: both keB and khL)
  HfU: x=34, y=0,  w=14, h_init=6,  mud=18
  wmR: x=0,  y=24, w=14, h_init=8,  mud=48
  AEF: x=34, y=24, w=14, h_init=10, mud=48

Gel actions (ZGd coords from game engine):
  (16,0)  HMp->RmM   (12,0)  RmM->HMp
  (16,24) wmR->RmM   (12,24) RmM->wmR
  (30,24) AEF->RmM   (34,24) RmM->AEF
  (30,0)  HfU->RmM   (34,0)  RmM->HfU

Switches (zHk coords from game engine):
  (14,30): left=wmR, right=RmM, fires when wmR.utq=30 AND RmM.utq=30
  (32,8):  left=RmM, right=HfU, fires when RmM.utq=8  AND HfU.utq=8
  (32,30): left=RmM, right=AEF, fires when RmM.utq=30 AND AEF.utq=30
"""
import sys
sys.path.insert(0, 'B:/M/the-search/experiments/environment_files/vc33/9851e02b')
sys.path.insert(0, 'B:/M/the-search/experiments')
import logging
logging.getLogger().setLevel(logging.WARNING)

import time
from collections import deque

# Track constants
TRACK_BASE_Y = {'HMp': 0, 'RmM': 0, 'HfU': 0, 'wmR': 24, 'AEF': 24}
MUD = {'HMp': 22, 'RmM': 48, 'HfU': 18, 'wmR': 48, 'AEF': 48}
TRACKS = ['HMp', 'RmM', 'HfU', 'wmR', 'AEF']
TRAINS = ['ChX', 'PPS', 'VAJ']

# UXg adjacency
SUO_keB = {'RmM', 'HfU', 'AEF'}
SUO_khL = {'HMp', 'RmM', 'wmR'}

# Win targets: train -> (target_y, required_uxg)
WIN_TARGET = {
    'ChX': (18, 'keB'),   # EtZ(32,18)
    'PPS': (10, 'khL'),   # cCW(14,10)
    'VAJ': (42, 'khL'),   # xIX(14,42)
}

# Gel actions: (pmj, chd, click_x, click_y)
GEL_ACTIONS = [
    ('HMp', 'RmM', 16, 0),
    ('RmM', 'HMp', 12, 0),
    ('wmR', 'RmM', 16, 24),
    ('RmM', 'wmR', 12, 24),
    ('AEF', 'RmM', 30, 24),
    ('RmM', 'AEF', 34, 24),
    ('HfU', 'RmM', 30, 0),
    ('RmM', 'HfU', 34, 0),
]

# Switch actions: (left_track, right_track, sw_y, click_x, click_y)
SWITCH_ACTIONS = [
    ('wmR', 'RmM', 30, 14, 30),
    ('RmM', 'HfU', 8,  32, 8),
    ('RmM', 'AEF', 30, 32, 30),
]


def utq(h_tuple, track_idx):
    return TRACK_BASE_Y[TRACKS[track_idx]] + h_tuple[track_idx]


def make_state(h, trains):
    """
    State = (HMp_h, RmM_h, HfU_h, wmR_h, AEF_h,
             ChX_track_idx, ChX_y, PPS_track_idx, PPS_y, VAJ_track_idx, VAJ_y)
    All values are integers. Train track_idx in 0-4 for TRACKS list.
    """
    return (
        h['HMp'], h['RmM'], h['HfU'], h['wmR'], h['AEF'],
        TRACKS.index(trains['ChX']['track']), trains['ChX']['y'],
        TRACKS.index(trains['PPS']['track']), trains['PPS']['y'],
        TRACKS.index(trains['VAJ']['track']), trains['VAJ']['y'],
    )


def check_win(s):
    # s = (HMp_h, RmM_h, HfU_h, wmR_h, AEF_h, ChX_t, ChX_y, PPS_t, PPS_y, VAJ_t, VAJ_y)
    chx_t, chx_y = TRACKS[s[5]], s[6]
    pps_t, pps_y = TRACKS[s[7]], s[8]
    vaj_t, vaj_y = TRACKS[s[9]], s[10]

    if chx_y != 18 or chx_t not in SUO_keB:
        return False
    if pps_y != 10 or pps_t not in SUO_khL:
        return False
    if vaj_y != 42 or vaj_t not in SUO_khL:
        return False
    return True


def apply_gel(s, pmj_name, chd_name):
    pmj_idx = TRACKS.index(pmj_name)
    chd_idx = TRACKS.index(chd_name)

    # Condition: source height > 0 AND dest utq < mud
    h = list(s[:5])
    if h[pmj_idx] == 0:
        return None
    chd_utq = TRACK_BASE_Y[chd_name] + h[chd_idx]
    if chd_utq >= MUD[chd_name]:
        return None

    # Apply height changes
    h[pmj_idx] -= 2
    h[chd_idx] += 2

    # Apply train movements
    trains_y = [s[6], s[8], s[10]]
    trains_t = [s[5], s[7], s[9]]

    for i in range(3):
        if trains_t[i] == pmj_idx:
            trains_y[i] -= 2  # source: move up
        elif trains_t[i] == chd_idx:
            trains_y[i] += 2  # dest: move down

    return (h[0], h[1], h[2], h[3], h[4],
            trains_t[0], trains_y[0],
            trains_t[1], trains_y[1],
            trains_t[2], trains_y[2])


def apply_switch(s, left_name, right_name, sw_y):
    left_idx = TRACKS.index(left_name)
    right_idx = TRACKS.index(right_name)

    h = s[:5]
    left_utq = TRACK_BASE_Y[left_name] + h[left_idx]
    right_utq = TRACK_BASE_Y[right_name] + h[right_idx]

    if left_utq != sw_y or right_utq != sw_y:
        return None  # krt = False

    trains_t = [s[5], s[7], s[9]]
    trains_y = [s[6], s[8], s[10]]

    # Swap tracks for trains on left <-> right
    for i in range(3):
        if trains_t[i] == left_idx:
            trains_t[i] = right_idx
        elif trains_t[i] == right_idx:
            trains_t[i] = left_idx
    # y values unchanged

    return (h[0], h[1], h[2], h[3], h[4],
            trains_t[0], trains_y[0],
            trains_t[1], trains_y[1],
            trains_t[2], trains_y[2])


def bfs_solve(max_depth=60):
    init_s = (
        20, 8, 6, 8, 10,        # HMp, RmM, HfU, wmR, AEF heights
        TRACKS.index('wmR'), 32, # ChX
        TRACKS.index('AEF'), 34, # PPS
        TRACKS.index('HfU'), 6,  # VAJ
    )

    if check_win(init_s):
        print("Already at win state!")
        return []

    # BFS: (state_hash, click_sequence)
    visited = {init_s: []}
    queue = deque([(init_s, [])])

    t_start = time.time()
    expansions = 0

    while queue:
        curr_s, curr_seq = queue.popleft()

        if len(curr_seq) >= max_depth:
            continue

        # Try all gel actions
        for pmj, chd, cx, cy in GEL_ACTIONS:
            new_s = apply_gel(curr_s, pmj, chd)
            if new_s is None:
                continue

            expansions += 1
            new_seq = curr_seq + [(cx, cy)]

            if check_win(new_s):
                elapsed = time.time() - t_start
                print(f"WIN at depth {len(new_seq)}! ({expansions} expansions, {elapsed:.2f}s)")
                return new_seq

            if new_s not in visited:
                visited[new_s] = new_seq
                queue.append((new_s, new_seq))

        # Try all switch actions
        for lt, rt, sw_y, cx, cy in SWITCH_ACTIONS:
            new_s = apply_switch(curr_s, lt, rt, sw_y)
            if new_s is None:
                continue

            expansions += 1
            new_seq = curr_seq + [(cx, cy)]

            if check_win(new_s):
                elapsed = time.time() - t_start
                print(f"WIN at depth {len(new_seq)}! ({expansions} expansions, {elapsed:.2f}s)")
                return new_seq

            if new_s not in visited:
                visited[new_s] = new_seq
                queue.append((new_s, new_seq))

        if len(visited) % 5000 == 0:
            elapsed = time.time() - t_start
            print(f"  {len(visited)} states, depth={len(curr_seq)}, "
                  f"{expansions} exp, {elapsed:.2f}s")

    elapsed = time.time() - t_start
    print(f"No solution found ({len(visited)} states, {expansions} exp, {elapsed:.2f}s)")
    return None


def verify_solution(solution):
    """Verify solution against actual game engine."""
    import arc_agi
    from arcengine import GameState

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

    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    env = arc.make(vc33.game_id)
    action6 = env.action_space[0]
    obs = env.reset()
    for lvl in range(6):
        for cx, cy in SOLUTIONS_PREV[lvl]:
            obs = env.step(action6, data={'x': cx, 'y': cy})

    game = getattr(env, 'game', None) or getattr(env, '_game', None)
    print(f"\nVerifying {len(solution)} clicks against game engine...")

    # Try solution with and without NOP after switches
    SWITCH_COORDS = {(14,30), (32,8), (32,30)}
    NOP = (47, 47)

    for nop_mode in ['no_nop', 'nop_after_switch']:
        obs = env.reset()
        for lvl in range(6):
            for cx, cy in SOLUTIONS_PREV[lvl]:
                obs = env.step(action6, data={'x': cx, 'y': cy})

        seq_to_run = []
        for cx, cy in solution:
            seq_to_run.append((cx, cy))
            if nop_mode == 'nop_after_switch' and (cx, cy) in SWITCH_COORDS:
                seq_to_run.append(NOP)

        for i, (cx, cy) in enumerate(seq_to_run):
            obs = env.step(action6, data={'x': cx, 'y': cy})

            if obs.state == GameState.WIN or obs.levels_completed > 6:
                print(f"  [{nop_mode}] VERIFIED: level advanced at step {i}! "
                      f"(total={i+1} steps, sol={len(seq_to_run)})")
                return seq_to_run  # return the full sequence with NOPs

        print(f"  [{nop_mode}] FAIL after {len(seq_to_run)} steps")
        rdn = {sp.name: sp.height for sp in game.current_level.get_sprites_by_tag('rDn')}
        hqb = {sp.name: (sp.x, sp.y) for sp in game.current_level.get_sprites_by_tag('HQB')}
        print(f"    final heights: {rdn}")
        print(f"    final trains: {hqb}")

    return None


if __name__ == "__main__":
    print("=== VC33 Level 7 Analytical BFS ===\n")
    print("Win targets: ChX.y=18(keB), PPS.y=10(khL), VAJ.y=42(khL)")
    print("Initial: ChX(wmR,y=32), PPS(AEF,y=34), VAJ(HfU,y=6)\n")

    solution = bfs_solve(max_depth=80)

    if solution:
        print(f"\nAnalytical solution ({len(solution)} clicks):")
        print(solution)

        verified = verify_solution(solution)
        if verified:
            print(f"\nFINAL VERIFIED SOLUTION ({len(verified)} clicks):")
            print(verified)
        else:
            print("\nGame engine verification failed!")
            print("Analytical solution (may need NOP adjustment):")
            print(solution)
    else:
        print("No analytical solution found!")
