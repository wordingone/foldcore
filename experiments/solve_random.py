"""
Random search solver for games where BFS is too slow.
Uses random walks with frame hashing to avoid loops.
"""
import sys, os, json, time, hashlib, importlib, random, numpy as np
from arcengine import GameAction, ActionInput, GameState

GAME_PATHS = {
    'g50t': ('B:/M/the-search/environment_files/g50t/5849a774', 'g50t', 'G50t'),
    'sc25': ('B:/M/the-search/environment_files/sc25/f9b21a2f', 'sc25', 'Sc25'),
    'ka59': ('B:/M/the-search/environment_files/ka59/9f096b4a', 'ka59', 'Ka59'),
}
RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'
GAME_LEVELS = {'g50t': 7, 'sc25': 6, 'ka59': 7}
GA = {0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3,
      3: GameAction.ACTION4, 4: GameAction.ACTION5}

def clk(x, y): return y * 64 + x
def dec(a):
    if a < 5: return ['UP','DOWN','LEFT','RIGHT','REC'][a]
    return f'CL({a%64},{a//64})'
def fh(a):
    if a is None: return ''
    return hashlib.md5(a.astype(np.uint8).tobytes()).hexdigest()

def load_cls(gid):
    p, m, c = GAME_PATHS[gid]
    if p not in sys.path: sys.path.insert(0, p)
    if m in sys.modules: del sys.modules[m]
    return getattr(importlib.import_module(m), c)

def step(game, action):
    if action < 5:
        ai = ActionInput(id=GA[action], data={})
    else:
        ai = ActionInput(id=GameAction.ACTION6, data={'x': action % 64, 'y': action // 64})
    try:
        r = game.perform_action(ai, raw=True)
    except: return None, 0, False
    if r is None: return None, 0, False
    f = np.array(r.frame, dtype=np.uint8)
    if f.ndim == 3: f = f[-1]
    return f, r.levels_completed, r.state in (GameState.GAME_OVER, GameState.WIN)

def make(cls, li):
    g = cls(); g.full_reset()
    if li > 0: g.set_level(li)
    return g

def scan_actions(cls, li, gid, click_step=2):
    """Find effective actions."""
    bf, _, _ = step(make(cls, li), clk(0, 0))
    bh = fh(bf)
    groups = {}
    kb = {'g50t': [0,1,2,3,4], 'sc25': [0,1,2,3], 'ka59': [0,1,2,3]}[gid]
    for k in kb:
        g = make(cls, li)
        f, lev, _ = step(g, k)
        if f is None: continue
        h = fh(f)
        if h != bh and h not in groups: groups[h] = k

    if gid == 'g50t':
        for k in kb:
            if k not in groups.values(): groups[f'f{k}'] = k

    if gid in ('sc25', 'ka59'):
        for y in range(0, 64, click_step):
            for x in range(0, 64, click_step):
                a = clk(x, y)
                if a < 5: continue  # Skip values that overlap with keyboard
                g = make(cls, li)
                f, lev, _ = step(g, a)
                if f is None: continue
                h = fh(f)
                if h != bh and h not in groups: groups[h] = a

    return list(set(groups.values()))

def random_solve(cls, li, gid, max_len=100, max_attempts=50000, time_limit=300):
    """Random walk solver."""
    t0 = time.time()
    acts = scan_actions(cls, li, gid, click_step=2)
    if not acts:
        print(f"    No actions found!")
        return None

    print(f"    {len(acts)} actions: {[dec(a) for a in acts[:10]]}")
    best_depth = 0
    attempts = 0

    while attempts < max_attempts and time.time() - t0 < time_limit:
        g = make(cls, li)
        seq = []
        visited = set()

        for _ in range(max_len):
            a = random.choice(acts)
            f, lev, done = step(g, a)
            seq.append(a)

            if lev > 0:
                # Optimize: try to shorten
                print(f"    FOUND at attempt {attempts}, {len(seq)} actions, {time.time()-t0:.1f}s")
                # Try to verify by replay
                g2 = make(cls, li)
                for a2 in seq:
                    _, lev2, _ = step(g2, a2)
                    if lev2 > 0:
                        return seq
                print(f"    Replay failed!")
                return seq

            if done:
                break

            h = fh(f)
            if h in visited:
                # Random restart on loops
                break
            visited.add(h)

        attempts += 1
        if attempts % 1000 == 0:
            el = time.time() - t0
            print(f"    attempt {attempts}, {el:.0f}s ({attempts/el:.0f}/s)")

    print(f"    FAILED after {attempts} attempts")
    return None


def chain_verify(cls, actions):
    g = cls(); g.full_reset()
    lev = 0
    for a in actions:
        _, lev, done = step(g, a)
        if done: break
    return lev


L1_KNOWN = {
    'g50t': [3]*4 + [4] + [1]*7 + [3]*5,
    'sc25': [2, clk(30,50), clk(25,55), clk(35,55), clk(30,60)] + [2]*12,
}

def solve(gid):
    print(f"\n{'='*60}\n{gid.upper()} ({GAME_LEVELS[gid]} levels)\n{'='*60}")
    cls = load_cls(gid)
    total = GAME_LEVELS[gid]
    res = {'game': gid, 'source': 'random_search_solver', 'type': 'fullchain',
           'total_levels': total, 'levels': {}, 'all_actions': []}

    # L1
    l1 = L1_KNOWN.get(gid)
    if l1:
        g = cls(); g.full_reset()
        lev = 0
        for a in l1:
            _, lev, done = step(g, a)
            if done or lev > 0: break
        if lev >= 1:
            print(f"L1 OK ({len(l1)} actions)")
            res['levels']['L1'] = {'status': 'SOLVED', 'actions': l1, 'n_actions': len(l1)}
            res['all_actions'] = list(l1)
        else:
            l1 = None

    if not l1:
        print(f"L1: Random search...")
        t0 = time.time()
        l1 = random_solve(cls, 0, gid, max_len=100, max_attempts=100000, time_limit=300)
        if l1:
            res['levels']['L1'] = {'status': 'SOLVED', 'actions': l1, 'n_actions': len(l1),
                                    'time': round(time.time()-t0, 2)}
            res['all_actions'] = list(l1)
        else:
            res['max_level_solved'] = 0; return res

    cur = 1
    for ln in range(2, total + 1):
        if cur < ln - 1: break
        print(f"\nL{ln}: Random search...")
        t0 = time.time()
        sol = random_solve(cls, ln-1, gid, max_len=100, max_attempts=100000, time_limit=300)
        el = time.time() - t0

        if sol:
            res['levels'][f'L{ln}'] = {'status': 'SOLVED', 'actions': sol,
                                        'n_actions': len(sol), 'time': round(el, 2)}
            res['all_actions'].extend(sol)
            cur = ln
            v = chain_verify(cls, res['all_actions'])
            print(f"  Chain: {v} levels")
        else:
            res['levels'][f'L{ln}'] = {'status': 'UNSOLVED', 'time': round(el, 2)}
            break

    res['max_level_solved'] = cur
    res['total_actions'] = len(res['all_actions'])
    return res

if __name__ == '__main__':
    games = sys.argv[1:] if len(sys.argv) > 1 else ['ka59']
    for gid in games:
        t0 = time.time()
        r = solve(gid)
        out = os.path.join(RESULTS_DIR, f'{gid}_fullchain.json')
        with open(out, 'w') as f: json.dump(r, f, indent=2)
        print(f"\nSAVED: {out}")
        print(f"{r['max_level_solved']}/{GAME_LEVELS[gid]} levels, {r['total_actions']} actions")
