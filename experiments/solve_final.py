"""
Final fullchain solver for G50T, SC25, KA59.

Action encoding: 0-4 = keyboard (UP/DOWN/LEFT/RIGHT/RECORD),
                 clicks = y*64+x (no offset, matching existing fullchain files).

Strategy: Known L1 + BFS for remaining levels using set_level().
"""
import json, time, sys, os, hashlib, importlib, numpy as np
from collections import deque
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

def clk(x, y): return y * 64 + x  # click encoding

def dec(a):
    if a < 5: return ['UP','DOWN','LEFT','RIGHT','REC'][a]
    return f'CL({a%64},{a//64})'

def fh(arr):
    if arr is None: return ''
    return hashlib.md5(arr.astype(np.uint8).tobytes()).hexdigest()

def load_cls(gid):
    p, m, c = GAME_PATHS[gid]
    if p not in sys.path: sys.path.insert(0, p)
    if m in sys.modules: del sys.modules[m]
    return getattr(importlib.import_module(m), c)

def step(game, action):
    """Execute action. Returns (frame, levels_completed, done)."""
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

def replay(cls, li, actions):
    g = make(cls, li)
    f, lev, done = None, 0, False
    for a in actions:
        f, lev, done = step(g, a)
        if done or lev > 0: break
    return f, lev, done

def scan_actions(cls, li, gid, click_step=2):
    """Find effective actions at this level."""
    bf, _, _ = replay(cls, li, [clk(0, 0)])
    bh = fh(bf)
    groups = {}

    kb = {'g50t': [0,1,2,3,4], 'sc25': [0,1,2,3], 'ka59': [0,1,2,3]}[gid]
    for k in kb:
        f, lev, _ = replay(cls, li, [k])
        if f is None: continue
        if lev > 0: return [k], True
        h = fh(f)
        if h != bh and h not in groups: groups[h] = k

    if gid == 'g50t':
        for k in kb:
            if k not in groups.values(): groups[f'f{k}'] = k

    if gid in ('sc25', 'ka59'):
        for y in range(0, 64, click_step):
            for x in range(0, 64, click_step):
                a = clk(x, y)
                f, lev, _ = replay(cls, li, [a])
                if f is None: continue
                if lev > 0: return [a], True
                h = fh(f)
                if h != bh and h not in groups: groups[h] = a

    return list(set(groups.values())), False

def bfs(cls, li, gid, max_d=80, max_s=3000000, tl=300):
    t0 = time.time()
    bf, _, _ = replay(cls, li, [clk(0, 0)])
    ih = fh(bf)

    print(f"    Scanning...")
    acts, inst = scan_actions(cls, li, gid, click_step=2)
    if inst: return acts

    if gid == 'ka59' and sum(1 for a in acts if a >= 5) < 2:
        acts2, inst2 = scan_actions(cls, li, gid, click_step=1)
        if inst2: return acts2
        if len(acts2) > len(acts): acts = acts2

    kb_n = sum(1 for a in acts if a < 5)
    print(f"    {len(acts)} actions ({kb_n} kb, {len(acts)-kb_n} clicks)")
    if not acts: return None

    q = deque([()]); vis = {ih}; exp = 0; dep = 0

    while q:
        if time.time() - t0 > tl:
            print(f"    TIMEOUT (e={exp}, d={dep})"); return None

        seq = q.popleft()
        if len(seq) > dep:
            dep = len(seq)
            el = time.time() - t0
            print(f"      d={dep} v={len(vis)} q={len(q)} e={exp} t={el:.0f}s ({exp/max(el,.1):.0f}/s)")
        if len(seq) >= max_d: continue

        for a in acts:
            if len(seq) >= 1:
                l = seq[-1]
                if l < 5 and a < 5 and (l,a) in {(0,1),(1,0),(2,3),(3,2)}: continue
                if a == 4 and l == 4: continue
                if a >= 5 and l == a: continue

            ns = list(seq) + [a]
            f, lev, done = replay(cls, li, ns)
            exp += 1

            if lev > 0:
                print(f"    SOLVED! {len(ns)} actions, {exp} explored, {time.time()-t0:.1f}s")
                return ns
            if done or f is None: continue
            h = fh(f)
            if h not in vis:
                vis.add(h); q.append(tuple(ns))
            if exp >= max_s:
                print(f"    STATE LIMIT (d={dep})"); return None

    print(f"    EXHAUSTED (e={exp})"); return None

def chain_verify(cls, actions):
    g = cls(); g.full_reset()
    lev = 0
    for a in actions:
        _, lev, done = step(g, a)
        if done: break
    return lev

# Known L1 solutions
L1 = {
    'g50t': [3]*4 + [4] + [1]*7 + [3]*5,
    'sc25': [2, clk(30,50), clk(25,55), clk(35,55), clk(30,60)] + [2]*12,
    'ka59': None,
}

def solve(gid):
    print(f"\n{'='*60}\nSOLVING {gid.upper()} ({GAME_LEVELS[gid]} levels)\n{'='*60}")

    cls = load_cls(gid)
    total = GAME_LEVELS[gid]
    res = {'game': gid, 'source': 'analytical_solver', 'type': 'fullchain',
           'total_levels': total, 'levels': {}, 'all_actions': []}

    l1 = L1.get(gid)
    if l1:
        print(f"\nL1: Verifying...")
        g = cls(); g.full_reset()
        lev = 0
        for a in l1:
            _, lev, done = step(g, a)
            if done or lev > 0: break
        if lev >= 1:
            print(f"  L1 OK ({len(l1)} actions)")
            res['levels']['L1'] = {'status': 'SOLVED', 'actions': l1, 'n_actions': len(l1)}
            res['all_actions'] = list(l1)
        else:
            print(f"  L1 FAILED, BFS...")
            l1 = None

    if not l1:
        print(f"\nL1: BFS...")
        t0 = time.time()
        l1 = bfs(cls, 0, gid, max_d=60, max_s=2000000, tl=180)
        if l1:
            res['levels']['L1'] = {'status': 'SOLVED', 'actions': l1, 'n_actions': len(l1),
                                    'time': round(time.time()-t0, 2)}
            res['all_actions'] = list(l1)
            print(f"  L1: {[dec(a) for a in l1[:20]]}")
        else:
            res['max_level_solved'] = 0; return res

    cur = 1
    for ln in range(2, total + 1):
        if cur < ln - 1: break
        print(f"\nL{ln}:")
        t0 = time.time()
        sol = bfs(cls, ln-1, gid, max_d=80, max_s=3000000, tl=300)
        el = time.time() - t0

        if sol:
            res['levels'][f'L{ln}'] = {'status': 'SOLVED', 'actions': sol,
                                        'n_actions': len(sol), 'time': round(el, 2)}
            res['all_actions'].extend(sol)
            cur = ln
            v = chain_verify(cls, res['all_actions'])
            print(f"  Chain: {v} levels. Actions: {[dec(a) for a in sol[:15]]}")
        else:
            res['levels'][f'L{ln}'] = {'status': 'UNSOLVED', 'time': round(el, 2)}
            print(f"  UNSOLVED ({el:.1f}s)"); break

    res['max_level_solved'] = cur
    res['total_actions'] = len(res['all_actions'])
    return res

if __name__ == '__main__':
    games = sys.argv[1:] if len(sys.argv) > 1 else ['sc25', 'g50t', 'ka59']
    for gid in games:
        t0 = time.time()
        r = solve(gid)
        out = os.path.join(RESULTS_DIR, f'{gid}_fullchain.json')
        with open(out, 'w') as f: json.dump(r, f, indent=2)
        print(f"\n  SAVED: {out}")
        print(f"  {r['max_level_solved']}/{GAME_LEVELS[gid]} levels, {r['total_actions']} actions, {time.time()-t0:.0f}s")
    print(f"\n{'='*60}\nDONE\n{'='*60}")
