"""
Final fullchain solvers for WA30 (9 levels), BP35 (9 levels), LF52 (10 levels).
Combines abstract model BFS with known solutions and API verification.

Usage: PYTHONUTF8=1 python experiments/solve_wa30_bp35_lf52_final.py [game]
"""
import os, sys, json, time, hashlib
from collections import deque
os.chdir("B:/M/the-search")

import numpy as np
import arc_agi
from arcengine import GameAction, GameState

STEP = 4

def verify_and_save(game_id, all_actions, per_level, method):
    """Verify solution and save fullchain JSON."""
    max_level = 0
    if all_actions:
        print(f"\nVerifying {game_id} ({len(all_actions)} actions)...")
        arcade = arc_agi.Arcade()
        games = arcade.get_environments()
        info = next(g for g in games if game_id in g.game_id.lower())
        env = arcade.make(info.game_id)
        obs = env.reset()
        for i, a in enumerate(all_actions):
            try:
                if a >= 7:
                    ci = a - 7; px = ci % 64; py = ci // 64
                    obs = env.step(GameAction.ACTION6, data={"x": px, "y": py})
                else:
                    ga = list(GameAction)[a + 1]
                    obs = env.step(ga)
            except Exception as e:
                print(f"  Error at action {i}: {e}")
                break
            if obs and obs.levels_completed > max_level:
                max_level = obs.levels_completed
                print(f"  Level {max_level} completed at action {i}")
            if obs and obs.state == GameState.WIN:
                print(f"  WIN! levels={obs.levels_completed}")
                break
            if obs and obs.state == GameState.GAME_OVER:
                print(f"  GAME OVER at action {i}, levels={obs.levels_completed}")
                break
        print(f"  Verified: max_level={max_level}")

    result = {
        "game": game_id,
        "method": method,
        "max_level_solved": max_level,
        "total_actions": len(all_actions),
        "levels": per_level,
        "full_sequence": all_actions,
    }
    out = f"B:/M/the-search/experiments/results/prescriptions/{game_id}_fullchain.json"
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out}")
    return max_level


# =============================================================================
# WA30 SOLVER
# =============================================================================

def extract_wa30_level(level):
    sps = level.get_sprites()
    S = STEP; walls = set(); blocks = []; targets = set(); forbidden = set()
    player = None; ai1 = []; ai2 = []
    for sp in sps:
        tags = list(sp.tags) if hasattr(sp, 'tags') else []
        pos = (sp.x, sp.y)
        if 'wbmdvjhthc' in tags: player = pos
        elif 'geezpjgiyd' in tags: blocks.append(pos)
        elif 'fsjjayjoeg' in tags:
            for dy in range(sp.height):
                for dx in range(sp.width):
                    targets.add((sp.x+dx, sp.y+dy))
        elif 'kdweefinfi' in tags: ai1.append(pos)
        elif 'ysysltqlke' in tags: ai2.append(pos)
        elif 'bnzklblgdk' in tags: forbidden.add(pos)
        elif sp.is_collidable:
            for dy in range(0, sp.height, S):
                for dx in range(0, sp.width, S):
                    walls.add((sp.x+dx, sp.y+dy))
    for i in range(0, 64, S):
        walls.add((-S,i)); walls.add((64,i)); walls.add((i,-S)); walls.add((i,64))
    return {'player':player,'blocks':tuple(sorted(blocks)),'targets':targets,
            'walls':walls,'forbidden':forbidden,'ai1':ai1,'ai2':ai2,
            'budget':level.get_data('StepCounter')}


def wa30_bfs(ld, max_states=5000000):
    S = STEP; walls = ld['walls']; targets = ld['targets']
    forbidden = ld['forbidden']; blocks_init = ld['blocks']
    player = ld['player']; budget = ld['budget']
    DIRS = [(0,-S,0),(0,S,1),(-S,0,2),(S,0,3)]
    GRAB = {0:(0,-S),1:(0,S),2:(-S,0),3:(S,0)}

    def free(p,bs): return p not in walls and p not in forbidden and p not in bs and 0<=p[0]<64 and 0<=p[1]<64
    def win(bt): return all(b in targets for b in bt)

    init = (player, 0, blocks_init, -1)
    vis = {init}; q = deque([(init, [])]); ex = 0; md = 0
    while q:
        (p,f,bl,c), acts = q.popleft()
        d = len(acts)
        if d >= budget: continue
        if d > md: md = d; (md%10==0) and print(f"    d={md} q={len(q)} v={len(vis)}")
        ex += 1
        if ex > max_states: print(f"    Exhausted {ex} d={md}"); return None
        bs = set(bl)
        for aid in range(5):
            if aid < 4:
                dx,dy,nf = DIRS[aid]; np_ = (p[0]+dx,p[1]+dy)
                if c >= 0:
                    bp = bl[c]; off = (bp[0]-p[0],bp[1]-p[1]); nb = (np_[0]+off[0],np_[1]+off[1])
                    ob = bs - {bp}
                    if (np_ not in walls and np_ not in forbidden and np_ not in ob and
                        0<=np_[0]<64 and 0<=np_[1]<64 and np_!=nb and
                        nb not in walls and nb not in forbidden and nb not in ob and
                        0<=nb[0]<64 and 0<=nb[1]<64):
                        nl = list(bl); nl[c] = nb; nbt = tuple(sorted(nl))
                        nc = next(i for i,b in enumerate(nbt) if b==nb)
                        ns = (np_,f,nbt,nc)
                        if ns not in vis: vis.add(ns); q.append((ns, acts+[aid]))
                else:
                    if free(np_,bs):
                        ns = (np_,nf,bl,-1)
                        if ns not in vis: vis.add(ns); q.append((ns, acts+[aid]))
                    else:
                        ns = (p,nf,bl,-1)
                        if ns not in vis: vis.add(ns); q.append((ns, acts+[aid]))
            else:
                if c >= 0:
                    ns = (p,f,bl,-1)
                    if ns not in vis:
                        vis.add(ns); na = acts+[4]
                        if win(bl): print(f"    SOLVED d={len(na)} ex={ex}"); return na
                        q.append((ns, na))
                else:
                    gx,gy = GRAB[f]; tp = (p[0]+gx,p[1]+gy)
                    for bi,bp in enumerate(bl):
                        if bp == tp:
                            ns = (p,f,bl,bi)
                            if ns not in vis: vis.add(ns); q.append((ns, acts+[4]))
                            break
    print(f"    No solution: {ex} states d={md}"); return None


def solve_wa30():
    sys.path.insert(0, 'environment_files/wa30/ee6fef47')
    from wa30 import levels as WL
    all_acts = []; per_level = {}
    for li in range(9):
        print(f"\n{'='*60}\nWA30 L{li+1}/9\n{'='*60}")
        ld = extract_wa30_level(WL[li])
        print(f"  P:{ld['player']} B:{len(ld['blocks'])} AI:{len(ld['ai1'])+len(ld['ai2'])} Budget:{ld['budget']}")
        if ld['ai1'] or ld['ai2']:
            print("  Has AI movers - solving without AI sim (may not chain)")
        t0 = time.time(); sol = wa30_bfs(ld); el = time.time()-t0
        if sol is None:
            per_level[f"L{li+1}"] = {"status":"UNSOLVED","time":round(el,2)}
            print(f"  FAILED ({el:.1f}s)")
            if li == 0: break  # Can't chain without L1
            continue
        per_level[f"L{li+1}"] = {"status":"SOLVED","actions":sol,"count":len(sol),"time":round(el,2)}
        all_acts.extend(sol)
        print(f"  Solved: {len(sol)} in {el:.1f}s")

    verify_and_save("wa30", all_acts, per_level, "abstract_bfs")


# =============================================================================
# BP35 SOLVER
# =============================================================================

def bp35_bfs(grid_lines, legend, level_num, max_states=3000000):
    """BFS for BP35 with abstract gravity model. Grid is bottom=idx0."""
    tiles = {}; player = None; gem = None
    for y, row in enumerate(grid_lines):
        for x, ch in enumerate(row):
            if ch == ' ': continue
            tnames = legend.get(ch, [])
            if not tnames: continue
            tiles[(x,y)] = tnames[0]
            if tnames[0] == "player_right": player = (x,y)
            elif tnames[0] == "fjlzdjxhant": gem = (x,y)
    if not player or not gem: return None

    walls = set(); brk = set(); spk_d = set(); spk_u = set()
    gfl = set(); tog_on = set(); tog_off = set(); exp = set()
    for pos, name in tiles.items():
        if name == "xcjjwqfzjfe": walls.add(pos)
        elif name == "qclfkhjnaac": brk.add(pos)
        elif name == "ubhhgljbnpu": spk_d.add(pos)
        elif name == "hzusueifitk": spk_u.add(pos)
        elif name == "lrpkmzabbfa": gfl.add(pos)
        elif name == "yuuqpmlxorv": tog_on.add(pos)
        elif name == "oonshderxef": tog_off.add(pos)
        elif name == "etlsaqqtjvn": exp.add(pos)

    print(f"  P:{player} G:{gem} W:{len(walls)} B:{len(brk)} S:{len(spk_d)+len(spk_u)} Gf:{len(gfl)} T:{len(tog_on)+len(tog_off)} E:{len(exp)}")

    def solid(p, bn, tn, en):
        if p in walls: return True
        if p in brk and p not in bn: return True
        if p in tog_on and p not in tn: return True
        if p in tog_off and p in tn: return True
        if p in en: return True
        return False

    def fall(pos, gu, bn, tn, en):
        dy = -1 if gu else 1; x, y = pos
        while True:
            ny = y+dy; np_ = (x,ny)
            if np_ == gem: return (np_, True, False)
            if (np_ in spk_d and gu) or (np_ in spk_u and not gu): return ((x,y), False, True)
            if solid(np_, bn, tn, en): return ((x,y), False, False)
            if ny < -5 or ny > 60: return ((x,y), False, True)
            y = ny

    def click_enc(gpos, py_player):
        cam_y = py_player * 6 - 36
        px = gpos[0]*6+3; py = gpos[1]*6 - cam_y + 3
        return 7 + max(0,min(63,py))*64 + max(0,min(63,px))

    # Only consider breaking blocks in a cone from player toward gem
    def relevant_breaks(ppos, gu, bn):
        rel = set()
        for bp in brk:
            if bp in bn: continue
            if abs(bp[0]-ppos[0]) > 6: continue
            if gu and bp[1] >= ppos[1]: continue
            if not gu and bp[1] <= ppos[1]: continue
            rel.add(bp)
        return rel

    p0, fg0, fs0 = fall(player, True, frozenset(), frozenset(), frozenset())
    if fg0: return []
    if fs0: return None

    init = (p0, True, frozenset(), frozenset(), frozenset())
    vis = {init}; q = deque([(init, [])]); ex = 0; md = 0
    while q:
        (pos, gu, bn, tn, en), acts = q.popleft()
        d = len(acts)
        if d >= 80: continue
        if d > md: md = d; (md%5==0) and print(f"    d={md} q={len(q)} v={len(vis)}")
        ex += 1
        if ex > max_states: print(f"    Exhausted {ex} d={md}"); return None

        # Move L/R
        for dx in [-1, 1]:
            np_ = (pos[0]+dx, pos[1])
            if np_ == gem:
                print(f"    SOLVED! d={d+1} ex={ex}")
                return acts + [2 if dx<0 else 3]
            if solid(np_, bn, tn, en): continue
            fp, fg, fs = fall(np_, gu, bn, tn, en)
            if fg:
                print(f"    SOLVED! d={d+1} ex={ex}")
                return acts + [2 if dx<0 else 3]
            if fs: continue
            ns = (fp, gu, bn, tn, en)
            if ns not in vis: vis.add(ns); q.append((ns, acts+[2 if dx<0 else 3]))

        # Break blocks
        for bp in relevant_breaks(pos, gu, bn):
            nb = bn | {bp}
            fp, fg, fs = fall(pos, gu, nb, tn, en)
            if fg:
                print(f"    SOLVED! d={d+1} ex={ex}")
                return acts + [click_enc(bp, pos[1])]
            if fs: continue
            ns = (fp, gu, nb, tn, en)
            if ns not in vis: vis.add(ns); q.append((ns, acts+[click_enc(bp, pos[1])]))

        # Gravity flips
        for gp in gfl:
            ng = not gu
            fp, fg, fs = fall(pos, ng, bn, tn, en)
            if fg:
                print(f"    SOLVED! d={d+1} ex={ex}")
                return acts + [click_enc(gp, pos[1])]
            if fs: continue
            ns = (fp, ng, bn, tn, en)
            if ns not in vis: vis.add(ns); q.append((ns, acts+[click_enc(gp, pos[1])]))

        # Toggle blocks
        for tp in (tog_on | tog_off):
            nt = tn - {tp} if tp in tn else tn | {tp}
            fp, fg, fs = fall(pos, gu, bn, nt, en)
            if fg:
                print(f"    SOLVED! d={d+1} ex={ex}")
                return acts + [click_enc(tp, pos[1])]
            if fs: continue
            ns = (fp, gu, bn, nt, en)
            if ns not in vis: vis.add(ns); q.append((ns, acts+[click_enc(tp, pos[1])]))

        # Expand blocks
        for ep in exp:
            ne = set(en)
            for edx,edy in [(-1,0),(1,0),(0,-1),(0,1)]:
                epos = (ep[0]+edx, ep[1]+edy)
                if not solid(epos, bn, tn, en): ne.add(epos)
            nef = frozenset(ne)
            if nef == en: continue
            fp, fg, fs = fall(pos, gu, bn, tn, nef)
            if fg:
                print(f"    SOLVED! d={d+1} ex={ex}")
                return acts + [click_enc(ep, pos[1])]
            if fs: continue
            ns = (fp, gu, bn, tn, nef)
            if ns not in vis: vis.add(ns); q.append((ns, acts+[click_enc(ep, pos[1])]))

    print(f"    No solution: {ex} states d={md}"); return None


def solve_bp35():
    # Known L1 solution (verified working)
    known = {
        1: [3,3,3,3,2097,2463,2,2,2,2079,2079,3,2085,2,2],
    }

    # Grids for BFS (bottom=idx0, source grids already reversed)
    grids = {
        1: (["ooooooooooo"]*7 + ["oo +     oo","oo       oo","oooooxxxooo","oo       oo","oo       oo",
             "oo  xxx  oo","oo       oo","oo       oo","ooxxxoooooo","ooxxx    oo","ooxxx    oo",
             "ooooo    oo","oooooxxxxoo","oo       oo","oo       oo","ooooooo ooo","oo n     oo",
             "oo       oo","oo       oo","oo       oo","oo       oo","oo       oo","oo       oo",
             "mmmmmmmmmmm","wwwwwwwwwww","wwwwwwwwwww","wwwwwwwwwww","wwwwwwwwwww","wwwwwwwwwww"],
            {"x":["qclfkhjnaac"],"o":["xcjjwqfzjfe"],"n":["player_right"],"v":["ubhhgljbnpu"],
             "m":["aknlbboysnc"],"w":["jcyhkseuorf"],"+":["fjlzdjxhant"]}),
        4: (["ooooooooooo"]*7 + ["oooogoooooo","ooooooooooo",
             "oo       oo","oo       oo","oo  +    oo","oo       oo",
             "oovv vv  oo","oooooooxxoo","ooogogoxxoo","ooooooo  oo",
             "oo    o  oo","oo       oo","oo   xxxxoo",
             "oo       oo","ooxxooooooo","oo       oo","oo       oo","oovvn    oo",
             "oooooo   oo","ooooooo  oo","oooooooo oo","oo       oo","oo       oo",
             "ooooooooooo","ooooogooooo","ooooooooooo"]*1 + ["ooooooooooo"]*3,
            {"x":["qclfkhjnaac"],"g":["lrpkmzabbfa"],"o":["xcjjwqfzjfe"],"n":["player_right"],
             "v":["ubhhgljbnpu"],"u":["hzusueifitk"],"+":["fjlzdjxhant"]}),
    }

    per_level = {}; all_acts = []
    for li in range(1, 10):
        print(f"\n{'='*60}\nBP35 L{li}/9\n{'='*60}")
        if li in known:
            sol = known[li]
            per_level[f"L{li}"] = {"status":"SOLVED","actions":sol,"length":len(sol),"method":"known"}
            all_acts.extend(sol)
            print(f"  Using known solution: {len(sol)} actions")
        elif li in grids:
            grid_lines, legend = grids[li]
            t0 = time.time()
            sol = bp35_bfs(grid_lines, legend, li)
            el = time.time() - t0
            if sol:
                per_level[f"L{li}"] = {"status":"SOLVED","actions":sol,"length":len(sol),"time":round(el,2)}
                all_acts.extend(sol)
                print(f"  Solved: {len(sol)} in {el:.1f}s")
            else:
                per_level[f"L{li}"] = {"status":"UNSOLVED","time":round(el,2)}
                print(f"  FAILED ({el:.1f}s)")
                if li <= 1: break
        else:
            per_level[f"L{li}"] = {"status":"UNSOLVED","note":"no grid data"}
            print(f"  No grid data - skipping")
            if not all_acts: break

    verify_and_save("bp35", all_acts, per_level, "abstract_bfs_gravity")


# =============================================================================
# LF52 SOLVER
# =============================================================================

def lf52_dfs(grid_lines, legend, level_num, max_depth=50):
    """DFS for peg solitaire."""
    board = set(); pieces = set(); red = set(); blue = set(); obs = set(); mov = set()
    for y, row in enumerate(grid_lines):
        for x, ch in enumerate(row):
            if ch == ' ': continue
            tt = legend.get(ch, [])
            if "hupkpseyuim" in tt: board.add((x,y))
            if "hupkpseyuim2" in tt: board.add((x,y)); mov.add((x,y))
            if "fozwvlovdui" in tt: pieces.add((x,y))
            if "fozwvlovdui_red" in tt: red.add((x,y))
            if "fozwvlovdui_blue" in tt: blue.add((x,y))
            if "dgxfozncuiz" in tt: obs.add((x,y))

    all_p = pieces | red | blue
    n = len(all_p)
    print(f"  Board:{len(board)} P:{len(pieces)} R:{len(red)} B:{len(blue)} Obs:{len(obs)} Mov:{len(mov)}")
    if n <= 1: return []
    if mov: print(f"  WARNING: moveable tiles present - DFS handles jumps only")

    DIRS = [(0,-1),(1,0),(0,1),(-1,0)]
    best = [None]
    def dfs(ps, moves, d):
        if best[0]: return
        if len(ps) == 1: best[0] = list(moves); return
        if d >= max_depth: return
        for p in list(ps):
            for dx,dy in DIRS:
                mid = (p[0]+dx,p[1]+dy); land = (p[0]+2*dx,p[1]+2*dy)
                if mid in ps and land in board and land not in ps and land not in obs:
                    moves.append((p,mid,land))
                    dfs(ps - {p,mid} | {land}, moves, d+1)
                    if best[0]: return
                    moves.pop()

    t0 = time.time(); dfs(frozenset(all_p), [], 0); el = time.time()-t0
    if not best[0]: print(f"  No solution ({el:.1f}s)"); return None
    print(f"  {len(best[0])} jumps in {el:.1f}s")

    off = {1:(10,5),2:(6,8),3:(5,5),10:(5,3)}
    ox,oy = off.get(level_num, (5,5))
    acts = []
    for piece, mid, land in best[0]:
        px = piece[0]*6+ox+3; py = piece[1]*6+oy+3
        acts.append(7 + py*64 + px)
        dx = land[0]-piece[0]; dy = land[1]-piece[1]
        adx = (dx//abs(dx)) if dx else 0; ady = (dy//abs(dy)) if dy else 0
        ax = piece[0]*6+ox+3+adx*12; ay = piece[1]*6+oy+3+ady*12
        acts.append(7 + ay*64 + ax)
    return acts


def solve_lf52():
    grids = {
        1: (["",".......",".xx.x..",".....x.","    ...","    .x.","    ...","    ..."],
            {"x":["fozwvlovdui","hupkpseyuim"],".":["hupkpseyuim"]}),
        2: (["....... ",".xx.x.x->","....... |","        |"," <--,---3"," |      "," |    .."," L----x."],
            {"x":["fozwvlovdui","hupkpseyuim"],".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
             "-":["kraubslpehi"],"|":["kraubslpehi-up"],"L":["kraubslpehi-L"],"3":["kraubslpehi-3"],
             "<":["kraubslpehi-<"],">":["kraubslpehi->"]}),
        3: ([".. ..     ...",".x .x-,--x...",".x.x      .xx.","..x.      ..","          .xx.",
             "      <-> ..","      | | x..",".x.x-,3 L-.x","...       .."],
            {"x":["fozwvlovdui","hupkpseyuim"],".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
             "-":["kraubslpehi"],"|":["kraubslpehi-up"],"L":["kraubslpehi-L"],"3":["kraubslpehi-3"],
             "<":["kraubslpehi-<"],">":["kraubslpehi->"]}),
        4: (["","",".......",".xp.p.x-,---x...x.",".......     .p..p.","            ......",
             "            .p..p.","        <---,..p..","        |","    .x. | ","   .p p.p..",
             "   .----,-.","   .p p....","    .x."],
            {"x":["fozwvlovdui","hupkpseyuim"],".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
             "p":["dgxfozncuiz","hupkpseyuim"],"-":["kraubslpehi"],"|":["kraubslpehi-up"],
             "L":["kraubslpehi-L"],"3":["kraubslpehi-3"],"<":["kraubslpehi-<"],">":["kraubslpehi->"]}),
    }

    per_level = {}; all_acts = []
    for li in range(1, 11):
        print(f"\n{'='*60}\nLF52 L{li}/10\n{'='*60}")
        if li not in grids:
            per_level[f"L{li}"] = {"status":"UNSOLVED","note":"grid not parsed"}
            print(f"  Grid not parsed - skipping")
            if not all_acts: break
            continue
        grid_lines, legend = grids[li]
        t0 = time.time(); sol = lf52_dfs(grid_lines, legend, li); el = time.time()-t0
        if sol is None:
            per_level[f"L{li}"] = {"status":"UNSOLVED","time":round(el,2)}
            print(f"  FAILED ({el:.1f}s)")
            if li <= 1: break
            continue
        per_level[f"L{li}"] = {"status":"SOLVED","actions":sol,"length":len(sol),"time":round(el,2)}
        all_acts.extend(sol)
        print(f"  Solved: {len(sol)} actions in {el:.1f}s")

    verify_and_save("lf52", all_acts, per_level, "abstract_dfs_peg_solitaire")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "all"
    if game in ["wa30","all"]: solve_wa30()
    if game in ["bp35","all"]: solve_bp35()
    if game in ["lf52","all"]: solve_lf52()
