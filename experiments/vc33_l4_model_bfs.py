"""
VC33 Level 4 (LEVELS[4] = "Level 5") model-based BFS.

State: (bUo_x, bUo_w, UWO_x, UWO_w, SnP_x, SnP_w, Xfy_x, Xfy_w, cni_x, Ubu_x, cni_on, Ubu_on)
Where _on = 0=bUo, 1=UWO, 2=SnP, 3=Xfy

Initial:
- bUo: x=16, w=48
- UWO: x=58, w=6
- SnP: x=46, w=18
- Xfy: x=43, w=21
- cni: x=10, on bUo (x+w=16 => lho=16=bUo.x)
- Ubu: x=37, on Xfy (x+w=37+6=43=Xfy.x)

TiD=[3,0] means rsi=3, qir=0.
gel(pmj, chd) when lia()=True: chd.x > mud(chd).

mud values (computed in diagnostic):
- SnP: mud=25
- UWO: mud=25  (actually: utq of tHm.x=25 and XTW.x=16... max=25)
- Xfy: mud=10
- bUo: mud=16

gel condition: chd.x > mud(chd).

For gel(A, B): B.x > mud(B)
Also: jqo(pmj)=pmj.w > 0 (pmj has cells to give, i.e., width > 0)

Wait, jqo(sp) = sp.width when oro[0]=3 (True), so jqo=width.
But actually: pmj transfers rsi=3 pixels from its right side. So pmj.w must be > 0.
Actually wait: pmj.pixels = pmj.pixels[:-abs(rsi)] => removes 3 columns => pmj.w -= 3.
So need pmj.w >= 3 (i.e., pmj.w > 0 with step 3).

Actually the condition is jqo(pmj) > 0, which is pmj.width > 0.
With rsi=3: removes 3 columns at once. So need pmj.w >= 3.

Switches (zHk):
krt(dBq) with TiD=[3,0]:
- lho(dBq) = dBq.x + dBq.w
- Fch = rDn where utq(rDn)=lho(dBq) i.e. rDn.x = dBq.x+dBq.w
- IlL = [rDn in Fch where urh(rDn)+brj(rDn)==urh(dBq)] = rDn where rDn.y+rDn.h == dBq.y
- BVv = [rDn in Fch where urh(rDn)==urh(dBq)+brj(dBq)] = rDn where rDn.y == dBq.y+dBq.h

Switches and their activation conditions:
1. Switch (25,49): dBq.w=12, dBq.h=3 (but h here is fixed 3 for the switch sprite itself... actually dBq is zHk)
   - lho = 25+12=37. Fch: rDn.x=37. IlL: rDn.y+rDn.h=49. BVv: rDn.y=52.
   - SnP.y+SnP.h = 35+14=49 → SnP in IlL if SnP.x=37
   - Xfy.y=52 → Xfy in BVv if Xfy.x=37
   - CONDITION: SnP.x=37 AND Xfy.x=37 → cross SnP↔Xfy
   - teu(): jwi=SnP, Xhj=Xfy. Ubu (on Xfy) moves to SnP center.

2. Switch (28,14): lho=28+12=40. IlL: rDn.y+h=14. BVv: rDn.y=17.
   - bUo.y+bUo.h=0+14=14 → bUo in IlL if bUo.x=40
   - UWO.y=17 → UWO in BVv if UWO.x=40
   - CONDITION: bUo.x=40 AND UWO.x=40 → cross bUo↔UWO
   - teu(): jwi=bUo, Xhj=UWO. cni (on bUo) moves to UWO center.

3. Switch (40,32): lho=40+12=52. IlL: rDn.y+h=32. BVv: rDn.y=35.
   - UWO.y+UWO.h=17+15=32 → UWO in IlL if UWO.x=52
   - SnP.y=35 → SnP in BVv if SnP.x=52
   - CONDITION: UWO.x=52 AND SnP.x=52 → cross UWO↔SnP
   - teu(): jwi=UWO, Xhj=SnP. HQB on SnP moves to UWO, HQB on UWO moves to SnP.

WIN condition (gug() returns True):
- cni must match eoH (cni.x=10) AND be on SnP or Xfy (so BqR in suo):
  BqR in suo(SnP): SnP.y+SnP.h=49=BqR.y ✓ always (SnP.y=35, h=14 fixed in y)
  BqR in suo(Xfy): Xfy.y=52=BqR.y+BqR.h ✓ always
  So: cni.x=10 AND (cni on SnP or cni on Xfy)

- Ubu must match UIT (Ubu.x=16) AND be on bUo or UWO (so XTW in suo):
  XTW in suo(bUo): bUo.y+h=14=XTW.y ✓ always
  XTW in suo(UWO): UWO.y=17... wait, need XTW.y+XTW.h=17=UWO.y? XTW.y=14, h=3: 14+3=17=UWO.y ✓!
  So: Ubu.x=16 AND (Ubu on bUo or Ubu on UWO)

Also need to check if AkL(HQB) matches fZK pixels. Assuming cni↔eoH and Ubu↔UIT (different colors).

PLAN:
BFS over model state. Actions: gel operations + switches when conditions met.
"""
from collections import deque

# Fixed y positions (don't change with TiD=[3,0])
# rDn y positions are fixed; only x changes
RDN_Y = {'bUo': 0, 'UWO': 17, 'SnP': 35, 'Xfy': 52}
RDN_H = {'bUo': 14, 'UWO': 15, 'SnP': 14, 'Xfy': 12}
MUD = {'bUo': 16, 'UWO': 25, 'SnP': 25, 'Xfy': 10}

# HQB dimensions
HQB_W = 6
HQB_H = 6

# Initial state
INIT = {
    'bUo_x': 16, 'bUo_w': 48,
    'UWO_x': 58, 'UWO_w': 6,
    'SnP_x': 46, 'SnP_w': 18,
    'Xfy_x': 43, 'Xfy_w': 21,
    'cni_x': 10, 'cni_on': 'bUo',
    'Ubu_x': 37, 'Ubu_on': 'Xfy',
}

TOTAL_W = 48 + 6 + 18 + 21  # = 93

def state_to_tuple(s):
    return (s['bUo_x'], s['bUo_w'],
            s['UWO_x'], s['UWO_w'],
            s['SnP_x'], s['SnP_w'],
            s['Xfy_x'], s['Xfy_w'],
            s['cni_x'], s['cni_on'],
            s['Ubu_x'], s['Ubu_on'])

def check_win(s):
    """gug() returns True when:
    - cni.x=10 AND cni on SnP or Xfy
    - Ubu.x=16 AND Ubu on bUo or UWO
    """
    cni_win = (s['cni_x'] == 10 and s['cni_on'] in ('SnP', 'Xfy'))
    ubu_win = (s['Ubu_x'] == 16 and s['Ubu_on'] in ('bUo', 'UWO'))
    return cni_win and ubu_win

def apply_gel(s, pmj, chd, rsi=3):
    """Apply gel(pmj, chd). Returns new state or None if not applicable."""
    pmj_x = s[f'{pmj}_x']
    pmj_w = s[f'{pmj}_w']
    chd_x = s[f'{chd}_x']
    chd_w = s[f'{chd}_w']
    mud_chd = MUD[chd]

    # Condition: pmj.w >= rsi AND chd.x > mud(chd)
    if pmj_w < rsi:
        return None
    if chd_x <= mud_chd:
        return None

    ns = dict(s)

    # pmj moves right by rsi, shrinks
    ns[f'{pmj}_x'] = pmj_x + rsi
    ns[f'{pmj}_w'] = pmj_w - rsi

    # chd moves left by rsi, grows
    ns[f'{chd}_x'] = chd_x - rsi
    ns[f'{chd}_w'] = chd_w + rsi

    # Move HQB on pmj right by rsi
    if s['cni_on'] == pmj:
        ns['cni_x'] = s['cni_x'] + rsi
    if s['Ubu_on'] == pmj:
        ns['Ubu_x'] = s['Ubu_x'] + rsi

    # Move HQB on chd left by rsi
    if s['cni_on'] == chd:
        ns['cni_x'] = s['cni_x'] - rsi
    if s['Ubu_on'] == chd:
        ns['Ubu_x'] = s['Ubu_x'] - rsi

    return ns

def apply_switch(s, switch_name):
    """Apply switch (train crossing). Returns new state or None if not applicable."""
    if switch_name == '(25,49)':
        # Need SnP.x=37 AND Xfy.x=37; crosses HQBs between SnP↔Xfy
        if s['SnP_x'] != 37 or s['Xfy_x'] != 37:
            return None
        ns = dict(s)
        # Move HQB from Xfy to SnP center: jwi=SnP, Xhj=Xfy
        # HQB on Xfy -> moves to SnP center
        # HQB on SnP -> moves to Xfy center
        # Center: new_x = other_rdn.x + other_rdn.w//2 - HQB_W//2
        new_cni_x = ns['cni_x']
        new_cni_on = ns['cni_on']
        new_ubu_x = ns['Ubu_x']
        new_ubu_on = ns['Ubu_on']

        if s['Ubu_on'] == 'Xfy':
            new_ubu_x = s['SnP_x'] + s['SnP_w'] // 2 - HQB_W // 2
            new_ubu_on = 'SnP'
        elif s['Ubu_on'] == 'SnP':
            new_ubu_x = s['Xfy_x'] + s['Xfy_w'] // 2 - HQB_W // 2
            new_ubu_on = 'Xfy'

        if s['cni_on'] == 'Xfy':
            new_cni_x = s['SnP_x'] + s['SnP_w'] // 2 - HQB_W // 2
            new_cni_on = 'SnP'
        elif s['cni_on'] == 'SnP':
            new_cni_x = s['Xfy_x'] + s['Xfy_w'] // 2 - HQB_W // 2
            new_cni_on = 'Xfy'

        ns['cni_x'] = new_cni_x
        ns['cni_on'] = new_cni_on
        ns['Ubu_x'] = new_ubu_x
        ns['Ubu_on'] = new_ubu_on
        return ns

    elif switch_name == '(28,14)':
        # Need bUo.x=40 AND UWO.x=40; crosses HQBs between bUo↔UWO
        if s['bUo_x'] != 40 or s['UWO_x'] != 40:
            return None
        ns = dict(s)
        new_cni_x = ns['cni_x']
        new_cni_on = ns['cni_on']
        new_ubu_x = ns['Ubu_x']
        new_ubu_on = ns['Ubu_on']

        if s['cni_on'] == 'bUo':
            new_cni_x = s['UWO_x'] + s['UWO_w'] // 2 - HQB_W // 2
            new_cni_on = 'UWO'
        elif s['cni_on'] == 'UWO':
            new_cni_x = s['bUo_x'] + s['bUo_w'] // 2 - HQB_W // 2
            new_cni_on = 'bUo'

        if s['Ubu_on'] == 'bUo':
            new_ubu_x = s['UWO_x'] + s['UWO_w'] // 2 - HQB_W // 2
            new_ubu_on = 'UWO'
        elif s['Ubu_on'] == 'UWO':
            new_ubu_x = s['bUo_x'] + s['bUo_w'] // 2 - HQB_W // 2
            new_ubu_on = 'bUo'

        ns['cni_x'] = new_cni_x
        ns['cni_on'] = new_cni_on
        ns['Ubu_x'] = new_ubu_x
        ns['Ubu_on'] = new_ubu_on
        return ns

    elif switch_name == '(40,32)':
        # Need UWO.x=52 AND SnP.x=52; crosses HQBs between UWO↔SnP
        if s['UWO_x'] != 52 or s['SnP_x'] != 52:
            return None
        ns = dict(s)
        new_cni_x = ns['cni_x']
        new_cni_on = ns['cni_on']
        new_ubu_x = ns['Ubu_x']
        new_ubu_on = ns['Ubu_on']

        if s['cni_on'] == 'UWO':
            new_cni_x = s['SnP_x'] + s['SnP_w'] // 2 - HQB_W // 2
            new_cni_on = 'SnP'
        elif s['cni_on'] == 'SnP':
            new_cni_x = s['UWO_x'] + s['UWO_w'] // 2 - HQB_W // 2
            new_cni_on = 'UWO'

        if s['Ubu_on'] == 'UWO':
            new_ubu_x = s['SnP_x'] + s['SnP_w'] // 2 - HQB_W // 2
            new_ubu_on = 'SnP'
        elif s['Ubu_on'] == 'SnP':
            new_ubu_x = s['UWO_x'] + s['UWO_w'] // 2 - HQB_W // 2
            new_ubu_on = 'UWO'

        ns['cni_x'] = new_cni_x
        ns['cni_on'] = new_cni_on
        ns['Ubu_x'] = new_ubu_x
        ns['Ubu_on'] = new_ubu_on
        return ns

    return None

# All gel operations: (action_name, pmj, chd, display_coord)
GEL_OPS = [
    ('A0_gel(UWO,bUo)',  'UWO',  'bUo',  (61,11)),
    ('A1_gel(bUo,UWO)',  'bUo',  'UWO',  (61,17)),
    ('A2_gel(SnP,UWO)',  'SnP',  'UWO',  (61,29)),
    ('A3_gel(UWO,SnP)',  'UWO',  'SnP',  (61,35)),
    ('A4_gel(Xfy,SnP)',  'Xfy',  'SnP',  (61,46)),
    ('A5_gel(SnP,Xfy)',  'SnP',  'Xfy',  (61,52)),
]

SWITCH_OPS = [
    ('SW1_(25,49)',  '(25,49)',  (25,49)),
    ('SW2_(28,14)',  '(28,14)',  (28,14)),
    ('SW3_(40,32)',  '(40,32)',  (40,32)),
]

ALL_OPS = [(name, 'gel', pmj, chd, coord)
           for name, pmj, chd, coord in GEL_OPS] + \
          [(name, 'switch', sname, None, coord)
           for name, sname, coord in SWITCH_OPS]


def bfs_solve(max_depth=60):
    init_tuple = state_to_tuple(INIT)
    visited = {init_tuple: None}  # state -> (prev_state, action)
    queue = deque([(INIT, 0)])

    found = None

    while queue:
        s, depth = queue.popleft()
        s_tuple = state_to_tuple(s)

        if check_win(s):
            found = s_tuple
            break

        if depth >= max_depth:
            continue

        for op_info in ALL_OPS:
            name = op_info[0]
            op_type = op_info[1]

            if op_type == 'gel':
                _, _, pmj, chd, coord = op_info
                ns = apply_gel(s, pmj, chd)
            else:  # switch
                _, _, sname, _, coord = op_info
                ns = apply_switch(s, sname)

            if ns is None:
                continue

            ns_tuple = state_to_tuple(ns)
            if ns_tuple not in visited:
                visited[ns_tuple] = (s_tuple, name, coord)
                queue.append((ns, depth + 1))

    if found is None:
        print(f"No solution found within {max_depth} moves.")
        print(f"States explored: {len(visited)}")
        return None

    # Reconstruct path
    path = []
    cur = found
    while visited[cur] is not None:
        prev_s, action_name, coord = visited[cur]
        path.append((action_name, coord))
        cur = prev_s

    path.reverse()
    print(f"Solution found! {len(path)} moves:")
    for i, (name, coord) in enumerate(path):
        print(f"  {i}: {name} -> click {coord}")

    return path


def bfs_astar(max_depth=70):
    """A*-like search with heuristic to guide toward win condition."""
    import heapq

    def heuristic(s):
        # Lower = better
        h = 0
        # cni needs to be on SnP/Xfy with x=10
        if s['cni_on'] in ('SnP', 'Xfy'):
            h += abs(s['cni_x'] - 10) // 3
        else:
            # Not on target rDn yet; need to cross
            h += 10 + abs(s['cni_x'] - 10) // 3
        # Ubu needs to be on bUo/UWO with x=16
        if s['Ubu_on'] in ('bUo', 'UWO'):
            h += abs(s['Ubu_x'] - 16) // 3
        else:
            h += 10 + abs(s['Ubu_x'] - 16) // 3
        return h

    init_tuple = state_to_tuple(INIT)
    h0 = heuristic(INIT)
    # Priority queue: (f, g, state_tuple, state_dict, path)
    pq = [(h0, 0, init_tuple, INIT, [])]
    visited = {}  # state_tuple -> min g-cost

    while pq:
        f, g, s_tuple, s, path = heapq.heappop(pq)

        if s_tuple in visited and visited[s_tuple] <= g:
            continue
        visited[s_tuple] = g

        if check_win(s):
            print(f"A* Solution found! {len(path)} moves:")
            for i, (name, coord) in enumerate(path):
                print(f"  {i}: {name} -> click {coord}")
            return path

        if g >= max_depth:
            continue

        for op_info in ALL_OPS:
            name = op_info[0]
            op_type = op_info[1]

            if op_type == 'gel':
                _, _, pmj, chd, coord = op_info
                ns = apply_gel(s, pmj, chd)
            else:
                _, _, sname, _, coord = op_info
                ns = apply_switch(s, sname)

            if ns is None:
                continue

            ns_tuple = state_to_tuple(ns)
            new_g = g + 1
            if ns_tuple not in visited or visited[ns_tuple] > new_g:
                new_h = heuristic(ns)
                new_f = new_g + new_h
                heapq.heappush(pq, (new_f, new_g, ns_tuple, ns, path + [(name, coord)]))

    print(f"No solution found. States explored: {len(visited)}")
    return None


if __name__ == "__main__":
    print("=== VC33 Level 4 Model BFS ===\n")
    print(f"Initial state:")
    for k, v in INIT.items():
        print(f"  {k}={v}")
    print(f"Win condition: cni.x=10 on SnP/Xfy AND Ubu.x=16 on bUo/UWO")
    print()

    print("Running BFS (max_depth=60)...")
    result = bfs_solve(max_depth=60)

    if result is None:
        print("\nRunning A* (max_depth=70)...")
        result = bfs_astar(max_depth=70)

    if result:
        coords = [coord for name, coord in result]
        print(f"\nClick sequence (display coords): {coords}")
