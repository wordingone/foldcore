#!/usr/bin/env python3
"""
Step 375 -- VC33 zone mapping + 3-class process().

1. Map 256 click positions to 3 zones.
2. Run process() with 3 classes, encoding = [timer, zone_response] (2-dim).
3. 2K steps. Kill: level completes.
Script: scripts/run_step375_vc33_zones.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class CF:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev
    def _fa(self,x,l):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self._ut()
    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        s=self.V[idx]@self.V.T; t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())
    def pn(self,x,nc):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0
        si=self.V@x; ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)
        if tm.sum()==0 or si[tm].max()<self.thresh:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self._ut()
        else:
            ts=si.clone(); ts[~tm]=-float('inf'); w=ts.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return p


def main():
    t0 = time.time()
    print("Step 375 -- VC33 zone mapping + 3-class process()", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(flush=True)

    import arc_agi
    from arcengine import GameState

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    vc33 = next(g for g in games if 'vc33' in g.game_id.lower())

    # ========================================================================
    # Phase 1: Map zones
    # ========================================================================
    print("[Phase 1] Map click zones (256 positions from step 10):", flush=True)

    zone_frames = {}  # frame_hash → zone_id
    position_zones = {}  # (x,y) → zone_id
    zone_representatives = {}  # zone_id → (x, y)

    for x in range(0, 64, 4):
        for y in range(0, 64, 4):
            env = arc.make(vc33.game_id)
            obs = env.reset()
            # Advance to step 10
            for _ in range(10):
                obs = env.step(env.action_space[0], data={"x": 32, "y": 32})
                if obs is None or obs.state != GameState.NOT_FINISHED: break
            if obs is None or obs.state != GameState.NOT_FINISHED: continue
            # Click at (x, y)
            obs = env.step(env.action_space[0], data={"x": x, "y": y})
            if obs is None: continue
            fh = hash(np.array(obs.frame[0]).tobytes())
            if fh not in zone_frames:
                zone_id = len(zone_frames)
                zone_frames[fh] = zone_id
            position_zones[(x, y)] = zone_frames[fh]
            if zone_frames[fh] not in zone_representatives:
                zone_representatives[zone_frames[fh]] = (x, y)

    n_zones = len(zone_frames)
    print(f"  Found {n_zones} zones", flush=True)
    for z_id, (x, y) in zone_representatives.items():
        count = sum(1 for v in position_zones.values() if v == z_id)
        print(f"  Zone {z_id}: representative=({x},{y})  positions={count}/256", flush=True)

    # Print zone map as 16x16 grid
    print("\n  Zone map (16x16 grid, x across, y down):", flush=True)
    print("      " + " ".join(f"{x:2d}" for x in range(0, 64, 4)), flush=True)
    for y in range(0, 64, 4):
        row = []
        for x in range(0, 64, 4):
            z = position_zones.get((x, y), -1)
            row.append(f"{z:2d}")
        print(f"  y={y:2d} " + " ".join(row), flush=True)

    if n_zones < 2:
        print("\nKILL: only 1 zone found. Can't differentiate clicks.", flush=True)
        return

    # ========================================================================
    # Phase 2: Run process() with n_zones classes
    # ========================================================================
    print(f"\n[Phase 2] Run process() with {n_zones} classes (2K steps):", flush=True)
    print(f"Encoding: [timer_frac, zone_response_onehot] ({1+n_zones} dims)", flush=True)

    D_ENC = 1 + n_zones  # timer + one-hot zone
    fold = CF(d=D_ENC, k=3)

    env = arc.make(vc33.game_id)
    obs = env.reset()
    action_space = env.action_space

    total_steps = 0; go = 0; levels = 0; seeded = False
    zone_counts = {z: 0 for z in range(n_zones)}
    last_zone = 0

    max_steps = 2000

    while total_steps < max_steps and go < 200:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); last_zone = 0
            if obs is None: break; continue
        if obs.state == GameState.WIN:
            print(f"    WIN at step {total_steps}! levels={obs.levels_completed}", flush=True)
            break

        # Encode: timer fraction + zone response one-hot
        frame = np.array(obs.frame[0])
        # Timer = fraction of row 0 that's value 7 (not yet ticked)
        timer_frac = (frame[0] == 7).sum() / 64.0

        enc = np.zeros(D_ENC, dtype=np.float32)
        enc[0] = timer_frac
        enc[1 + last_zone] = 1.0  # one-hot of last zone response
        enc_t = torch.from_numpy(enc)

        if not seeded and fold.V.shape[0] < n_zones:
            i = fold.V.shape[0]; fold._fa(enc_t, i)
            zone = i
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
            cx, cy = zone_representatives[zone]
            obs = env.step(action_space[0], data={"x": cx, "y": cy})
            total_steps += 1
            # Determine zone response
            if obs is not None:
                fh = hash(np.array(obs.frame[0]).tobytes())
                last_zone = zone_frames.get(fh, 0)
            if fold.V.shape[0] >= n_zones: seeded = True
            continue
        if not seeded: seeded = True

        zone = fold.pn(enc_t, nc=n_zones)
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
        cx, cy = zone_representatives[zone]

        ol = obs.levels_completed
        obs = env.step(action_space[0], data={"x": cx, "y": cy})
        total_steps += 1
        if obs is None: break

        # Determine zone response from resulting frame
        fh = hash(np.array(obs.frame[0]).tobytes())
        last_zone = zone_frames.get(fh, 0)

        if obs.levels_completed > ol:
            levels = obs.levels_completed
            print(f"    LEVEL {levels} at step {total_steps} cb={fold.V.shape[0]}", flush=True)

        if total_steps % 500 == 0:
            print(f"    [step {total_steps:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.4f}"
                  f"  levels={levels} go={go} zones={zone_counts}", flush=True)

        if obs.state == GameState.WIN:
            print(f"    WIN at step {total_steps}!", flush=True); break

    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 375 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"zones={n_zones}  levels={levels}  steps={total_steps}  go={go}", flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
    print(f"zone_counts: {zone_counts}", flush=True)
    if levels > 0:
        print("PASS: VC33 level completed with zone-based process()!", flush=True)
    else:
        print("KILL: no level completion.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
