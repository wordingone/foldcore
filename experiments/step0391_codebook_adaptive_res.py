#!/usr/bin/env python3
"""
Step 391 -- Adaptive resolution discovery. LS20.

Substrate tries 4 resolutions (64,32,16,8), measures discrimination×diversity,
picks the best. Then runs full game at selected resolution.

Stage 8: encoding IS state-derived.
Script: scripts/run_step391_adaptive_res.py
"""

import time, math, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CB_CAP = 10000; THRESH_INT = 100

class CF:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0
    def reset(self):
        self.V=torch.zeros(0,self.d,device=self.dev)
        self.labels=torch.zeros(0,dtype=torch.long,device=self.dev)
        self.thresh=0.7; self.spawn_count=0
    def _fa(self,x,l):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INT==0: self._ut()
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
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts=si.clone(); ts[~tm]=-float('inf'); w=ts.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return p
    def sim_std(self):
        if self.V.shape[0]<10: return 0.0
        ss=min(200,self.V.shape[0]); idx=torch.randperm(self.V.shape[0],device=self.dev)[:ss]
        s=self.V[idx]@self.V.T
        return float(s.std().item())


def avgpool(frame, res):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    if res == 64: return arr.flatten()
    k = 64 // res
    return arr.reshape(res, k, res, k).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, fold):
    t = F.normalize(torch.from_numpy(pooled.astype(np.float32)), dim=0)
    if fold.V.shape[0] > 2:
        t = t - fold.V.mean(dim=0).cpu()
    return t


def action_entropy(counts):
    total = sum(counts.values())
    if total == 0: return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def explore_resolution(arc, game_id, res, steps=200):
    from arcengine import GameState
    d = res * res
    fold = CF(d=d, k=3)
    env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space)
    action_counts = {}; sd = False; ts = 0; go = 0

    while ts < steps and go < 20:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset()
            if obs is None: break; continue
        if obs.state == GameState.WIN: break

        enc = centered_enc(avgpool(obs.frame, res), fold)
        if not sd and fold.V.shape[0] < na:
            i = fold.V.shape[0]; fold._fa(enc, i)
            obs = env.step(env.action_space[i]); ts += 1
            action_counts[env.action_space[i].name] = action_counts.get(env.action_space[i].name, 0) + 1
            if fold.V.shape[0] >= na: sd = True; fold._ut()
            continue
        if not sd: sd = True

        c = fold.pn(enc, nc=na)
        action = env.action_space[c % na]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1
        obs = env.step(action); ts += 1
        if obs is None: break

    ss = fold.sim_std()
    ae = action_entropy(action_counts)
    score = ss * ae
    return {'res': res, 'dims': d, 'cb': fold.V.shape[0], 'sim_std': ss,
            'action_entropy': ae, 'score': score, 'actions': action_counts}


def run_game(arc, game_id, res, max_steps=50000):
    from arcengine import GameState
    d = res * res
    fold = CF(d=d, k=3); env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space)
    ts = 0; go = 0; lvls = 0; sd = False
    action_counts = {}

    while ts < max_steps and go < 500:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset()
            if obs is None: break; continue
        if obs.state == GameState.WIN: break

        enc = centered_enc(avgpool(obs.frame, res), fold)
        if not sd and fold.V.shape[0] < na:
            i = fold.V.shape[0]; fold._fa(enc, i)
            obs = env.step(env.action_space[i]); ts += 1
            action_counts[env.action_space[i].name] = action_counts.get(env.action_space[i].name, 0) + 1
            if fold.V.shape[0] >= na: sd = True; fold._ut()
            continue
        if not sd: sd = True

        c = fold.pn(enc, nc=na)
        action = env.action_space[c % na]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1
        ol = obs.levels_completed
        obs = env.step(action); ts += 1
        if obs is None: break
        if obs.levels_completed > ol:
            lvls = obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} go={go}", flush=True)
        if ts % 10000 == 0:
            print(f"    [step {ts:6d}] cb={fold.V.shape[0]} levels={lvls} go={go}", flush=True)

    return {'levels': lvls, 'steps': ts, 'go': go, 'cb': fold.V.shape[0],
            'actions': action_counts}


def main():
    t0 = time.time()
    print("Step 391 -- Adaptive resolution discovery. LS20.", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    # Phase 1: Explore 4 resolutions
    print("=== EXPLORATION PHASE (200 steps each) ===", flush=True)
    resolutions = [64, 32, 16, 8]
    results = []
    for res in resolutions:
        r = explore_resolution(arc, ls20.game_id, res, steps=200)
        results.append(r)
        print(f"  {res}x{res} ({r['dims']}d): sim_std={r['sim_std']:.4f}"
              f"  entropy={r['action_entropy']:.3f}  score={r['score']:.4f}"
              f"  cb={r['cb']}  acts={r['actions']}", flush=True)

    # Select best
    best = max(results, key=lambda r: r['score'])
    print(f"\n  SELECTED: {best['res']}x{best['res']} (score={best['score']:.4f})", flush=True)
    print(flush=True)

    # Phase 2: Full run at selected resolution
    print(f"=== EXPLOITATION PHASE ({best['res']}x{best['res']}, 50K steps) ===", flush=True)
    game_r = run_game(arc, ls20.game_id, best['res'], max_steps=50000)

    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 391 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Selected resolution: {best['res']}x{best['res']} ({best['dims']}d)", flush=True)
    print(f"Selection scores:", flush=True)
    for r in results:
        marker = " <-- SELECTED" if r['res'] == best['res'] else ""
        print(f"  {r['res']:2d}x{r['res']:2d}: sim_std={r['sim_std']:.4f}"
              f"  entropy={r['action_entropy']:.3f}"
              f"  score={r['score']:.4f}{marker}", flush=True)
    print(flush=True)
    print(f"Game result: levels={game_r['levels']}  steps={game_r['steps']}"
          f"  go={game_r['go']}  cb={game_r['cb']}", flush=True)
    print(f"actions: {game_r['actions']}", flush=True)
    if game_r['levels'] > 0 and best['res'] != 16:
        print(f"\nPASS: Level completed at self-discovered resolution {best['res']}x{best['res']}!", flush=True)
    elif game_r['levels'] > 0:
        print(f"\nPASS: Level completed. Substrate correctly discovered 16x16.", flush=True)
    elif best['res'] == 16:
        print(f"\nMARGINAL: Correctly selected 16x16 but no level (stochastic).", flush=True)
    else:
        print(f"\nKILL: Wrong resolution selected ({best['res']}x{best['res']}) or no level.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
