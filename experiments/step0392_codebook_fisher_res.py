#!/usr/bin/env python3
"""
Step 392 -- Adaptive resolution with Fisher discriminant ratio.

Fisher score = between-class distance / (1 - within-class similarity).
Selects resolution where different actions produce distinguishable outcomes.

LS20. 200 steps per resolution. 50K at selected.
Script: scripts/run_step392_fisher_res.py
"""

import time, math, logging, numpy as np, torch, torch.nn.functional as F
from itertools import combinations
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

def avgpool(frame, res):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    if res == 64: return arr.flatten()
    k = 64 // res
    return arr.reshape(res, k, res, k).mean(axis=(1, 3)).flatten()

def centered_enc(pooled, fold):
    t = F.normalize(torch.from_numpy(pooled.astype(np.float32)), dim=0)
    if fold.V.shape[0] > 2: t = t - fold.V.mean(dim=0).cpu()
    return t

def fisher_score(fold, na):
    """Compute Fisher discriminant: between-class / (1 - within-class)."""
    if fold.V.shape[0] < na * 2: return 0.0, 0.0, 0.0

    # Within-class: avg pairwise cos_sim within same action
    within = 0.0; n_within = 0
    class_means = []
    for c in range(na):
        m = (fold.labels == c)
        entries = fold.V[m]
        if entries.shape[0] > 1:
            s = entries @ entries.T
            # Exclude diagonal
            mask = ~torch.eye(entries.shape[0], device=fold.dev, dtype=torch.bool)
            within += float(s[mask].mean().item())
            n_within += 1
        if entries.shape[0] > 0:
            class_means.append(F.normalize(entries.mean(dim=0), dim=0))
        else:
            class_means.append(None)

    within = within / max(n_within, 1)

    # Between-class: avg (1 - cos_sim) between class means
    between = 0.0; n_between = 0
    for i, j in combinations(range(na), 2):
        if class_means[i] is not None and class_means[j] is not None:
            cs = float((class_means[i] @ class_means[j]).item())
            between += (1.0 - cs)
            n_between += 1
    between = between / max(n_between, 1)

    score = between / (1.0 - within + 1e-8)
    return score, within, between

def explore_resolution(arc, game_id, res, steps=200):
    from arcengine import GameState
    d = res * res
    fold = CF(d=d, k=3); env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space); sd = False; ts = 0; go = 0

    while ts < steps and go < 20:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset()
            if obs is None: break; continue
        if obs.state == GameState.WIN: break
        enc = centered_enc(avgpool(obs.frame, res), fold)
        if not sd and fold.V.shape[0] < na:
            i = fold.V.shape[0]; fold._fa(enc, i)
            obs = env.step(env.action_space[i]); ts += 1
            if fold.V.shape[0] >= na: sd = True; fold._ut()
            continue
        if not sd: sd = True
        c = fold.pn(enc, nc=na)
        obs = env.step(env.action_space[c % na]); ts += 1
        if obs is None: break

    fs, within, between = fisher_score(fold, na)
    return {'res': res, 'dims': d, 'cb': fold.V.shape[0], 'fisher': fs,
            'within': within, 'between': between}

def run_game(arc, game_id, res, max_steps=50000):
    from arcengine import GameState
    d = res * res; fold = CF(d=d, k=3); env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space); ts = 0; go = 0; lvls = 0; sd = False
    while ts < max_steps and go < 500:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset()
            if obs is None: break; continue
        if obs.state == GameState.WIN: break
        enc = centered_enc(avgpool(obs.frame, res), fold)
        if not sd and fold.V.shape[0] < na:
            i = fold.V.shape[0]; fold._fa(enc, i)
            obs = env.step(env.action_space[i]); ts += 1
            if fold.V.shape[0] >= na: sd = True; fold._ut()
            continue
        if not sd: sd = True
        c = fold.pn(enc, nc=na); ol = obs.levels_completed
        obs = env.step(env.action_space[c % na]); ts += 1
        if obs is None: break
        if obs.levels_completed > ol:
            lvls = obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} go={go}", flush=True)
        if ts % 10000 == 0:
            print(f"    [step {ts:6d}] cb={fold.V.shape[0]} levels={lvls} go={go}", flush=True)
    return {'levels': lvls, 'steps': ts, 'go': go, 'cb': fold.V.shape[0]}

def main():
    t0 = time.time()
    print("Step 392 -- Adaptive resolution with Fisher score. LS20.", flush=True)
    print(f"Device: {DEVICE}", flush=True); print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    print("=== EXPLORATION (Fisher score per resolution) ===", flush=True)
    resolutions = [64, 32, 16, 8]
    results = []
    for res in resolutions:
        r = explore_resolution(arc, ls20.game_id, res, steps=200)
        results.append(r)
        print(f"  {res:2d}x{res:2d}: fisher={r['fisher']:.4f}"
              f"  within={r['within']:.4f}  between={r['between']:.4f}"
              f"  cb={r['cb']}", flush=True)

    best = max(results, key=lambda r: r['fisher'])
    print(f"\n  SELECTED: {best['res']}x{best['res']} (fisher={best['fisher']:.4f})", flush=True)
    print(flush=True)

    print(f"=== EXPLOITATION ({best['res']}x{best['res']}, 50K steps) ===", flush=True)
    game_r = run_game(arc, ls20.game_id, best['res'], max_steps=50000)

    elapsed = time.time() - t0
    print(flush=True); print("=" * 60, flush=True)
    print("STEP 392 SUMMARY", flush=True); print("=" * 60, flush=True)
    print(f"Selected: {best['res']}x{best['res']}", flush=True)
    for r in results:
        m = " <-- SELECTED" if r['res'] == best['res'] else ""
        print(f"  {r['res']:2d}x{r['res']:2d}: fisher={r['fisher']:.4f}"
              f"  within={r['within']:.4f}  between={r['between']:.4f}{m}", flush=True)
    print(f"\nGame: levels={game_r['levels']} steps={game_r['steps']} go={game_r['go']}", flush=True)
    if game_r['levels'] > 0:
        print(f"\nPASS: Level at self-discovered {best['res']}x{best['res']}!", flush=True)
    elif best['res'] == 16:
        print(f"\nMARGINAL: Correctly selected 16x16, no level (stochastic).", flush=True)
    else:
        print(f"\nKILL: Selected {best['res']}x{best['res']}.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__ == '__main__':
    main()
