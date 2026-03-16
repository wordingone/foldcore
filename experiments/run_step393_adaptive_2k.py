#!/usr/bin/env python3
"""
Step 393 -- Adaptive resolution: 2000 steps/res + self-feeding displacement test.

2000 steps per resolution (not 200). Self-feed 100 entries through process()
and measure total codebook displacement. Most reorganization = best resolution.

LS20. 50K at selected.
Script: scripts/run_step393_adaptive_2k.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
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
    def self_feed_displacement(self, n_samples=100):
        """Feed random entries through process, measure displacement of EXISTING entries."""
        n=self.V.shape[0]
        if n<10: return 0.0
        nc=int(self.labels.max().item())+1
        V_before=self.V[:n].clone()
        labels_before=self.labels[:n].clone()
        idx=torch.randperm(n,device=self.dev)[:min(n_samples,n)]
        for i in idx:
            self.pn(V_before[i],nc=nc)
        # Only compare original entries (first n rows)
        n_compare=min(n,self.V.shape[0])
        disp=float((self.V[:n_compare]-V_before[:n_compare]).norm(dim=1).sum().item())
        # Restore
        self.V=V_before; self.labels=labels_before
        return disp

def avgpool(frame, res):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    if res == 64: return arr.flatten()
    k = 64 // res
    return arr.reshape(res, k, res, k).mean(axis=(1, 3)).flatten()

def centered_enc(pooled, fold):
    t = F.normalize(torch.from_numpy(pooled.astype(np.float32)), dim=0)
    if fold.V.shape[0] > 2: t = t - fold.V.mean(dim=0).cpu()
    return t

def explore_resolution(arc, game_id, res, steps=2000):
    from arcengine import GameState
    d = res * res; fold = CF(d=d, k=3)
    env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space); sd = False; ts = 0; go = 0
    action_counts = {}
    while ts < steps and go < 50:
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

    disp = fold.self_feed_displacement(100)
    # Also compute Fisher for comparison
    within = 0.0; nw = 0; class_means = []
    for c in range(na):
        m = (fold.labels == c); entries = fold.V[m]
        if entries.shape[0] > 1:
            s = entries @ entries.T
            mask = ~torch.eye(entries.shape[0], device=fold.dev, dtype=torch.bool)
            within += float(s[mask].mean().item()); nw += 1
        if entries.shape[0] > 0:
            class_means.append(F.normalize(entries.mean(0), dim=0))
        else: class_means.append(None)
    within /= max(nw, 1)
    between = 0.0; nb = 0
    for i, j in combinations(range(na), 2):
        if class_means[i] is not None and class_means[j] is not None:
            between += 1.0 - float((class_means[i] @ class_means[j]).item()); nb += 1
    between /= max(nb, 1)
    fisher = between / (1.0 - within + 1e-8)

    return {'res': res, 'dims': d, 'cb': fold.V.shape[0], 'displacement': disp,
            'fisher': fisher, 'within': within, 'between': between, 'actions': action_counts}

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
    print("Step 393 -- Adaptive resolution: 2000 steps/res + self-feed. LS20.", flush=True)
    print(f"Device: {DEVICE}", flush=True); print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    print("=== EXPLORATION (2000 steps each) ===", flush=True)
    results = []
    for res in [64, 32, 16, 8]:
        r = explore_resolution(arc, ls20.game_id, res, steps=2000)
        results.append(r)
        print(f"  {res:2d}x{res:2d}: displacement={r['displacement']:.2f}"
              f"  fisher={r['fisher']:.4f}  cb={r['cb']}"
              f"  within={r['within']:.3f}  between={r['between']:.4f}", flush=True)

    best = max(results, key=lambda r: r['displacement'])
    print(f"\n  SELECTED: {best['res']}x{best['res']} (displacement={best['displacement']:.2f})", flush=True)
    print(flush=True)

    print(f"=== EXPLOITATION ({best['res']}x{best['res']}, 50K) ===", flush=True)
    game_r = run_game(arc, ls20.game_id, best['res'], max_steps=50000)

    elapsed = time.time() - t0
    print(flush=True); print("=" * 60, flush=True)
    print("STEP 393 SUMMARY", flush=True); print("=" * 60, flush=True)
    print(f"Selected: {best['res']}x{best['res']}", flush=True)
    for r in results:
        m = " <-- SELECTED" if r['res'] == best['res'] else ""
        print(f"  {r['res']:2d}x{r['res']:2d}: disp={r['displacement']:.2f}"
              f"  fisher={r['fisher']:.4f}{m}", flush=True)
    print(f"\nGame: levels={game_r['levels']} steps={game_r['steps']} go={game_r['go']}", flush=True)
    if game_r['levels'] > 0:
        print(f"\nPASS!", flush=True)
    elif best['res'] == 16:
        print(f"\nMARGINAL: correct resolution, stochastic failure.", flush=True)
    else:
        print(f"\nKILL.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__ == '__main__':
    main()
