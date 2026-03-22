#!/usr/bin/env python3
"""
Step 382 -- Raw 64x64 DIFFS + variance weighting. LS20.

Codebook stores frame diffs (what changed), not absolute frames.
Variance weighting: timer diffs constant→suppressed, sprite diffs variable→amplified.
2K steps. Kill: sim variance increase vs Step 381.
Script: scripts/run_step382_raw64_diffs.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INTERVAL = 100

class VarFold:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0
        self.weights=torch.ones(d,device=dev); self.step_count=0
    def _fa(self,x,l):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INTERVAL==0: self._ut()
    def update_weights(self):
        if self.V.shape[0]<10: return
        cv=self.V.var(dim=0); mx=cv.max()
        self.weights=(cv/mx) if mx>0 else torch.ones(self.d,device=self.dev)
    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        w=self.weights.unsqueeze(0)
        wV_s=F.normalize(self.V[idx]*w,dim=1); wV_a=F.normalize(self.V*w,dim=1)
        s=wV_s@wV_a.T; t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())
    def pn(self,x,nc):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        self.step_count+=1
        if self.step_count%100==0: self.update_weights()
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0,0.0
        w=self.weights
        wx=F.normalize(x*w,dim=0); wV=F.normalize(self.V*w.unsqueeze(0),dim=1)
        si=wV@wx
        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)
        nsim=float(si.max().item())
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INTERVAL==0: self._ut()
        else:
            ts=si.clone(); ts[~tm]=-float('inf'); w_i=ts.argmax().item()
            raw_sim=float((self.V[w_i]@x).item())
            a=1.0-raw_sim; self.V[w_i]=F.normalize(self.V[w_i]+a*(x-self.V[w_i]),dim=0)
        return p,nsim


def main():
    t0=time.time()
    print(f"Step 382 -- Raw 64x64 DIFFS + variance weighting on LS20.",flush=True)
    print(f"Device: {DEVICE}",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=VarFold(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    action_counts={}; sims_log=[]
    prev_raw=None

    while ts<2000 and go<50:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset(); prev_raw=None
            if obs is None: break; continue
        if obs.state==GameState.WIN: break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)

        # Compute diff
        if prev_raw is not None:
            diff=raw-prev_raw
            # Skip if diff is all zeros (no change)
            if diff.abs().max() < 1e-6:
                prev_raw=raw.clone()
                obs=env.step(env.action_space[0]); ts+=1
                continue
            diff_norm=F.normalize(diff,dim=0)
        else:
            diff_norm=F.normalize(raw,dim=0)  # first frame: use raw

        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(diff_norm,i)
            action=env.action_space[i]
            action_counts[action.name]=action_counts.get(action.name,0)+1
            prev_raw=raw.clone()
            obs=env.step(action); ts+=1
            if fold.V.shape[0]>=na: sd=True; fold.update_weights(); fold._ut()
            continue
        if not sd: sd=True

        c,nsim=fold.pn(diff_norm,nc=na)
        sims_log.append(nsim)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        prev_raw=raw.clone()
        obs=env.step(action); ts+=1
        if obs is None: break

        if ts%500==0:
            w=fold.weights.cpu().numpy()
            n_high=(w>0.5).sum(); n_zero=(w<0.001).sum()
            avg_sim=np.mean(sims_log[-100:]) if sims_log else 0
            sim_std=np.std(sims_log[-100:]) if len(sims_log)>10 else 0
            print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.6f}"
                  f"  avg_sim={avg_sim:.4f} sim_std={sim_std:.4f}"
                  f"  high_dims={n_high} zero_dims={n_zero}",flush=True)
            print(f"      actions: {action_counts}",flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 382 SUMMARY",flush=True); print("="*60,flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}",flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.6f}",flush=True)
    print(f"action_counts: {action_counts}",flush=True)

    if sims_log:
        sl=np.array(sims_log)
        print(f"\nWeighted diff sim: min={sl.min():.4f} max={sl.max():.4f} mean={sl.mean():.4f} std={sl.std():.4f}",flush=True)
        print(f"Compare 377 (raw abs):    mean=0.984 std=0.009",flush=True)
        print(f"Compare 381 (weighted abs): mean=0.994 std=0.009",flush=True)

    w=fold.weights.cpu().numpy()
    n_high=(w>0.5).sum(); n_zero=(w<0.001).sum(); n_active=(w>0.01).sum()
    print(f"\nDiff-weight stats: active={n_active} high={n_high} zero={n_zero} / {D}",flush=True)

    # Top rows
    w_grid=w.reshape(64,64)
    row_weights=w_grid.mean(axis=1)
    top_rows=np.argsort(row_weights)[-5:][::-1]
    print(f"Top 5 rows by diff-variance weight:",flush=True)
    for r in top_rows:
        print(f"  row {r:2d}: avg_weight={row_weights[r]:.4f}",flush=True)

    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
