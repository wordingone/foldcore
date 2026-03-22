#!/usr/bin/env python3
"""
Step 410 -- Spawn-delta importance mask. Substrate discovers dims from growth.

At each spawn, record per-dim |x - V[winner]|. P95 per dim separates
sprite (high) from timer (low). Mask -> normalized cosine -> Goldilocks.

LS20. 64x64. 50K steps.
Script: scripts/run_step410_spawn_mask.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100; MEAN_BOOT = 200
MIN_SPAWNS = 200; MASK_UPDATE = 100

class SpawnMaskFold:
    def __init__(self, d, nc, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.99; self.k=k; self.d=d; self.nc=nc; self.dev=dev
        self.spawn_count=0; self.mean=None; self.phase2=False
        self.spawn_deltas=[]; self.mask=None; self.n_active=0
        self.step_count=0

    def _update_mean(self):
        if self.V.shape[0]>=MEAN_BOOT:
            self.mean=self.V.mean(dim=0)
            if not self.phase2: self.phase2=True

    def _update_mask(self):
        if len(self.spawn_deltas)<MIN_SPAWNS: return
        all_d=torch.stack(self.spawn_deltas[-1000:])  # last 1000 spawns max
        p95=torch.quantile(all_d.float(), 0.95, dim=0)
        p_active=p95[p95>1e-6]
        if len(p_active)<10: return
        thresh_mask=float(p_active.mean()+2*p_active.std())
        self.mask=(p95>thresh_mask).float()
        self.n_active=int(self.mask.sum().item())

    def _compute_sims(self, x):
        if self.mask is not None and self.n_active>=5:
            xm=F.normalize(x*self.mask, dim=0)
            Vm=F.normalize(self.V*self.mask.unsqueeze(0), dim=1)
            return Vm@xm
        elif self.phase2 and self.mean is not None:
            xc=x-self.mean; Vc=self.V-self.mean.unsqueeze(0)
            return Vc@xc
        else:
            return F.normalize(self.V,dim=1)@F.normalize(x,dim=0)

    def _fa(self,x,l):
        self.V=torch.cat([self.V,x.to(self.dev).float().unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count==MEAN_BOOT:
            self._update_mean(); self._ut()
        elif self.spawn_count%THRESH_INT==0:
            if self.phase2: self._update_mean()
            self._ut()

    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        if self.mask is not None and self.n_active>=5:
            Vm=F.normalize(self.V*self.mask.unsqueeze(0), dim=1)
            s=Vm[idx]@Vm.T
        elif self.phase2 and self.mean is not None:
            Vc=self.V-self.mean.unsqueeze(0)
            s=Vc[idx]@Vc.T
        else:
            Vn=F.normalize(self.V,dim=1)
            s=Vn[idx]@Vn.T
        t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())

    def process(self, x):
        x=x.to(self.dev).float()
        self.step_count+=1

        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev)
            return 0

        si=self._compute_sims(x)

        # Class vote -> argmin
        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,self.nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        action=sc[:self.nc].argmin().item()
        p=action; tm=(self.labels==p)

        # Find winner for spawn delta
        w_i=int(si.argmax().item())

        # Spawn/attract
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            # Record spawn delta
            delta=torch.abs(x-self.V[w_i])
            self.spawn_deltas.append(delta)
            # Spawn
            self.V=torch.cat([self.V,x.unsqueeze(0)])
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count==MEAN_BOOT:
                self._update_mean(); self._ut()
            elif self.spawn_count%THRESH_INT==0:
                if self.phase2: self._update_mean()
                self._ut()
            # Update mask periodically
            if len(self.spawn_deltas)>=MIN_SPAWNS and self.spawn_count%MASK_UPDATE==0:
                old_active=self.n_active
                self._update_mask()
                if self.n_active!=old_active and self.n_active>=5:
                    self._ut()  # recompute thresh with new mask
        else:
            alpha=1.0/(1.0+max(float(si[w_i].item()),0.0)) if self.mask is None else (1.0-float(si[w_i].item()))
            self.V[w_i]=self.V[w_i]+alpha*(x-self.V[w_i])
            if self.mask is not None and self.n_active>=5:
                self.V[w_i]=self.V[w_i]  # no renormalization needed — stored raw

        return action


def main():
    t0=time.time()
    print(f"Step 410 -- Spawn-delta importance mask. 64x64. 50K. LS20.", flush=True)
    print(f"Device: {DEVICE}  min_spawns={MIN_SPAWNS}  cb_cap={CB_CAP}", flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)
    fold=SpawnMaskFold(d=D, nc=na, k=3)

    ts=0; go=0; lvls=0; sd=False
    action_counts={}; mask_logged=False

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break
            continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!", flush=True); break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)

        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(raw,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if fold.V.shape[0]>=na: sd=True; fold._ut()
            continue
        if not sd: sd=True

        c=fold.process(raw)

        # Log mask activation
        if fold.mask is not None and fold.n_active>=5 and not mask_logged:
            mask_logged=True
            mask_np=fold.mask.cpu().numpy()
            active_idx=np.where(mask_np>0)[0]
            rows=sorted(set(active_idx//64))
            p95_np=torch.quantile(torch.stack(fold.spawn_deltas[-500:]).float(),0.95,dim=0).cpu().numpy()
            print(f"    MASK ACTIVATED at step {ts}: n_active={fold.n_active}", flush=True)
            print(f"      Active rows: {rows[:25]}{'...' if len(rows)>25 else ''}", flush=True)
            print(f"      p95: min={p95_np.min():.4f} max={p95_np.max():.4f}"
                  f"  mean={p95_np.mean():.4f} std={p95_np.std():.4f}", flush=True)
            active_p95=p95_np[mask_np>0]
            print(f"      Active p95: min={active_p95.min():.4f} max={active_p95.max():.4f}"
                  f"  mean={active_p95.mean():.4f}", flush=True)

        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]}"
                  f"  n_active={fold.n_active} go={go}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            mode='MASKED_COS' if (fold.mask is not None and fold.n_active>=5) else ('DOT' if fold.phase2 else 'COS')
            print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.3f}"
                  f"  mode={mode}  n_active={fold.n_active}"
                  f"  spawns={len(fold.spawn_deltas)}"
                  f"  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 410 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"cb={fold.V.shape[0]}  thresh={fold.thresh:.3f}  n_active={fold.n_active}", flush=True)
    print(f"total_spawns={len(fold.spawn_deltas)}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0

    if fold.mask is not None:
        mask_np=fold.mask.cpu().numpy()
        active_idx=np.where(mask_np>0)[0]
        if len(active_idx)>0:
            rows=sorted(set(active_idx//64))
            print(f"Active rows: {rows}", flush=True)

    if lvls>0:
        print(f"\nPASS: Level with spawn-delta mask at 64x64!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% dominance.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
