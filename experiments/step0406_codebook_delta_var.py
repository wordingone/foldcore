#!/usr/bin/env python3
"""
Step 406 -- Attract-delta variance as self-derived encoding.

Track per-dim variance of attract deltas. Timer = low var (predictable).
Sprite = high var (unpredictable). Mask keeps high-variance dims.
Process on masked + centered + unnormalized representation.

LS20. 64x64. 50K steps.
Script: scripts/run_step406_delta_var.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100; MEAN_BOOT = 200; DELTA_WARMUP = 500

class DeltaVarFold:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.99; self.k=k; self.d=d; self.dev=dev
        self.spawn_count=0; self.mean=None; self.phase2=False
        # Delta variance tracking (Welford's)
        self.delta_mean=torch.zeros(d,device=dev)
        self.delta_m2=torch.zeros(d,device=dev)
        self.delta_n=0
        # Mask
        self.mask=None; self.n_active=0; self.mask_ready=False

    def _update_mean(self):
        if self.V.shape[0]>=MEAN_BOOT:
            self.mean=self.V.mean(dim=0)
            if not self.phase2: self.phase2=True

    def _update_delta_var(self, delta):
        """Welford's online variance on absolute delta."""
        d=delta.abs()
        self.delta_n+=1
        old_mean=self.delta_mean.clone()
        self.delta_mean+=(d-self.delta_mean)/self.delta_n
        self.delta_m2+=(d-old_mean)*(d-self.delta_mean)

    def _compute_mask(self):
        if self.delta_n<DELTA_WARMUP: return
        variance=self.delta_m2/self.delta_n
        v_active=variance[variance>1e-10]
        if len(v_active)<10: return
        thresh_var=float(v_active.mean()+2*v_active.std())
        self.mask=(variance>thresh_var).float()
        self.n_active=int(self.mask.sum().item())
        self.mask_ready=self.n_active>=5

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
        if self.mask_ready and self.mask is not None:
            # Masked + centered
            Vm=self.V*self.mask.unsqueeze(0)
            m=Vm.mean(dim=0)
            Vc=Vm-m.unsqueeze(0)
            s=Vc[idx]@Vc.T
        elif self.phase2 and self.mean is not None:
            Vc=self.V-self.mean.unsqueeze(0)
            s=Vc[idx]@Vc.T
        else:
            Vn=F.normalize(self.V,dim=1)
            s=Vn[idx]@Vn.T
        for i,j in enumerate(idx): s[i,j]=-1e9
        nn=s.max(dim=1).values
        self.thresh=float(nn.median())

    def pn(self,x,nc):
        x=x.to(self.dev).float()
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0

        # Similarity computation
        if self.mask_ready and self.mask is not None:
            xm=x*self.mask
            Vm=self.V*self.mask.unsqueeze(0)
            m=Vm.mean(dim=0)
            xc=xm-m; Vc=Vm-m.unsqueeze(0)
            si=Vc@xc
        elif self.phase2 and self.mean is not None:
            xc=x-self.mean; Vc=self.V-self.mean.unsqueeze(0)
            si=Vc@xc
        else:
            xn=F.normalize(x,dim=0); Vn=F.normalize(self.V,dim=1)
            si=Vn@xn

        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)

        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)])
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count==MEAN_BOOT:
                self._update_mean(); self._ut()
            elif self.spawn_count%THRESH_INT==0:
                if self.phase2: self._update_mean()
                self._ut()
        else:
            w_i=si.argmax().item()
            alpha=1.0/(1.0+max(float(si[w_i].item()),0.0))
            delta=alpha*(x-self.V[w_i])
            self.V[w_i]=self.V[w_i]+delta
            # Track delta variance
            self._update_delta_var(delta)
            # Check if mask should activate
            if not self.mask_ready and self.delta_n>=DELTA_WARMUP:
                self._compute_mask()
                if self.mask_ready:
                    self._ut()  # recompute thresh with mask
        return p


def main():
    t0=time.time()
    print(f"Step 406 -- Attract-delta variance encoding. 64x64. 50K. LS20.", flush=True)
    print(f"Device: {DEVICE}  delta_warmup={DELTA_WARMUP}  mean_boot={MEAN_BOOT}", flush=True)
    print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=DeltaVarFold(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

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

        c=fold.pn(raw,nc=na)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        # Log mask activation
        if fold.mask_ready and not mask_logged:
            mask_logged=True
            mask_np=fold.mask.cpu().numpy()
            active_idx=np.where(mask_np>0)[0]
            rows=sorted(set(active_idx//64))
            cols=sorted(set(active_idx%64))
            var_np=(fold.delta_m2/fold.delta_n).cpu().numpy()
            print(f"    MASK ACTIVATED at step {ts}: n_active={fold.n_active}", flush=True)
            print(f"      Active rows: {rows[:20]}{'...' if len(rows)>20 else ''}", flush=True)
            print(f"      Active cols: {cols[:20]}{'...' if len(cols)>20 else ''}", flush=True)
            print(f"      Delta var: min={var_np.min():.6f} max={var_np.max():.6f}"
                  f"  mean={var_np.mean():.6f} std={var_np.std():.6f}", flush=True)
            active_var=var_np[mask_np>0]
            print(f"      Active var: min={active_var.min():.6f} max={active_var.max():.6f}"
                  f"  mean={active_var.mean():.6f}", flush=True)
            # Compare to Step 400 change-rate dims
            print(f"      thresh_var used: mean+2std = {var_np[var_np>1e-10].mean()+2*var_np[var_np>1e-10].std():.6f}", flush=True)

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]}"
                  f"  n_active={fold.n_active} go={go}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            mode='MASKED' if fold.mask_ready else ('DOT' if fold.phase2 else 'COS')
            print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.2f}"
                  f"  mode={mode}  n_active={fold.n_active}"
                  f"  delta_n={fold.delta_n}  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 406 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"cb={fold.V.shape[0]}  thresh={fold.thresh:.2f}  mask_ready={fold.mask_ready}"
          f"  n_active={fold.n_active}  delta_n={fold.delta_n}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0

    if fold.delta_n>0:
        var_np=(fold.delta_m2/fold.delta_n).cpu().numpy()
        print(f"\nDelta variance distribution:", flush=True)
        for lo,hi,label in [(0,1e-6,'zero'),(1e-6,1e-4,'low'),(1e-4,1e-2,'medium'),(1e-2,1,'high')]:
            n=((var_np>=lo)&(var_np<hi)).sum()
            print(f"  [{lo:.0e},{hi:.0e}): {n} dims ({n/len(var_np)*100:.1f}%)", flush=True)

    if fold.mask is not None:
        mask_np=fold.mask.cpu().numpy()
        active_idx=np.where(mask_np>0)[0]
        if len(active_idx)>0:
            rows=sorted(set(active_idx//64))
            print(f"\nActive rows: {rows}", flush=True)

    if lvls>0:
        print(f"\nPASS: Level with delta-variance encoding at 64x64!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% dominance.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
