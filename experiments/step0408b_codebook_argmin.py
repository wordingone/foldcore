#!/usr/bin/env python3
"""
Step 408b -- Mask + centered unnorm + ARGMIN (standard process).

Same mask+centered as 408, but standard argmin on class votes
instead of count-based cycling. Let the codebook decide.

LS20. 64x64. 50K steps.
Script: scripts/run_step408b_argmin.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100; MASK_WARMUP = 200; EPS = 0.01

class CombinedFold:
    def __init__(self, d, nc, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.99; self.k=k; self.d=d; self.nc=nc; self.dev=dev
        self.spawn_count=0
        # Change-rate tracking
        self.x_prev=None; self.change_count=torch.zeros(d,device=dev)
        self.frame_count=0
        # Mask
        self.mask=None; self.n_active=0; self.mask_ready=False
        # Count-based
        self.win_action_counts={}
        self.unique_winners=set()

    def _update_stats(self, x):
        x=x.to(self.dev).float()
        if self.x_prev is not None:
            changed=(torch.abs(x-self.x_prev)>EPS).float()
            self.change_count+=changed
        self.frame_count+=1
        self.x_prev=x.clone()

    def _compute_mask(self):
        if self.frame_count<MASK_WARMUP: return
        rate=self.change_count/self.frame_count
        active=rate>0.001
        if active.sum()<10: return
        r=rate[active]
        threshold=float(r.mean()+2*r.std())
        self.mask=(rate>threshold).float()
        self.n_active=int(self.mask.sum().item())
        self.mask_ready=self.n_active>=5

    def _compute_sims(self, x):
        """Masked + centered unnormalized similarity."""
        if self.mask_ready and self.mask is not None:
            xm=x*self.mask
            Vm=self.V*self.mask.unsqueeze(0)
            m=Vm.mean(dim=0)
            xc=xm-m; Vc=Vm-m.unsqueeze(0)
            return Vc@xc
        else:
            # Pre-mask: standard centered unnorm (after bootstrap) or cosine
            if self.V.shape[0]>50:
                mean=self.V.mean(dim=0)
                xc=x-mean; Vc=self.V-mean.unsqueeze(0)
                return Vc@xc
            else:
                return F.normalize(self.V,dim=1)@F.normalize(x,dim=0)

    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        if self.mask_ready and self.mask is not None:
            Vm=self.V*self.mask.unsqueeze(0)
            m=Vm.mean(dim=0); Vc=Vm-m.unsqueeze(0)
            s=Vc[idx]@Vc.T
        elif self.V.shape[0]>50:
            mean=self.V.mean(dim=0); Vc=self.V-mean.unsqueeze(0)
            s=Vc[idx]@Vc.T
        else:
            Vn=F.normalize(self.V,dim=1)
            s=Vn[idx]@Vn.T
        for i,j in enumerate(idx): s[i,j]=-1e9
        nn=s.max(dim=1).values
        self.thresh=float(nn.median())

    def _fa(self, x, l):
        self.V=torch.cat([self.V,x.to(self.dev).float().unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INT==0: self._ut()

    def process(self, x):
        """Combined: mask + centered unnorm + standard argmin."""
        x=x.to(self.dev).float()
        self._update_stats(x)

        # Check mask activation
        if not self.mask_ready and self.frame_count>=MASK_WARMUP:
            self._compute_mask()
            if self.mask_ready:
                self._ut()

        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev)
            return 0

        si=self._compute_sims(x)

        # Standard class vote -> argmin (novelty-seeking)
        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,self.nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        action=sc[:self.nc].argmin().item()
        p=action; tm=(self.labels==p)

        # Track winner for logging
        w_i=int(si.argmax().item())
        self.unique_winners.add(w_i)

        # Spawn/attract
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)])
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            alpha=1.0/(1.0+max(float(si[w_i].item()),0.0))
            self.V[w_i]=self.V[w_i]+alpha*(x-self.V[w_i])

        return action


def main():
    t0=time.time()
    print(f"Step 408b -- Mask + centered unnorm + ARGMIN. 64x64. 50K. LS20.", flush=True)
    print(f"Device: {DEVICE}  mask_warmup={MASK_WARMUP}  cb_cap={CB_CAP}", flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    fold=CombinedFold(d=D, nc=na, k=3)
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
            fold._update_stats(raw)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if fold.V.shape[0]>=na: sd=True; fold._ut()
            continue
        if not sd: sd=True

        c=fold.process(raw)

        # Log mask activation
        if fold.mask_ready and not mask_logged:
            mask_logged=True
            mask_np=fold.mask.cpu().numpy()
            active_idx=np.where(mask_np>0)[0]
            rows=sorted(set(active_idx//64))
            print(f"    MASK ACTIVATED at step {ts}: n_active={fold.n_active}", flush=True)
            print(f"      Active rows: {rows}", flush=True)

        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]}"
                  f"  unique_winners={len(fold.unique_winners)} go={go}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            n_uw=len(fold.unique_winners)
            mode='MASKED' if fold.mask_ready else 'RAW'
            print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.2f}"
                  f"  mode={mode}  n_active={fold.n_active}"
                  f"  unique_winners={n_uw}  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 408b SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"cb={fold.V.shape[0]}  thresh={fold.thresh:.2f}  mask_ready={fold.mask_ready}"
          f"  n_active={fold.n_active}", flush=True)
    print(f"unique_winners={len(fold.unique_winners)}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0

    active_winners=sum(1 for c in fold.win_action_counts.values() if sum(c)>5)
    print(f"active_winners(>5 visits)={active_winners}", flush=True)

    if lvls>0:
        print(f"\nPASS: Level with combined approach at 64x64!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% dominance.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
