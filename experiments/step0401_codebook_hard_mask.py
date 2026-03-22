#!/usr/bin/env python3
"""
Step 401 -- Hard threshold on change rate. DELETE uninformative dims.

Same change-rate tracking as Step 400. Instead of soft weight, zero out
dims below mean+2*std threshold. Cosine on surviving ~150 dims.

LS20. 64x64. 50K steps.
Script: scripts/run_step401_hard_mask.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100; WARMUP = 100; EPS = 0.01
MASK_UPDATE = 1000

class MaskedFold:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0
        self.x_prev=None; self.change_count=torch.zeros(d,device=dev)
        self.frame_count=0; self.mask=None; self.n_active=0

    def _update_stats(self, x):
        x=x.to(self.dev).float()
        if self.x_prev is not None:
            changed=(torch.abs(x-self.x_prev)>EPS).float()
            self.change_count+=changed
        self.frame_count+=1
        self.x_prev=x.clone()

    def _update_mask(self):
        if self.frame_count<WARMUP: return
        rate=self.change_count/self.frame_count
        active=rate>0.001
        if active.sum()<10: return
        r=rate[active]
        threshold=float(r.mean()+2*r.std())
        self.mask=(rate>threshold).float()
        self.n_active=int(self.mask.sum().item())

    def _masked_sim(self, x):
        """Compute similarity with mask applied."""
        if self.mask is not None and self.n_active>5:
            xm=F.normalize(x*self.mask, dim=0)
            Vm=F.normalize(self.V*self.mask.unsqueeze(0), dim=1)
            return Vm@xm
        else:
            return F.normalize(self.V,dim=1)@F.normalize(x,dim=0)

    def _fa(self,x,l):
        x=x.to(self.dev).float()
        self._update_stats(x)
        self.V=torch.cat([self.V,x.unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INT==0: self._ut()
        if self.frame_count==WARMUP: self._update_mask()

    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        if self.mask is not None and self.n_active>5:
            Vm=F.normalize(self.V*self.mask.unsqueeze(0), dim=1)
            s=Vm[idx]@Vm.T
        else:
            Vn=F.normalize(self.V,dim=1)
            s=Vn[idx]@Vn.T
        t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())

    def pn(self,x,nc):
        x=x.to(self.dev).float()
        self._update_stats(x)
        if self.frame_count%MASK_UPDATE==0: self._update_mask(); self._ut()

        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0

        si=self._masked_sim(x)
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
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts_=si.clone(); ts_[~tm]=-float('inf'); w_i=ts_.argmax().item()
            a=1.0-float(si[w_i].item())
            self.V[w_i]=self.V[w_i]+a*(x-self.V[w_i])
        return p


def main():
    t0=time.time()
    print(f"Step 401 -- Hard mask on change rate. 64x64. 50K. LS20.", flush=True)
    print(f"Device: {DEVICE}  warmup={WARMUP}  eps={EPS}  mask_update={MASK_UPDATE}", flush=True)
    print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=MaskedFold(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    action_counts={}

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

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} n_active={fold.n_active} go={go}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.3f}"
                  f"  n_active={fold.n_active}  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)
            if fold.mask is not None:
                rate=(fold.change_count/fold.frame_count).cpu().numpy()
                mask_np=fold.mask.cpu().numpy()
                active_rates=rate[mask_np>0]
                if len(active_rates)>0:
                    # Find which rows of 64x64 grid the active dims are in
                    active_idx=np.where(mask_np>0)[0]
                    rows=active_idx//64
                    cols=active_idx%64
                    print(f"      active rows: {sorted(set(rows))[:20]}{'...' if len(set(rows))>20 else ''}", flush=True)
                    print(f"      active rate: min={active_rates.min():.4f} max={active_rates.max():.4f}"
                          f"  mean={active_rates.mean():.4f}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 401 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"cb={fold.V.shape[0]}  thresh={fold.thresh:.3f}  n_active={fold.n_active}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    if fold.mask is not None:
        rate=(fold.change_count/fold.frame_count).cpu().numpy()
        mask_np=fold.mask.cpu().numpy()
        print(f"rate dist: min={rate.min():.4f} max={rate.max():.4f} mean={rate.mean():.4f} std={rate.std():.4f}", flush=True)
        r_active=rate[mask_np>0]
        if len(r_active)>0:
            active_idx=np.where(mask_np>0)[0]
            rows=sorted(set(active_idx//64))
            print(f"active dims: {fold.n_active}  rows: {rows}", flush=True)
            print(f"active rates: min={r_active.min():.4f} max={r_active.max():.4f} mean={r_active.mean():.4f}", flush=True)
    if lvls>0:
        print(f"\nPASS: Level with hard-masked cosine at 64x64!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% dominance or n_active={fold.n_active} out of range.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
