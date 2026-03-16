#!/usr/bin/env python3
"""
Step 387 -- Centered UNNORMALIZED dot product at raw 64x64.

Center by codebook mean. NO F.normalize. Raw dot product preserves
magnitude (sprite signal). Min-max rescale sims to [0,1].

50K steps on LS20. Kill: rescaled sim spread < 0.1 after 5K.
Script: scripts/run_step387_centered_unnorm.py
"""

import time, logging, numpy as np, torch
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100; MEAN_BOOTSTRAP = 200

class UnnormFold:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.99; self.k=k; self.d=d; self.dev=dev
        self.spawn_count=0; self.mean=None; self.phase2=False

    def _update_mean(self):
        if self.V.shape[0]>=MEAN_BOOTSTRAP:
            self.mean=self.V.mean(dim=0)
            if not self.phase2:
                self.phase2=True

    def _center(self, x):
        if self.mean is None: return x
        return x - self.mean

    def _sims(self, x):
        """Centered dot product, min-max rescaled to [0,1]."""
        xc=self._center(x)
        Vc=self.V-self.mean.unsqueeze(0) if self.mean is not None else self.V
        raw=Vc@xc
        mn=raw.min(); mx=raw.max()
        rng=mx-mn
        if rng<1e-8: return torch.ones_like(raw)*0.5, raw
        return (raw-mn)/rng, raw

    def _fa(self,x,l):
        self.V=torch.cat([self.V,x.to(self.dev).float().unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INT==0:
            self._update_mean(); self._ut()
        elif self.spawn_count==MEAN_BOOTSTRAP:
            self._update_mean(); self._ut()

    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        if self.mean is None:
            # Phase 1: cosine thresh
            import torch.nn.functional as Fn
            Vn=Fn.normalize(self.V,dim=1)
            ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
            s=Vn[idx]@Vn.T; t=s.topk(min(2,n),dim=1).values
            self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())
            return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        Vc=self.V-self.mean.unsqueeze(0)
        raw=Vc[idx]@Vc.T  # (ss, n)
        # Per-row rescale then find nearest neighbor
        mn=raw.min(dim=1,keepdim=True).values
        mx=raw.max(dim=1,keepdim=True).values
        rng=mx-mn; rng=rng.clamp(min=1e-8)
        scaled=(raw-mn)/rng
        # Exclude self (set to 0)
        for i,j in enumerate(idx):
            scaled[i,j]=0.0
        nn=scaled.max(dim=1).values
        self.thresh=float(nn.median())

    def pn(self,x,nc):
        x=x.to(self.dev).float()
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0,0.0,0.0

        if not self.phase2:
            # Phase 1: cosine similarity (normalized) to build codebook
            import torch.nn.functional as Fn
            xn=Fn.normalize(x,dim=0); Vn=Fn.normalize(self.V,dim=1)
            sims=Vn@xn
            nsim=float(sims.max().item()); raw_spread=0.0
        else:
            sims,raw_sims=self._sims(x)
            nsim=float(sims.max().item())
            raw_spread=float((raw_sims.max()-raw_sims.min()).item())

        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=sims[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)

        if (tm.sum()==0 or sims[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)])  # store RAW
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0:
                self._update_mean(); self._ut()
        else:
            w_i=sims.argmax().item()
            a=max(0.0,1.0-float(sims[w_i].item()))
            self.V[w_i]=self.V[w_i]+a*(x-self.V[w_i])  # attract in RAW space, no normalize
        return p,nsim,raw_spread


def main():
    t0=time.time()
    print(f"Step 387 -- Centered UNNORMALIZED dot product. Raw 64x64. 50K steps.",flush=True)
    print(f"Device: {DEVICE}  Cap={CB_CAP}  Mean bootstrap={MEAN_BOOTSTRAP}",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=UnnormFold(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    action_counts={}; sim_log=[]; spread_log=[]

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!",flush=True); break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)

        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(raw,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if fold.V.shape[0]>=na: sd=True
            continue
        if not sd: sd=True

        c,nsim,rspread=fold.pn(raw,nc=na)
        sim_log.append(nsim); spread_log.append(rspread)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} go={go}",flush=True)

        if ts%5000==0:
            sl=np.array(sim_log[-500:]); sp=np.array(spread_log[-500:])
            print(f"    [step {ts:6d}] cb={fold.V.shape[0]} thresh={fold.thresh:.4f}"
                  f"  phase={'CENTERED' if fold.phase2 else 'RAW'}"
                  f"  sim: mean={sl.mean():.4f} std={sl.std():.4f}"
                  f"  raw_spread: mean={sp.mean():.2f}"
                  f"  levels={lvls} go={go}",flush=True)
            print(f"      acts: {action_counts}",flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 387 SUMMARY",flush=True); print("="*60,flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}",flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.4f}  phase2={fold.phase2}",flush=True)
    print(f"action_counts: {action_counts}",flush=True)
    if sim_log:
        sl=np.array(sim_log)
        print(f"Rescaled sim: min={sl.min():.4f} max={sl.max():.4f} mean={sl.mean():.4f} std={sl.std():.4f}",flush=True)
    if spread_log:
        sp=np.array(spread_log)
        print(f"Raw dot spread: min={sp.min():.2f} max={sp.max():.2f} mean={sp.mean():.2f}",flush=True)
    if lvls>0:
        print("\nPASS: Level from raw 64x64 with unnormalized centered dot product!",flush=True)
    elif sim_log and np.std(sim_log[-500:])>0.05:
        print("\nMARGINAL: spread achieved but no level.",flush=True)
    else:
        print("\nKILL.",flush=True)
    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
