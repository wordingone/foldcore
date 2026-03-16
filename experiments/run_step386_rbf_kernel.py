#!/usr/bin/env python3
"""
Step 386 -- RBF kernel at raw 64x64. Nonlinear similarity amplification.

sims = exp(-(1 - cos_sims) / sigma_sq)
sigma_sq = state-derived (median of 1-cos for sampled codebook pairs).
Amplifies small cosine differences: [0.97,0.99] -> [0.15,0.54].

50K steps on LS20. Kill: RBF sim spread > 0.1 after 5K steps.
Script: scripts/run_step386_rbf_kernel.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100

class RBFFold:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev
        self.spawn_count=0; self.sigma_sq=0.01  # initial

    def _update_sigma(self):
        n=self.V.shape[0]
        if n<10: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        cos_sims=self.V[idx]@self.V.T  # (ss, n)
        # Exclude self-similarity
        cos_sims.fill_diagonal_(-1)
        # Get nearest-neighbor cos_sim per sample row
        nn_cos=cos_sims.max(dim=1).values
        # sigma_sq = median of (1 - nn_cos)
        distances=1.0-nn_cos
        self.sigma_sq=float(distances.median().clamp(min=1e-6))

    def _rbf(self, cos_sims):
        return torch.exp(-(1.0 - cos_sims) / max(self.sigma_sq, 1e-6))

    def _fa(self,x,l):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INT==0:
            self._update_sigma(); self._ut()

    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        cos_s=self.V[idx]@self.V.T
        rbf_s=self._rbf(cos_s)
        t=rbf_s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())

    def pn(self,x,nc):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0,0.0,0.0
        cos_sims=self.V@x
        sims=self._rbf(cos_sims)

        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=sims[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)
        nsim=float(sims.max().item())
        ncos=float(cos_sims.max().item())

        if (tm.sum()==0 or sims[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0:
                self._update_sigma(); self._ut()
        else:
            w_i=sims.argmax().item()
            a=1.0-float(sims[w_i].item())
            self.V[w_i]=F.normalize(self.V[w_i]+a*(x-self.V[w_i]),dim=0)
        return p,nsim,ncos


def main():
    t0=time.time()
    print(f"Step 386 -- RBF kernel at raw 64x64. LS20. 50K steps.",flush=True)
    print(f"Device: {DEVICE}  Cap={CB_CAP}",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=RBFFold(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    action_counts={}; rbf_log=[]; cos_log=[]

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
            if fold.V.shape[0]>=na: sd=True; fold._update_sigma(); fold._ut()
            continue
        if not sd: sd=True

        c,nsim,ncos=fold.pn(raw,nc=na)
        rbf_log.append(nsim); cos_log.append(ncos)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} sigma_sq={fold.sigma_sq:.6f}",flush=True)

        if ts%5000==0:
            rb=np.array(rbf_log[-500:]); co=np.array(cos_log[-500:])
            print(f"    [step {ts:6d}] cb={fold.V.shape[0]} thresh={fold.thresh:.4f}"
                  f"  sigma_sq={fold.sigma_sq:.6f}"
                  f"  rbf: mean={rb.mean():.4f} std={rb.std():.4f}"
                  f"  cos: mean={co.mean():.4f} std={co.std():.4f}"
                  f"  levels={lvls} go={go}",flush=True)
            print(f"      acts: {action_counts}",flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 386 SUMMARY",flush=True); print("="*60,flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}",flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.4f}  sigma_sq={fold.sigma_sq:.6f}",flush=True)
    print(f"action_counts: {action_counts}",flush=True)
    if rbf_log:
        rb=np.array(rbf_log); co=np.array(cos_log)
        print(f"\nRBF sim: min={rb.min():.4f} max={rb.max():.4f} mean={rb.mean():.4f} std={rb.std():.4f}",flush=True)
        print(f"Cos sim: min={co.min():.4f} max={co.max():.4f} mean={co.mean():.4f} std={co.std():.4f}",flush=True)
        rbf_spread=rb.max()-rb.min()
        cos_spread=co.max()-co.min()
        print(f"Amplification: cos spread={cos_spread:.4f} -> rbf spread={rbf_spread:.4f} ({rbf_spread/max(cos_spread,1e-6):.1f}x)",flush=True)
    if lvls>0:
        print("\nPASS: Level from raw 64x64 with RBF kernel!",flush=True)
    elif rbf_log and np.std(rbf_log[-500:])>0.05:
        print("\nMARGINAL: RBF increased spread but no level.",flush=True)
    else:
        print("\nKILL.",flush=True)
    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
