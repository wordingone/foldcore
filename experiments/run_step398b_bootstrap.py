#!/usr/bin/env python3
"""
Step 398b -- Two-codebook with bootstrap. Force-spawn first 200 steps.

Raw codebook: force-spawn 200 entries, then centered unnormalized dot.
Meta codebook: 4D normalized cosine on class vote vectors.
Raw reads world -> class votes -> meta reads votes -> action.

LS20. 50K steps.
Script: scripts/run_step398b_bootstrap.py
"""

import time, random, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_RAW = 4096; CB_CAP_RAW = 10000; CB_CAP_META = 1000
THRESH_INT = 100; BOOTSTRAP = 200

class RawCodebook:
    """64x64 centered unnormalized dot with bootstrap."""
    def __init__(self, d, nc, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.99; self.k=k; self.d=d; self.nc=nc; self.dev=dev
        self.spawn_count=0; self.mean=None; self.bootstrapped=False

    def _update_mean(self):
        self.mean=self.V.mean(dim=0)

    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        if self.bootstrapped and self.mean is not None:
            Vc=self.V-self.mean.unsqueeze(0)
            s=Vc[idx]@Vc.T
        else:
            Vn=F.normalize(self.V,dim=1)
            s=Vn[idx]@Vn.T
        for i,j in enumerate(idx): s[i,j]=-1e9
        nn=s.max(dim=1).values
        self.thresh=float(nn.median())

    def force_spawn(self, x, label):
        """Bootstrap: unconditionally add entry."""
        self.V=torch.cat([self.V,x.to(self.dev).float().unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([label],device=self.dev)])
        self.spawn_count+=1

    def finish_bootstrap(self):
        """Called after bootstrap phase. Switch to centered dot."""
        self._update_mean()
        self.bootstrapped=True
        self._ut()

    def process_votes(self, x):
        """Process and return class vote vector. Updates codebook."""
        x=x.to(self.dev).float()
        sc=torch.zeros(self.nc,device=self.dev)

        if self.V.shape[0]==0:
            return sc

        if self.bootstrapped and self.mean is not None:
            xc=x-self.mean; Vc=self.V-self.mean.unsqueeze(0)
            si=Vc@xc
        else:
            xn=F.normalize(x,dim=0); Vn=F.normalize(self.V,dim=1)
            si=Vn@xn

        ac=int(self.labels.max().item())+1
        sc_full=torch.zeros(max(ac,self.nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc_full[c]=cs.topk(min(self.k,len(cs))).values.sum()
        sc=sc_full[:self.nc]

        # Spawn/attract
        p=sc.argmin().item(); tm=(self.labels==p)
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP_RAW:
            self.V=torch.cat([self.V,x.unsqueeze(0)])
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0:
                self._update_mean(); self._ut()
        else:
            w_i=si.argmax().item()
            alpha=1.0/(1.0+max(float(si[w_i].item()),0.0))
            self.V[w_i]=self.V[w_i]+alpha*(x-self.V[w_i])

        return sc


class MetaCodebook:
    """Standard normalized cosine on class vote vectors."""
    def __init__(self, d, nc, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.nc=nc; self.dev=dev; self.spawn_count=0

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

    def process(self,x):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0

        si=self.V@x; ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,self.nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:self.nc].argmin().item(); tm=(self.labels==p)

        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP_META:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts=si.clone(); ts[~tm]=-float('inf'); w=ts.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return p


def main():
    t0=time.time()
    print("Step 398b -- Two-codebook with bootstrap. LS20. 50K.", flush=True)
    print(f"Device: {DEVICE}  bootstrap={BOOTSTRAP}  raw_cap={CB_CAP_RAW}  meta_cap={CB_CAP_META}", flush=True)
    print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)
    raw=RawCodebook(d=D_RAW, nc=na, k=3)
    meta=MetaCodebook(d=na, nc=na, k=3)

    ts=0; go=0; lvls=0
    action_counts={}; vote_log=[]
    bootstrap_done=False

    print(f"na={na}  meta_d={na}", flush=True); print(flush=True)

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break
            continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!", flush=True); break

        x_raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)

        # Bootstrap phase: force-spawn into raw, random actions
        if not bootstrap_done:
            label=ts%na  # round-robin labels
            raw.force_spawn(x_raw, label)
            action=env.action_space[random.randint(0,na-1)]
            action_counts[action.name]=action_counts.get(action.name,0)+1
            obs=env.step(action); ts+=1
            if ts>=BOOTSTRAP:
                raw.finish_bootstrap()
                bootstrap_done=True
                print(f"    Bootstrap done at step {ts}: raw_cb={raw.V.shape[0]}"
                      f"  thresh={raw.thresh:.2f}  mean_norm={raw.mean.norm():.2f}", flush=True)
            continue

        # Get class votes from raw codebook
        votes=raw.process_votes(x_raw)
        vote_log.append(votes.cpu().numpy().copy())

        # Seed meta codebook
        if meta.V.shape[0]<na:
            i=meta.V.shape[0]; meta._fa(votes,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if meta.V.shape[0]>=na: meta._ut()
            continue

        # Meta codebook selects action
        c=meta.process(votes)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} raw_cb={raw.V.shape[0]}"
                  f"  meta_cb={meta.V.shape[0]} go={go}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            vl=np.array(vote_log[-500:]) if vote_log else np.zeros((1,na))
            v_std=vl.std(axis=0) if len(vl)>1 else np.zeros(na)
            v_mean=vl.mean(axis=0)
            print(f"    [step {ts:5d}] raw_cb={raw.V.shape[0]} meta_cb={meta.V.shape[0]}"
                  f"  raw_thresh={raw.thresh:.2f}  meta_thresh={meta.thresh:.3f}"
                  f"  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)
            print(f"      vote_mean=[{', '.join(f'{v:.1f}' for v in v_mean)}]"
                  f"  vote_std=[{', '.join(f'{v:.1f}' for v in v_std)}]", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 398b SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"raw: cb={raw.V.shape[0]}  thresh={raw.thresh:.2f}  bootstrapped={raw.bootstrapped}", flush=True)
    print(f"meta: cb={meta.V.shape[0]}  thresh={meta.thresh:.3f}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    if vote_log:
        vl=np.array(vote_log)
        print(f"vote stats: mean=[{', '.join(f'{v:.1f}' for v in vl.mean(axis=0))}]", flush=True)
        print(f"            std =[{', '.join(f'{v:.1f}' for v in vl.std(axis=0))}]", flush=True)
    if lvls>0:
        print(f"\nPASS: Level with two-codebook at 64x64!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% action dominance.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
