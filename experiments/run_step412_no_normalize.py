#!/usr/bin/env python3
"""
Step 412 -- Delete F.normalize from 16x16 baseline. The compression test.

Same as Step 353 but no F.normalize on x or V. Raw dot product.
Does the 20-line substrate work at 16x16?

LS20. 16x16 avgpool. 50K x 3 seeds.
Script: scripts/run_step412_no_normalize.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 256; CB_CAP = 10000; THRESH_INT = 100

class RawDotFold:
    """Same as Step 353 but NO F.normalize anywhere."""
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0
    def _fa(self,x,l):
        # NO normalize
        self.V=torch.cat([self.V,x.to(self.dev).float().unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INT==0: self._ut()
    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        # NO normalize — raw dot product
        s=self.V[idx]@self.V.T
        t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())
    def pn(self,x,nc):
        # NO normalize
        x=x.to(self.dev).float()
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0
        si=self.V@x
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
            w_i=int(si.argmax().item())
            # NO normalize on attract
            alpha=1.0/(1.0+max(float(si[w_i].item()),0.0))
            self.V[w_i]=self.V[w_i]+alpha*(x-self.V[w_i])
        return p

def avgpool16(frame):
    arr=np.array(frame[0],dtype=np.float32)/15.0
    return arr.reshape(16,4,16,4).mean(axis=(1,3)).flatten()

def run_seed(arc, game_id, seed, max_steps=50000):
    from arcengine import GameState
    import random; random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    env=arc.make(game_id); obs=env.reset()
    na=len(env.action_space)
    fold=RawDotFold(d=D,k=3)
    ts=0; go=0; lvls=0; sd=False; first_level=None
    action_counts={}
    while ts<max_steps and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break; continue
        if obs.state==GameState.WIN: break
        x=torch.from_numpy(avgpool16(obs.frame).astype(np.float32))
        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(x,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if fold.V.shape[0]>=na: sd=True; fold._ut()
            continue
        if not sd: sd=True
        c=fold.pn(x,nc=na)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break
        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            if first_level is None: first_level=ts
            print(f"    [seed {seed}] LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} go={go}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    return {'seed':seed,'levels':lvls,'steps':ts,'go':go,'first_level':first_level,
            'cb':fold.V.shape[0],'thresh':fold.thresh,'dom':dom,'actions':action_counts}

def main():
    t0=time.time()
    print("Step 412 -- No F.normalize at 16x16. LS20. 50K x 3 seeds.", flush=True)
    print(f"Device: {DEVICE}  cb_cap={CB_CAP}", flush=True)
    print(f"Baseline: Step 353 = 60% level completion at 50K", flush=True); print(flush=True)

    import arc_agi
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    results=[]
    for seed in [42, 123, 777]:
        print(f"--- Seed {seed} ---", flush=True)
        r=run_seed(arc, ls20.game_id, seed)
        results.append(r)
        fl=r['first_level'] if r['first_level'] else 'none'
        print(f"    levels={r['levels']}  first_level={fl}  cb={r['cb']}"
              f"  thresh={r['thresh']:.3f}  dom={r['dom']:.0f}%", flush=True); print(flush=True)

    elapsed=time.time()-t0
    print("="*60, flush=True)
    print("STEP 412 SUMMARY", flush=True); print("="*60, flush=True)
    lc=sum(1 for r in results if r['levels']>0)
    print(f"Level completion: {lc}/3 ({lc/3*100:.0f}%)", flush=True)
    print(f"Baseline (Step 353 normalized): 3/5 (60%)", flush=True); print(flush=True)
    for r in results:
        fl=r['first_level'] if r['first_level'] else 'none'
        print(f"  seed {r['seed']}: levels={r['levels']}  first_level={fl}"
              f"  cb={r['cb']}  thresh={r['thresh']:.3f}  dom={r['dom']:.0f}%", flush=True)
    print(flush=True)
    if lc>=2:
        print("PASS: Raw dot works at 16x16!", flush=True)
    elif lc>0:
        print("MARGINAL: Some levels but not better than baseline.", flush=True)
    else:
        print("KILL: F.normalize was necessary at 16x16.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
