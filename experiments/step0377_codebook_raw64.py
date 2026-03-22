#!/usr/bin/env python3
"""
Step 377 -- Raw 64x64 frames, no prescribed encoding. LS20.

frame.flatten() / 15.0 -> 4096 dims. F.normalize. That's it.
Can process() bootstrap its own representation from raw pixels?

2K steps. Kill: does codebook grow beyond 1?
Script: scripts/run_step377_raw64.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096

CB_CAP = 10000
THRESH_INTERVAL = 100

class CF:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0
    def _fa(self,x,l):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INTERVAL==0: self._ut()
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
            if self.spawn_count%THRESH_INTERVAL==0: self._ut()
        else:
            ts=si.clone(); ts[~tm]=-float('inf'); w=ts.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return p


def main():
    t0=time.time()
    print(f"Step 377 -- Raw 64x64 on LS20. {D} dims. No prescribed encoding.",flush=True)
    print(f"Device: {DEVICE}",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=CF(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    unique_states=set(); action_counts={}
    sims_log=[]  # track consecutive frame sims

    max_steps=2000
    prev_enc=None

    while ts<max_steps and go<50:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset(); prev_enc=None
            if obs is None: break; continue
        if obs.state==GameState.WIN: break

        frame=np.array(obs.frame[0],dtype=np.float32)/15.0
        raw=torch.from_numpy(frame.flatten())
        enc=F.normalize(raw,dim=0)

        unique_states.add(hash(frame.tobytes()))

        # Track sim to previous frame
        if prev_enc is not None:
            sim=float((enc@prev_enc).item())
            sims_log.append(sim)
        prev_enc=enc.clone()

        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(enc,i)
            action=env.action_space[i]; obs=env.step(action); ts+=1
            action_counts[action.name]=action_counts.get(action.name,0)+1
            if fold.V.shape[0]>=na:
                sd=True; fold._ut()
                print(f"    [seed done, step {ts}] cb={fold.V.shape[0]} thresh={fold.thresh:.6f}",flush=True)
            continue
        if not sd: sd=True

        c=fold.pn(enc,nc=na)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        obs=env.step(action); ts+=1
        if obs is None: break

        if ts%500==0:
            avg_sim=np.mean(sims_log[-100:]) if sims_log else 0
            print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.6f}"
                  f"  unique={len(unique_states)} go={go}"
                  f"  avg_consecutive_sim={avg_sim:.6f}",flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 377 SUMMARY",flush=True); print("="*60,flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}",flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.6f}",flush=True)
    print(f"unique_states={len(unique_states)}",flush=True)
    print(f"action_counts: {action_counts}",flush=True)
    print(flush=True)

    if sims_log:
        sl=np.array(sims_log)
        print(f"Consecutive frame similarity:",flush=True)
        print(f"  min={sl.min():.6f}  max={sl.max():.6f}  mean={sl.mean():.6f}  std={sl.std():.6f}",flush=True)
    print(flush=True)

    if fold.V.shape[0]>10:
        print(f"PASS: codebook grew to {fold.V.shape[0]} entries from raw pixels.",flush=True)
    elif fold.V.shape[0]>1:
        print(f"MARGINAL: codebook={fold.V.shape[0]} (small growth).",flush=True)
    else:
        print(f"KILL: codebook stuck at {fold.V.shape[0]}. Raw 64x64 cos sim too high.",flush=True)

    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
