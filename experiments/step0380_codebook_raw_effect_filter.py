#!/usr/bin/env python3
"""
Step 380 -- Raw 64x64 + effect filter. Only stamp when frame changes.

The codebook concentrates on state-changing frames (sprite region).
Static background never enters. Codebook becomes a signal-focused projection.
2K steps on LS20.
Script: scripts/run_step380_raw_effect_filter.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INTERVAL = 100

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
    def predict(self,x,nc):
        """Predict only — no spawn/attract."""
        x=F.normalize(x.to(self.dev).float(),dim=0)
        if self.V.shape[0]==0: return 0
        si=self.V@x; ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        return sc[:nc].argmin().item()
    def stamp(self,x,label):
        """Stamp (spawn or attract) with given label."""
        x=F.normalize(x.to(self.dev).float(),dim=0)
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([label],device=self.dev); return
        si=self.V@x; tm=(self.labels==label)
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([label],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INTERVAL==0: self._ut()
        elif tm.sum()>0:
            ts=si.clone(); ts[~tm]=-float('inf'); w=ts.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)


def raw_enc(frame):
    return torch.from_numpy(np.array(frame[0],dtype=np.float32).flatten()/15.0)


def main():
    t0=time.time()
    print(f"Step 380 -- Raw 64x64 + effect filter on LS20. {D} dims.",flush=True)
    print(f"Device: {DEVICE}",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=CF(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    unique_states=set(); action_counts={}
    stamped=0; filtered=0
    prev_enc=None

    max_steps=2000

    while ts<max_steps and go<50:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset(); prev_enc=None
            if obs is None: break; continue
        if obs.state==GameState.WIN: break

        enc=raw_enc(obs.frame)
        unique_states.add(hash(enc.numpy().tobytes()))

        # Seed phase
        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(enc,i)
            action=env.action_space[i]; prev_enc=enc.clone()
            obs=env.step(action); ts+=1
            action_counts[action.name]=action_counts.get(action.name,0)+1
            if fold.V.shape[0]>=na: sd=True; fold._ut()
            continue
        if not sd: sd=True

        # Predict
        cls=fold.predict(enc,nc=na)
        action=env.action_space[cls%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1

        # Step
        prev_enc_step=enc.clone()
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        # Effect filter: only stamp if frame changed
        new_enc=raw_enc(obs.frame)
        if not torch.allclose(prev_enc_step,new_enc,atol=0.01):
            fold.stamp(new_enc,label=cls)
            stamped+=1
        else:
            filtered+=1
        prev_enc=new_enc

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]}",flush=True)

        if ts%500==0:
            sr=stamped/(stamped+filtered) if (stamped+filtered)>0 else 0
            print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.6f}"
                  f"  unique={len(unique_states)} stamped={stamped} filtered={filtered}"
                  f"  stamp_rate={sr:.1%}",flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 380 SUMMARY",flush=True); print("="*60,flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}",flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.6f}",flush=True)
    print(f"unique_states={len(unique_states)}",flush=True)
    print(f"stamped={stamped}  filtered={filtered}  stamp_rate={stamped/(stamped+filtered):.1%}" if (stamped+filtered)>0 else "",flush=True)
    print(f"action_counts: {action_counts}",flush=True)
    print(flush=True)

    # Compare to Step 377
    print(f"Compare to Step 377 (raw, no filter):",flush=True)
    print(f"  377: cb=1736  unique=1521  (all frames stamped)",flush=True)
    print(f"  380: cb={fold.V.shape[0]}  unique={len(unique_states)}  (effect-filtered)",flush=True)
    if fold.V.shape[0]<1736 and len(unique_states)>0:
        ratio=len(unique_states)/max(fold.V.shape[0],1)
        print(f"  Discriminability: {ratio:.2f} unique_states/cb_entry (higher=better)",flush=True)
    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
