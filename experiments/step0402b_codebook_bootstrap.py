#!/usr/bin/env python3
"""
Step 402b -- Mask + group + force-spawn 200. Bootstrap the codebook.

Same encoding as 402 (62 super-dims from grouped active pixels).
Force-spawn first 200 encoded observations after warmup.
Then normal process() with 200-entry codebook in 62D.

LS20. 64x64. 50K steps.
Script: scripts/run_step402b_bootstrap.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100; WARMUP = 200; EPS = 0.01
K_GROUP = 4  # group size for adjacent active dims
BOOTSTRAP = 200  # force-spawn this many after encoding ready

class GroupedFold:
    def __init__(self, k=3, dev=DEVICE):
        self.dev=dev; self.k=k
        # Change-rate tracking (on raw 4096 dims)
        self.x_prev=None; self.change_count=torch.zeros(D,device=dev)
        self.frame_count=0
        # Mask and groups
        self.active_indices=None; self.groups=None; self.n_groups=0
        self.encoding_ready=False
        # Codebook (on grouped encoding)
        self.V=torch.zeros(0,0,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.spawn_count=0

    def _update_stats(self, x):
        x=x.to(self.dev).float()
        if self.x_prev is not None:
            changed=(torch.abs(x-self.x_prev)>EPS).float()
            self.change_count+=changed
        self.frame_count+=1
        self.x_prev=x.clone()

    def _compute_encoding(self):
        rate=self.change_count/self.frame_count
        active=rate>0.001
        if active.sum()<10: return False
        r=rate[active]
        threshold=float(r.mean()+2*r.std())
        mask=(rate>threshold)
        self.active_indices=torch.where(mask)[0]
        n_active=len(self.active_indices)
        if n_active<K_GROUP: return False
        self.n_groups=n_active//K_GROUP
        self.groups=self.active_indices[:self.n_groups*K_GROUP].reshape(self.n_groups,K_GROUP)
        self.encoding_ready=True
        # Reset codebook for new encoding dimension
        self.V=torch.zeros(0,self.n_groups,device=self.dev)
        self.labels=torch.zeros(0,dtype=torch.long,device=self.dev)
        self.thresh=0.7; self.spawn_count=0
        return True

    def encode(self, x):
        """Extract active dims and pool within groups."""
        x=x.to(self.dev).float()
        vals=x[self.groups]  # (n_groups, K_GROUP)
        pooled=vals.mean(dim=1)  # (n_groups,)
        return F.normalize(pooled, dim=0)

    def _fa(self, enc, l):
        self.V=torch.cat([self.V, enc.unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INT==0: self._ut()

    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        s=self.V[idx]@self.V.T; t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())

    def pn(self, enc, nc):
        if self.V.shape[0]==0:
            self.V=enc.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0

        si=self.V@enc; ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)

        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V, enc.unsqueeze(0)])
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts_=si.clone(); ts_[~tm]=-float('inf'); w_i=ts_.argmax().item()
            a=1.0-float(si[w_i].item())
            self.V[w_i]=F.normalize(self.V[w_i]+a*(enc-self.V[w_i]),dim=0)
        return p


def main():
    t0=time.time()
    print(f"Step 402b -- Mask + group + bootstrap {BOOTSTRAP}. 64x64. 50K. LS20.", flush=True)
    print(f"Device: {DEVICE}  warmup={WARMUP}  K_group={K_GROUP}  bootstrap={BOOTSTRAP}", flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=GroupedFold(k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False; bootstrap_count=0; bootstrap_done=False
    action_counts={}; import random

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break
            continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!", flush=True); break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)

        # Phase 1: warmup (random actions, collect change stats)
        if not fold.encoding_ready:
            fold._update_stats(raw)
            if fold.frame_count>=WARMUP:
                ok=fold._compute_encoding()
                if ok:
                    print(f"    Encoding ready at step {ts}: n_groups={fold.n_groups}"
                          f"  n_active={len(fold.active_indices)}  enc_dim={fold.n_groups}", flush=True)
                    active_np=fold.active_indices.cpu().numpy()
                    rows=sorted(set(active_np//64))
                    print(f"    Active rows: {rows}", flush=True)
                else:
                    print(f"    Encoding failed at step {ts}. Extending warmup.", flush=True)
            action=env.action_space[random.randint(0,na-1)]
            action_counts[action.name]=action_counts.get(action.name,0)+1
            obs=env.step(action); ts+=1
            continue

        # Phase 2: encoded process
        fold._update_stats(raw)
        enc=fold.encode(raw)

        # Bootstrap: force-spawn first BOOTSTRAP observations
        if not bootstrap_done:
            label=bootstrap_count%na
            fold._fa(enc, label)
            bootstrap_count+=1
            action=env.action_space[random.randint(0,na-1)]
            action_counts[action.name]=action_counts.get(action.name,0)+1
            obs=env.step(action); ts+=1
            if bootstrap_count>=BOOTSTRAP:
                bootstrap_done=True; sd=True; fold._ut()
                print(f"    Bootstrap done at step {ts}: cb={fold.V.shape[0]}"
                      f"  thresh={fold.thresh:.3f}", flush=True)
            continue

        c=fold.pn(enc, nc=na)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} go={go}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.3f}"
                  f"  enc_dim={fold.n_groups}  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 402b SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"cb={fold.V.shape[0]}  thresh={fold.thresh:.3f}  enc_dim={fold.n_groups}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    if lvls>0:
        print(f"\nPASS: Level with self-derived grouped encoding at 64x64!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% dominance.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
