#!/usr/bin/env python3
"""
Step 399 -- Two-codebook at 16x16. Does meta-encoding improve over baseline?

V_raw: 16x16 avgpool (256d), normalized cosine. Same as Step 353.
V_meta: 4D normalized cosine on class vote vectors.
Baseline: Step 353 single-codebook = 60% level completion at 50K.

LS20. 50K steps x 3 seeds.
Script: scripts/run_step399_twocb_16x16.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CB_CAP_RAW = 10000; CB_CAP_META = 1000; THRESH_INT = 100

class CF:
    """Standard normalized cosine codebook."""
    def __init__(self, d, k=3, cap=10000, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0; self.cap=cap
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
    def process_votes(self, x, nc):
        """Returns class vote vector AND updates codebook."""
        x=F.normalize(x.to(self.dev).float(),dim=0)
        sc=torch.zeros(nc,device=self.dev)
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev)
            return sc
        si=self.V@x; ac=int(self.labels.max().item())+1
        sc_full=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc_full[c]=cs.topk(min(self.k,len(cs))).values.sum()
        sc=sc_full[:nc]
        p=sc.argmin().item(); tm=(self.labels==p)
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<self.cap:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts_=si.clone(); ts_[~tm]=-float('inf'); w=ts_.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return sc
    def process(self, x, nc):
        """Standard process. Returns action index."""
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
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<self.cap:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts_=si.clone(); ts_[~tm]=-float('inf'); w=ts_.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return p

def avgpool16(frame):
    arr=np.array(frame[0],dtype=np.float32)/15.0
    return arr.reshape(16,4,16,4).mean(axis=(1,3)).flatten()

def run_seed(arc, game_id, seed, max_steps=50000):
    from arcengine import GameState
    import random; random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

    env=arc.make(game_id); obs=env.reset()
    na=len(env.action_space)
    raw=CF(d=256, k=3, cap=CB_CAP_RAW)
    meta=CF(d=na, k=3, cap=CB_CAP_META)

    ts=0; go=0; lvls=0; first_level_step=None
    raw_seeded=False; meta_seeded=False
    action_counts={}; vote_log=[]

    while ts<max_steps and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break
            continue
        if obs.state==GameState.WIN:
            break

        x=torch.from_numpy(avgpool16(obs.frame).astype(np.float32))

        # Seed raw
        if not raw_seeded and raw.V.shape[0]<na:
            i=raw.V.shape[0]; raw._fa(x,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if raw.V.shape[0]>=na: raw_seeded=True; raw._ut()
            continue

        # Get class votes
        votes=raw.process_votes(x, nc=na)
        vote_log.append(votes.cpu().numpy().copy())

        # Seed meta
        if not meta_seeded and meta.V.shape[0]<na:
            i=meta.V.shape[0]; meta._fa(votes,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if meta.V.shape[0]>=na: meta_seeded=True; meta._ut()
            continue

        c=meta.process(votes, nc=na)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            if first_level_step is None: first_level_step=ts
            print(f"    [seed {seed}] LEVEL {lvls} at step {ts} raw_cb={raw.V.shape[0]}"
                  f"  meta_cb={meta.V.shape[0]} go={go}", flush=True)

    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    vl=np.array(vote_log) if vote_log else np.zeros((1,na))
    v_diff=vl.max(axis=1).mean()-vl.min(axis=1).mean() if len(vl)>1 else 0

    return {
        'seed':seed, 'levels':lvls, 'steps':ts, 'go':go,
        'first_level':first_level_step,
        'raw_cb':raw.V.shape[0], 'meta_cb':meta.V.shape[0],
        'raw_thresh':raw.thresh, 'meta_thresh':meta.thresh,
        'dom':dom, 'actions':action_counts,
        'vote_diff':v_diff
    }

def main():
    t0=time.time()
    print("Step 399 -- Two-codebook at 16x16. LS20. 50K x 3 seeds.", flush=True)
    print(f"Device: {DEVICE}  raw_cap={CB_CAP_RAW}  meta_cap={CB_CAP_META}", flush=True)
    print(f"Baseline: Step 353 = 60% at 50K, level in 84-113 steps", flush=True)
    print(flush=True)

    import arc_agi
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    results=[]
    for seed in [42, 123, 777]:
        print(f"--- Seed {seed} ---", flush=True)
        r=run_seed(arc, ls20.game_id, seed)
        results.append(r)
        print(f"    levels={r['levels']}  first_level={r['first_level']}"
              f"  raw_cb={r['raw_cb']}  meta_cb={r['meta_cb']}"
              f"  dom={r['dom']:.0f}%  vote_diff={r['vote_diff']:.3f}", flush=True)
        print(flush=True)

    elapsed=time.time()-t0
    print("="*60, flush=True)
    print("STEP 399 SUMMARY", flush=True); print("="*60, flush=True)
    level_count=sum(1 for r in results if r['levels']>0)
    print(f"Level completion: {level_count}/3 ({level_count/3*100:.0f}%)", flush=True)
    print(f"Baseline: 3/5 (60%)", flush=True); print(flush=True)
    for r in results:
        fl=r['first_level'] if r['first_level'] else 'none'
        print(f"  seed {r['seed']}: levels={r['levels']}  first_level={fl}"
              f"  raw_cb={r['raw_cb']}  meta_cb={r['meta_cb']}"
              f"  meta_thresh={r['meta_thresh']:.3f}"
              f"  dom={r['dom']:.0f}%  vote_diff={r['vote_diff']:.3f}", flush=True)
    print(flush=True)
    if level_count>=2:
        print("PASS: Two-codebook works at 16x16!", flush=True)
    elif level_count>0:
        print("MARGINAL: Some levels but not better than baseline.", flush=True)
    else:
        print("KILL: Meta-encoding hurt at 16x16.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
