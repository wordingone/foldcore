#!/usr/bin/env python3
"""
Step 395 -- In-game resolution cascade. Start 64x64, drop when stuck.

Start at 64x64 with normalized cosine. When codebook growth stalls
(500 consecutive steps with no spawns), drop resolution (64→32→16→8).
Reset codebook at each drop. Stay if level transition detected.

LS20. 200K steps total.
Script: scripts/run_step395_cascade.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CB_CAP = 10000; THRESH_INT = 100; STALL_LIMIT = 500

class CF:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0
    def reset(self, d=None):
        if d is not None: self.d=d
        self.V=torch.zeros(0,self.d,device=self.dev)
        self.labels=torch.zeros(0,dtype=torch.long,device=self.dev)
        self.thresh=0.7; self.spawn_count=0
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
    def pn(self,x,nc):
        """Returns (action, spawned, max_sim)."""
        x=F.normalize(x.to(self.dev).float(),dim=0)
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0,True,0.0
        si=self.V@x; ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)
        max_sim=float(si.max().item())
        spawned=False
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1; spawned=True
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts=si.clone(); ts[~tm]=-float('inf'); w=ts.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return p,spawned,max_sim

def avgpool(frame, res):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    if res == 64: return arr.flatten()
    k = 64 // res
    return arr.reshape(res, k, res, k).mean(axis=(1, 3)).flatten()

def main():
    t0=time.time()
    print("Step 395 -- In-game resolution cascade. LS20. 200K.", flush=True)
    print(f"Device: {DEVICE}  stall_limit={STALL_LIMIT}", flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    resolutions=[64,32,16,8]
    res_idx=0; res=resolutions[res_idx]
    d=res*res; fold=CF(d=d,k=3)
    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    no_spawn_streak=0
    segment_start=0; segment_levels=0
    action_counts={}
    segments=[]  # log per resolution segment

    print(f"=== Starting at {res}x{res} (d={d}) ===", flush=True)

    while ts<200000 and go<2000:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break
            continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!", flush=True); break

        enc=torch.from_numpy(avgpool(obs.frame, res).astype(np.float32))

        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(enc,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if fold.V.shape[0]>=na: sd=True; fold._ut()
            continue
        if not sd: sd=True

        c,spawned,max_sim=fold.pn(enc,nc=na)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        # Track spawn stall
        if spawned:
            no_spawn_streak=0
        else:
            no_spawn_streak+=1

        # Level transition detection
        if obs.levels_completed>ol:
            lvls=obs.levels_completed; segment_levels+=1
            print(f"    LEVEL {lvls} at step {ts} res={res}x{res} cb={fold.V.shape[0]} go={go}", flush=True)

        # Check stall → drop resolution
        if no_spawn_streak>=STALL_LIMIT and segment_levels==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            seg={'res':res, 'start':segment_start, 'end':ts, 'steps':ts-segment_start,
                 'cb':fold.V.shape[0], 'levels':segment_levels, 'dom':dom,
                 'actions':dict(action_counts)}
            segments.append(seg)
            print(f"    STALL at {res}x{res}: {no_spawn_streak} steps no spawn. "
                  f"cb={fold.V.shape[0]} dom={dom:.0f}%", flush=True)

            res_idx+=1
            if res_idx>=len(resolutions):
                print(f"    CASCADE EXHAUSTED at step {ts}. All resolutions tried.", flush=True)
                break

            res=resolutions[res_idx]; d=res*res
            fold.reset(d=d); sd=False
            no_spawn_streak=0; segment_start=ts; segment_levels=0
            action_counts={}
            print(f"=== Dropping to {res}x{res} (d={d}) at step {ts} ===", flush=True)
            continue

        # Periodic logging
        if ts%10000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            print(f"    [step {ts:6d}] res={res}x{res} cb={fold.V.shape[0]} "
                  f"thresh={fold.thresh:.3f} stall={no_spawn_streak} "
                  f"dom={dom:.0f}% levels={lvls} go={go}", flush=True)

    # Final segment
    if action_counts:
        dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
        seg={'res':res, 'start':segment_start, 'end':ts, 'steps':ts-segment_start,
             'cb':fold.V.shape[0], 'levels':segment_levels, 'dom':dom,
             'actions':dict(action_counts)}
        segments.append(seg)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 395 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  total_levels={lvls}", flush=True)
    print(flush=True)
    for s in segments:
        marker=" <-- LEVEL!" if s['levels']>0 else ""
        print(f"  {s['res']:2d}x{s['res']:2d}: steps {s['start']:6d}-{s['end']:6d} "
              f"({s['steps']:6d} steps)  cb={s['cb']}  dom={s['dom']:.0f}%  "
              f"levels={s['levels']}{marker}", flush=True)
        print(f"         actions: {s['actions']}", flush=True)
    print(flush=True)
    if lvls>0:
        final_res=segments[-1]['res'] if segments else res
        print(f"PASS: Level found at self-discovered {final_res}x{final_res}!", flush=True)
    else:
        print("KILL.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
