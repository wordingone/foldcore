#!/usr/bin/env python3
"""
Step 396 -- Multi-resolution voting. All 4 resolutions simultaneously.

Maintain 4 codebooks (64,32,16,8). Each frame processed through all 4.
Action selected from resolution with highest class vote variance
(max_score - min_score). All codebooks updated every step.

LS20. 50K steps.
Script: scripts/run_step396_multires_vote.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CB_CAP = 10000; THRESH_INT = 100

class CF:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0
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
    def process(self,x,nc):
        """Returns (class_scores, action_idx). Updates codebook."""
        x=F.normalize(x.to(self.dev).float(),dim=0)
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev)
            sc=torch.zeros(nc,device=self.dev); sc[0]=1.0
            return sc, 0

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
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts=si.clone(); ts[~tm]=-float('inf'); w=ts.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return sc[:nc], p

def avgpool(frame, res):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    if res == 64: return arr.flatten()
    k = 64 // res
    return arr.reshape(res, k, res, k).mean(axis=(1, 3)).flatten()

def main():
    t0=time.time()
    print("Step 396 -- Multi-resolution voting. LS20. 50K.", flush=True)
    print(f"Device: {DEVICE}", flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    resolutions=[64,32,16,8]
    folds={res: CF(d=res*res, k=3) for res in resolutions}
    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0
    # Seed all codebooks
    seeded={res:False for res in resolutions}

    action_counts={}
    res_wins={res:0 for res in resolutions}  # how often each res drove action
    res_variances={res:[] for res in resolutions}

    print(f"Resolutions: {resolutions}  na={na}", flush=True); print(flush=True)

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break
            continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!", flush=True); break

        # Encode at all resolutions
        encs={}
        for res in resolutions:
            encs[res]=torch.from_numpy(avgpool(obs.frame, res).astype(np.float32))

        # Seed phase: seed all codebooks that need it
        all_seeded=all(seeded[r] for r in resolutions)
        if not all_seeded:
            for res in resolutions:
                fold=folds[res]
                if not seeded[res] and fold.V.shape[0]<na:
                    i=fold.V.shape[0]; fold._fa(encs[res],i)
                    if fold.V.shape[0]>=na:
                        seeded[res]=True; fold._ut()
            # Re-check: seeding loop may have completed all
            unseeded=[r for r in resolutions if not seeded[r]]
            if unseeded:
                action_i=min(na-1, min(folds[r].V.shape[0] for r in unseeded))
                action=env.action_space[action_i]
                action_counts[action.name]=action_counts.get(action.name,0)+1
                obs=env.step(action); ts+=1
            continue

        # Process through all codebooks
        scores={}; variances={}
        for res in resolutions:
            sc, _ = folds[res].process(encs[res], nc=na)
            scores[res]=sc
            v=float(sc.max().item()-sc.min().item())
            variances[res]=v
            res_variances[res].append(v)

        # Select action from most confident resolution
        best_res=max(resolutions, key=lambda r: variances[r])
        res_wins[best_res]+=1
        action_idx=scores[best_res][:na].argmin().item()  # novelty-seeking
        action=env.action_space[action_idx]
        action_counts[action.name]=action_counts.get(action.name,0)+1

        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} best_res={best_res}x{best_res} go={go}", flush=True)
            for r in resolutions:
                print(f"      {r:2d}x{r:2d}: cb={folds[r].V.shape[0]} wins={res_wins[r]}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            total_wins=sum(res_wins.values())
            print(f"    [step {ts:5d}] levels={lvls} dom={dom:.0f}% go={go}", flush=True)
            for r in resolutions:
                rv=res_variances[r][-500:] if res_variances[r] else [0]
                pct=res_wins[r]/max(total_wins,1)*100
                print(f"      {r:2d}x{r:2d}: wins={pct:4.0f}%  cb={folds[r].V.shape[0]}"
                      f"  thresh={folds[r].thresh:.3f}"
                      f"  var_mean={np.mean(rv):.3f}  var_std={np.std(rv):.3f}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 396 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    total_wins=sum(res_wins.values())
    for r in resolutions:
        pct=res_wins[r]/max(total_wins,1)*100
        rv=res_variances[r] if res_variances[r] else [0]
        print(f"  {r:2d}x{r:2d}: wins={pct:4.0f}% ({res_wins[r]}/{total_wins})"
              f"  cb={folds[r].V.shape[0]}  thresh={folds[r].thresh:.3f}"
              f"  var_mean={np.mean(rv):.4f}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    print(f"dom={dom:.0f}%", flush=True)
    if lvls>0:
        print(f"\nPASS: Level with multi-resolution voting!", flush=True)
    elif res_wins.get(16,0)>res_wins.get(64,0) and res_wins.get(16,0)>res_wins.get(8,0):
        print(f"\nMARGINAL: 16x16 drove most actions but no level (stochastic).", flush=True)
    else:
        top=max(resolutions, key=lambda r: res_wins[r])
        print(f"\nKILL: {top}x{top} drove most actions.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
