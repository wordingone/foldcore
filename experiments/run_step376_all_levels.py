#!/usr/bin/env python3
"""
Step 376 -- Codebook reset on level transition. ALL levels, ALL 3 games.

LS20: 500K steps, 4 actions, 16x16.
FT09: 100K steps, 69 click classes, 16x16.
VC33: 50K steps, 3 zones, timer+zone encoding.

Codebook resets to empty on each level_completed increase.
Script: scripts/run_step376_all_levels.py
"""

import time, logging, random, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLICK_GRID = [(gx*8+4, gy*8+4) for gy in range(8) for gx in range(8)]
NC = 64


CB_CAP = 10000
THRESH_INTERVAL = 100

class CF:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0
    def reset_codebook(self):
        self.V=torch.zeros(0,self.d,device=self.dev)
        self.labels=torch.zeros(0,dtype=torch.long,device=self.dev)
        self.thresh=0.7; self.spawn_count=0
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
        if tm.sum()==0 or si[tm].max()<self.thresh:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self._ut()
        else:
            ts=si.clone(); ts[~tm]=-float('inf'); w=ts.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return p


def ap16(f):
    a=np.array(f[0],dtype=np.float32)/15.0; return a.reshape(16,4,16,4).mean(axis=(1,3)).flatten()

def ce(p,fold):
    t=F.normalize(torch.from_numpy(p.astype(np.float32)),dim=0)
    if fold.V.shape[0]>2: t=t-fold.V[:,:256].mean(dim=0).cpu() if fold.d>=256 else t-fold.V.mean(dim=0).cpu()
    return t


def run_ls20(arc, game_id, max_steps):
    from arcengine import GameState
    fold=CF(d=256,k=3); env=arc.make(game_id); obs=env.reset()
    na=len(env.action_space); ts=0; go=0; lvls=0; sd=False; ls=0; evts=[]
    while ts<max_steps and go<5000:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset(); ls=ts; sd=False
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            print(f"    [LS20] WIN at step {ts}! levels={obs.levels_completed}",flush=True); break
        enc=ce(ap16(obs.frame),fold)
        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(enc,i); obs=env.step(env.action_space[i]); ts+=1
            if fold.V.shape[0]>=na: sd=True
            continue
        if not sd: sd=True
        c=fold.pn(enc,nc=na); ol=obs.levels_completed
        obs=env.step(env.action_space[c%na]); ts+=1
        if obs is None: break
        if obs.levels_completed>ol:
            lvls=obs.levels_completed; life_s=ts-ls
            evts.append({'level':lvls,'step':ts,'life':life_s,'go':go})
            ls=ts; fold.reset_codebook(); sd=False  # RESET
            print(f"    [LS20] LEVEL {lvls} at step {ts} (life={life_s}) go={go} -> RESET cb",flush=True)
        if ts%50000==0:
            print(f"    [LS20] step {ts:6d} cb={fold.V.shape[0]} lvls={lvls} go={go}",flush=True)
        if obs.state==GameState.WIN:
            print(f"    [LS20] WIN!",flush=True); break
    return {'game':'LS20','levels':lvls,'steps':ts,'go':go,'events':evts}


def run_ft09(arc, game_id, max_steps):
    from arcengine import GameState
    n_cls=NC+5; fold=CF(d=256,k=3); env=arc.make(game_id); obs=env.reset()
    as_=env.action_space; a6=next(a for a in as_ if a.is_complex())
    sa=[a for a in as_ if not a.is_complex()]
    ts=0; go=0; lvls=0; sd=False; ls=0; evts=[]
    while ts<max_steps and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset(); ls=ts; sd=False
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            print(f"    [FT09] WIN at step {ts}! levels={obs.levels_completed}",flush=True); break
        enc=ce(ap16(obs.frame),fold)
        if not sd and fold.V.shape[0]<n_cls:
            i=fold.V.shape[0]; fold._fa(enc,i)
            if i<NC: cx,cy=CLICK_GRID[i]; act,d=a6,{"x":cx,"y":cy}
            else: act,d=sa[(i-NC)%len(sa)],{}
            obs=env.step(act,data=d); ts+=1
            if fold.V.shape[0]>=n_cls: sd=True
            continue
        if not sd: sd=True
        c=fold.pn(enc,nc=n_cls); ol=obs.levels_completed
        if c<NC: cx,cy=CLICK_GRID[c]; act,d=a6,{"x":cx,"y":cy}
        else: act,d=sa[(c-NC)%len(sa)],{}
        obs=env.step(act,data=d); ts+=1
        if obs is None: break
        if obs.levels_completed>ol:
            lvls=obs.levels_completed; life_s=ts-ls
            evts.append({'level':lvls,'step':ts,'life':life_s,'go':go})
            ls=ts; fold.reset_codebook(); sd=False
            print(f"    [FT09] LEVEL {lvls} at step {ts} (life={life_s}) go={go} -> RESET cb",flush=True)
        if ts%10000==0:
            print(f"    [FT09] step {ts:6d} cb={fold.V.shape[0]} lvls={lvls} go={go}",flush=True)
        if obs.state==GameState.WIN:
            print(f"    [FT09] WIN!",flush=True); break
    return {'game':'FT09','levels':lvls,'steps':ts,'go':go,'events':evts}


def run_vc33(arc, game_id, max_steps, zone_reps, zone_frames_map):
    from arcengine import GameState
    n_zones=len(zone_reps); D=1+n_zones
    fold=CF(d=D,k=3); env=arc.make(game_id); obs=env.reset()
    as_=env.action_space
    ts=0; go=0; lvls=0; sd=False; ls=0; evts=[]; last_zone=0
    while ts<max_steps and go<2000:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset(); ls=ts; sd=False; last_zone=0
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            print(f"    [VC33] WIN at step {ts}! levels={obs.levels_completed}",flush=True); break
        frame=np.array(obs.frame[0]); timer_frac=(frame[0]==7).sum()/64.0
        enc=np.zeros(D,dtype=np.float32); enc[0]=timer_frac; enc[1+last_zone]=1.0
        enc_t=torch.from_numpy(enc)
        if not sd and fold.V.shape[0]<n_zones:
            i=fold.V.shape[0]; fold._fa(enc_t,i); zone=i
            cx,cy=zone_reps[zone]; obs=env.step(as_[0],data={"x":cx,"y":cy}); ts+=1
            if obs is not None:
                fh=hash(np.array(obs.frame[0]).tobytes()); last_zone=zone_frames_map.get(fh,0)
            if fold.V.shape[0]>=n_zones: sd=True
            continue
        if not sd: sd=True
        zone=fold.pn(enc_t,nc=n_zones); cx,cy=zone_reps[zone]
        ol=obs.levels_completed
        obs=env.step(as_[0],data={"x":cx,"y":cy}); ts+=1
        if obs is None: break
        if obs is not None:
            fh=hash(np.array(obs.frame[0]).tobytes()); last_zone=zone_frames_map.get(fh,0)
        if obs.levels_completed>ol:
            lvls=obs.levels_completed; life_s=ts-ls
            evts.append({'level':lvls,'step':ts,'life':life_s,'go':go})
            ls=ts; fold.reset_codebook(); sd=False
            print(f"    [VC33] LEVEL {lvls} at step {ts} (life={life_s}) go={go} -> RESET cb",flush=True)
        if ts%10000==0:
            print(f"    [VC33] step {ts:6d} cb={fold.V.shape[0]} lvls={lvls} go={go}",flush=True)
        if obs.state==GameState.WIN:
            print(f"    [VC33] WIN!",flush=True); break
    return {'game':'VC33','levels':lvls,'steps':ts,'go':go,'events':evts}


def map_vc33_zones(arc, vc33):
    from arcengine import GameState
    zone_frames={}; zone_reps={}
    for x in range(0,64,4):
        for y in range(0,64,4):
            env=arc.make(vc33.game_id); obs=env.reset()
            for _ in range(10):
                obs=env.step(env.action_space[0],data={"x":32,"y":32})
                if obs is None or obs.state!=GameState.NOT_FINISHED: break
            if obs is None or obs.state!=GameState.NOT_FINISHED: continue
            obs=env.step(env.action_space[0],data={"x":x,"y":y})
            if obs is None: continue
            fh=hash(np.array(obs.frame[0]).tobytes())
            if fh not in zone_frames:
                zid=len(zone_frames); zone_frames[fh]=zid; zone_reps[zid]=(x,y)
    return zone_reps, zone_frames


def main():
    t0=time.time()
    print("Step 376 -- ALL levels, ALL games, codebook reset on level transition",flush=True)
    print(f"Device: {DEVICE}",flush=True); print(flush=True)

    import arc_agi
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())
    ft09=next(g for g in games if 'ft09' in g.game_id.lower())
    vc33=next(g for g in games if 'vc33' in g.game_id.lower())

    # Map VC33 zones first
    print("Mapping VC33 zones...",flush=True)
    zone_reps, zone_frames = map_vc33_zones(arc, vc33)
    print(f"  {len(zone_reps)} zones found",flush=True); print(flush=True)

    results = []

    # FT09 first (fastest)
    print("=== FT09: 100K steps ===",flush=True)
    t1=time.time()
    r=run_ft09(arc, ft09.game_id, 100000)
    print(f"  Done: {time.time()-t1:.1f}s",flush=True); results.append(r); print(flush=True)

    # VC33 second
    print("=== VC33: 50K steps ===",flush=True)
    t1=time.time()
    r=run_vc33(arc, vc33.game_id, 50000, zone_reps, zone_frames)
    print(f"  Done: {time.time()-t1:.1f}s",flush=True); results.append(r); print(flush=True)

    # LS20 last (longest)
    print("=== LS20: 500K steps ===",flush=True)
    t1=time.time()
    r=run_ls20(arc, ls20.game_id, 500000)
    print(f"  Done: {time.time()-t1:.1f}s",flush=True); results.append(r); print(flush=True)

    elapsed=time.time()-t0
    print("="*60,flush=True); print("STEP 376 SUMMARY",flush=True); print("="*60,flush=True)
    for r in results:
        print(f"\n  {r['game']}: {r['levels']} levels in {r['steps']} steps ({r['go']} game_overs)",flush=True)
        for e in r['events']:
            print(f"    Level {e['level']}: step={e['step']} life={e['life']} go={e['go']}",flush=True)

    total_levels = sum(r['levels'] for r in results)
    print(f"\nTotal levels across all games: {total_levels}",flush=True)
    print(f"Elapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
