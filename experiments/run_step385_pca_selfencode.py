#!/usr/bin/env python3
"""
Step 385 -- PCA self-encoding at 64x64 raw on LS20.

Phase 1 (500 spawns): raw 4096-dim codebook.
Phase 2: PCA of codebook -> 256-dim projection. Re-encode all entries.
Recompute PCA every 500 spawns.

The encoding IS state-derived: PCA of the substrate's own experience.
50K steps. Kill: sim mean < 0.99 after PCA.
Script: scripts/run_step385_pca_selfencode.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_RAW = 4096; D_PCA = 256; CB_CAP = 10000; THRESH_INT = 100; PCA_INT = 500

class PCAFold:
    def __init__(self, d_raw, d_pca, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d_raw,device=dev)  # always store RAW
        self.V_proj=torch.zeros(0,d_pca,device=dev)  # projected view
        self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d_raw=d_raw; self.d_pca=d_pca; self.dev=dev
        self.spawn_count=0; self.W=None  # PCA projection (d_pca, d_raw)
        self.pca_active=False; self.pca_count=0
        self.explained_var=0.0

    def reset_cb(self):
        self.V=torch.zeros(0,self.d_raw,device=self.dev)
        self.V_proj=torch.zeros(0,self.d_pca,device=self.dev)
        self.labels=torch.zeros(0,dtype=torch.long,device=self.dev)
        self.thresh=0.7; self.spawn_count=0; self.W=None; self.pca_active=False

    def _compute_pca(self):
        n=self.V.shape[0]
        if n<self.d_pca+10: return
        # PCA via torch.pca_lowrank
        U,S,Vt=torch.pca_lowrank(self.V,q=self.d_pca)
        self.W=Vt.T  # (d_pca, d_raw)
        # Explained variance
        total_var=self.V.var(dim=0).sum()
        proj_var=(S[:self.d_pca]**2).sum()/n
        self.explained_var=float(proj_var/total_var) if total_var>0 else 0
        # Re-project all codebook entries
        self.V_proj=F.normalize(self.V@self.W.T,dim=1)  # (n, d_pca)
        self.pca_active=True
        self.pca_count+=1
        self._ut()

    def _project(self,x_raw):
        if self.W is None: return F.normalize(x_raw[:self.d_pca],dim=0)  # fallback
        return F.normalize(x_raw@self.W.T,dim=0)

    def _fa(self,x_raw,l):
        x_raw=F.normalize(x_raw.to(self.dev).float(),dim=0)
        self.V=torch.cat([self.V,x_raw.unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        if self.pca_active:
            xp=self._project(x_raw)
            self.V_proj=torch.cat([self.V_proj,xp.unsqueeze(0)])
        self.spawn_count+=1
        if self.spawn_count%PCA_INT==0: self._compute_pca()
        elif self.spawn_count%THRESH_INT==0: self._ut()

    def _ut(self):
        if self.pca_active:
            n=self.V_proj.shape[0]
            if n<2: return
            ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
            s=self.V_proj[idx]@self.V_proj.T; t=s.topk(min(2,n),dim=1).values
            self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())
        else:
            n=self.V.shape[0]
            if n<2: return
            ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
            s=self.V[idx]@self.V.T; t=s.topk(min(2,n),dim=1).values
            self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())

    def pn(self,x_raw,nc):
        x_raw=F.normalize(x_raw.to(self.dev).float(),dim=0)
        if self.V.shape[0]==0:
            self.V=x_raw.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev)
            return 0,0.0

        if self.pca_active:
            xp=self._project(x_raw)
            si=self.V_proj@xp
        else:
            si=self.V@x_raw

        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)
        nsim=float(si.max().item())

        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x_raw.unsqueeze(0)])
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            if self.pca_active:
                xp2=self._project(x_raw)
                self.V_proj=torch.cat([self.V_proj,xp2.unsqueeze(0)])
            self.spawn_count+=1
            if self.spawn_count%PCA_INT==0: self._compute_pca()
            elif self.spawn_count%THRESH_INT==0: self._ut()
        else:
            w_i=si.argmax().item()
            raw_sim=float((self.V[w_i]@x_raw).item())
            a=1.0-raw_sim
            self.V[w_i]=F.normalize(self.V[w_i]+a*(x_raw-self.V[w_i]),dim=0)
            if self.pca_active:
                self.V_proj[w_i]=self._project(self.V[w_i])
        return p,nsim


def main():
    t0=time.time()
    print(f"Step 385 -- PCA self-encoding. Raw 64x64 -> PCA 256. LS20. 50K steps.",flush=True)
    print(f"Device: {DEVICE}  PCA every {PCA_INT} spawns. Cap={CB_CAP}.",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=PCAFold(d_raw=D_RAW,d_pca=D_PCA,k=3)
    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False; ls_step=0
    action_counts={}; sims_log=[]

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset(); ls_step=ts; sd=False
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!",flush=True); break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)

        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(raw,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if fold.V.shape[0]>=na: sd=True; fold._ut()
            continue
        if not sd: sd=True

        c,nsim=fold.pn(raw,nc=na)
        sims_log.append(nsim)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed; life_s=ts-ls_step; ls_step=ts
            fold.reset_cb(); sd=False
            print(f"    LEVEL {lvls} at step {ts} (life={life_s}) go={go} -> RESET",flush=True)

        if ts%5000==0:
            avg_sim=np.mean(sims_log[-500:]) if sims_log else 0
            sim_std=np.std(sims_log[-500:]) if len(sims_log)>10 else 0
            print(f"    [step {ts:6d}] cb={fold.V.shape[0]} thresh={fold.thresh:.4f}"
                  f"  pca={'ON' if fold.pca_active else 'OFF'}"
                  f"  pca_count={fold.pca_count}"
                  f"  explained={fold.explained_var:.3f}"
                  f"  avg_sim={avg_sim:.4f} sim_std={sim_std:.4f}"
                  f"  levels={lvls} go={go}",flush=True)
            print(f"      acts: {action_counts}",flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 385 SUMMARY",flush=True); print("="*60,flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}",flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.4f}",flush=True)
    print(f"pca_active={fold.pca_active}  pca_count={fold.pca_count}  explained={fold.explained_var:.3f}",flush=True)
    print(f"action_counts: {action_counts}",flush=True)
    if sims_log:
        sl=np.array(sims_log)
        print(f"Sim stats: min={sl.min():.4f} max={sl.max():.4f} mean={sl.mean():.4f} std={sl.std():.4f}",flush=True)
    if lvls>0:
        print("\nPASS: Level completed with self-derived PCA encoding!",flush=True)
    elif fold.pca_active and sims_log and np.mean(sims_log[-500:])<0.99:
        print("\nMARGINAL: PCA reduced sim but no level.",flush=True)
    else:
        print("\nKILL.",flush=True)
    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
