#!/usr/bin/env python3
"""
Step 385c -- Center + PCA self-encoding. Raw 64x64 -> center -> PCA 256.

Phase 1 (500 spawns): raw 4096-dim.
Phase 2: center by mean, PCA to 256 dims, re-encode codebook.
Recompute every 500 spawns. The encoding is fully state-derived.

50K steps on LS20. Kill: sim mean < 0.95 after center+PCA.
Script: scripts/run_step385c_center_pca.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_RAW=4096; D_PCA=256; CB_CAP=10000; THRESH_INT=100; PCA_INT=500

class CenterPCAFold:
    def __init__(self, d_raw, d_pca, k=3, dev=DEVICE):
        self.V_raw=torch.zeros(0,d_raw,device=dev)  # raw storage
        self.V=torch.zeros(0,d_pca,device=dev)  # projected (working)
        self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d_raw=d_raw; self.d_pca=d_pca; self.dev=dev
        self.spawn_count=0; self.mean=None; self.Wt=None; self.phase2=False
        self.pca_count=0; self.explained_var=0.0

    def reset_cb(self):
        self.V_raw=torch.zeros(0,self.d_raw,device=self.dev)
        self.V=torch.zeros(0,self.d_pca,device=self.dev)
        self.labels=torch.zeros(0,dtype=torch.long,device=self.dev)
        self.thresh=0.7; self.spawn_count=0; self.mean=None; self.Wt=None; self.phase2=False

    def _do_pca(self):
        n=self.V_raw.shape[0]
        if n<self.d_pca+10: return
        self.mean=self.V_raw.mean(dim=0)
        Vc=self.V_raw-self.mean.unsqueeze(0)
        U,S,Vt=torch.pca_lowrank(Vc,q=self.d_pca)
        self.Wt=Vt  # (d_raw, d_pca) — Vt from pca_lowrank is (d_raw, q)
        # Explained variance
        total_var=Vc.var(dim=0).sum()*n
        proj_var=(S[:self.d_pca]**2).sum()
        self.explained_var=float(proj_var/total_var) if total_var>0 else 0
        # Re-project codebook
        self.V=F.normalize(Vc@self.Wt,dim=1)  # (n, d_pca)
        self.phase2=True; self.pca_count+=1
        self._ut()

    def _project(self,x_raw):
        if self.mean is None or self.Wt is None:
            return F.normalize(x_raw[:self.d_pca],dim=0)
        xc=x_raw-self.mean
        return F.normalize(xc@self.Wt,dim=0)  # (d_pca,)

    def _fa(self,x_raw,l):
        x_raw=F.normalize(x_raw.to(self.dev).float(),dim=0)
        self.V_raw=torch.cat([self.V_raw,x_raw.unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        if self.phase2:
            xp=self._project(x_raw)
            self.V=torch.cat([self.V,xp.unsqueeze(0)])
        self.spawn_count+=1
        if self.spawn_count%PCA_INT==0: self._do_pca()
        elif self.spawn_count%THRESH_INT==0: self._ut()

    def _ut(self):
        if self.phase2:
            n=self.V.shape[0]
            if n<2: return
            ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
            s=self.V[idx]@self.V.T; t=s.topk(min(2,n),dim=1).values
            self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())
        else:
            n=self.V_raw.shape[0]
            if n<2: return
            ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
            s=self.V_raw[idx]@self.V_raw.T; t=s.topk(min(2,n),dim=1).values
            self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())

    def pn(self,x_raw,nc):
        x_raw=F.normalize(x_raw.to(self.dev).float(),dim=0)
        if self.V_raw.shape[0]==0:
            self.V_raw=x_raw.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev)
            return 0,0.0

        if self.phase2:
            xp=self._project(x_raw)
            si=self.V@xp
        else:
            si=self.V_raw@x_raw

        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)
        nsim=float(si.max().item())

        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V_raw.shape[0]<CB_CAP:
            self.V_raw=torch.cat([self.V_raw,x_raw.unsqueeze(0)])
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            if self.phase2:
                xp2=self._project(x_raw)
                self.V=torch.cat([self.V,xp2.unsqueeze(0)])
            self.spawn_count+=1
            if self.spawn_count%PCA_INT==0: self._do_pca()
            elif self.spawn_count%THRESH_INT==0: self._ut()
        else:
            w_i=si.argmax().item()
            a=1.0-float(si[w_i].item())
            if self.phase2:
                self.V[w_i]=F.normalize(self.V[w_i]+a*(xp-self.V[w_i]),dim=0)
                # Also update raw (approximate — project back isn't exact)
                self.V_raw[w_i]=F.normalize(self.V_raw[w_i]+a*(x_raw-self.V_raw[w_i]),dim=0)
            else:
                self.V_raw[w_i]=F.normalize(self.V_raw[w_i]+a*(x_raw-self.V_raw[w_i]),dim=0)
        return p,nsim


def main():
    t0=time.time()
    print(f"Step 385c -- Center + PCA. Raw 64x64 -> center -> PCA {D_PCA}. 50K steps.",flush=True)
    print(f"Device: {DEVICE}  PCA every {PCA_INT} spawns. Cap={CB_CAP}.",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=CenterPCAFold(d_raw=D_RAW,d_pca=D_PCA,k=3)
    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    action_counts={}; sims_log=[]

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!",flush=True); break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)

        if not sd and fold.V_raw.shape[0]<na:
            i=fold.V_raw.shape[0]; fold._fa(raw,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if fold.V_raw.shape[0]>=na: sd=True; fold._ut()
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
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V_raw.shape[0]}"
                  f"  pca={'ON' if fold.phase2 else 'OFF'} go={go}",flush=True)

        if ts%5000==0:
            avg_sim=np.mean(sims_log[-500:]) if sims_log else 0
            sim_std=np.std(sims_log[-500:]) if len(sims_log)>10 else 0
            print(f"    [step {ts:6d}] cb={fold.V_raw.shape[0]}"
                  f"  phase={'PCA' if fold.phase2 else 'RAW'}"
                  f"  pca_count={fold.pca_count}"
                  f"  explained={fold.explained_var:.3f}"
                  f"  thresh={fold.thresh:.4f}"
                  f"  avg_sim={avg_sim:.4f} sim_std={sim_std:.4f}"
                  f"  levels={lvls} go={go}",flush=True)
            print(f"      acts: {action_counts}",flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 385c SUMMARY",flush=True); print("="*60,flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}",flush=True)
    print(f"cb_final={fold.V_raw.shape[0]}  thresh={fold.thresh:.4f}",flush=True)
    print(f"phase2={fold.phase2}  pca_count={fold.pca_count}  explained={fold.explained_var:.3f}",flush=True)
    print(f"action_counts: {action_counts}",flush=True)
    if sims_log:
        sl=np.array(sims_log)
        print(f"Sim: min={sl.min():.4f} max={sl.max():.4f} mean={sl.mean():.4f} std={sl.std():.4f}",flush=True)
    if lvls>0:
        print("\nPASS: Level from raw pixels with self-derived center+PCA!",flush=True)
    elif sims_log and np.mean(sims_log[-500:])<0.95:
        print("\nMARGINAL: center+PCA improved sim but no level.",flush=True)
    else:
        print("\nKILL.",flush=True)
    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
