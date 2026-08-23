"""TRUE frontier: routing coordinate-ascent + per-element oracle phase +
(beta,f) power sweep + demand-fill C_k, maximising R_tot SUBJECT TO QoS >= t.
Fixes the two structural limits of the AO probe: it never sweeps beta (private
power is always equal) and its C_k is hard-coded QoS-first."""
import sys; sys.path.insert(0,"/mnt/c/Project/IRS-assisted RSMA Quantum-RL"); sys.path.insert(0,"/mnt/c/Project/IRS-assisted RSMA Quantum-RL/analysis")
import numpy as np, params as P
from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.phase_oracle import oracle_phase_idx, est_channels
from analysis.oracle_alloc import oracle_ck_met, phi_from_idx
from analysis.probe_ao_rate import AORatePolicy
import argparse
_ap=argparse.ArgumentParser()
_ap.add_argument("--K",type=int,default=10); _ap.add_argument("--M",type=int,default=2)
_ap.add_argument("--P",type=float,default=50.0)
_ap.add_argument("--R-LoS",dest="rlos",type=float,default=0.5)
_ap.add_argument("--episodes",type=int,default=2)
_ap.add_argument("--steps",type=int,default=15)
_a=_ap.parse_args()
cfg=make_config(K=_a.K,M=_a.M,P_S_dBm=_a.P,R_LoS_km=_a.rlos); rate=RateComputer(cfg)
NAVG=P.reward_noise_avg; Dk=cfg.D_k_bps_hz; K,M=cfg.K,cfg.M
env=ISTNEnv(cfg=cfg,seed=42,n_steps_ep=_a.steps,reward_noise_avg=NAVG)
ao=AORatePolicy(cfg,rate,rounds=2,n_split=11,route_ascent=True,phase_sweeps=1)
BET=[0.0,0.25,0.5,1.0,2.0,4.0,8.0,16.0]; SPL=np.linspace(0.05,0.95,13)
TH=[1.00,0.99,0.95,0.90,0.80,0.50,0.0]
def evaluate(a,ch,sig=None):
    """best (R_tot,QoS) pairs over the (beta,f) grid for a fixed routing."""
    if not np.any(a>0): return []
    Phi=phi_from_idx(oracle_phase_idx(est_channels(ch),a,cfg),cfg)
    act=sorted(set(int(x) for x in a if x>0)); G=len(act)
    g=np.abs(rate.effective_channels_all(a,Phi,ch))**2+1e-30
    out=[]
    for b in BET:
        w=g**b
        for f in SPL:
            wp=w/w.sum()*(cfg.P_S*f); wc=np.full(G+1,cfg.P_S*(1-f)/(G+1))
            part=rate.compute_rates_partial(a,Phi,ch,wp,wc,active_irs_ids=act)
            _,Ck=oracle_ck_met(part["R_private"],part["R_c_group"],part["groups"],Dk)
            for gid,mem in part["groups"].items():
                mem=np.asarray(list(mem),dtype=int)
                if mem.size:
                    left=float(part["R_c_group"].get(int(gid),0.0))-float(Ck[mem].sum())
                    if left>1e-12: Ck[mem]+=left/mem.size
            if sig is None:
                o=rate.compute_sum_rate(a,Phi,ch,wp,wc,C_k=Ck,active_irs_ids=act,sigma2=cfg.sigma2,use_true=True)
                rt=np.asarray(o["R_private"])+np.asarray(o["C_k"])
                out.append((float(o["sum_rate"]),float(np.mean(rt>=Dk))))
            else:
                sr=0.0; rt=np.zeros(K)
                for s2 in sig:
                    o=rate.compute_sum_rate(a,Phi,ch,wp,wc,C_k=Ck,active_irs_ids=act,sigma2=s2,use_true=True)
                    sr+=float(o["sum_rate"]); rt+=np.asarray(o["R_private"])+np.asarray(o["C_k"])
                n=len(sig); out.append((sr/n,float(np.mean(rt/n>=Dk))))
    return out
def sc_at(a,ch,t):
    # soft constraint: feasible points score R_tot, infeasible ones get a graded
    # penalty so the ascent has a direction instead of a flat -inf plateau
    pts=evaluate(a,ch)
    if not pts: return -1e6
    return max(r-10.0*max(0.0,t-q) for r,q in pts)
best={t:[] for t in TH}; aos=[]
for ep in range(_a.episodes):
    env.reset(seed=42+ep)
    for _ in range(_a.steps):
        ch=env.channels; sig=[env.channel_model.sample_noise_sigma2() for _ in range(NAVG)]
        act_ao=ao.act(env._get_obs(),env)
        aa=np.asarray(act_ao["assignment"],dtype=int)
        Pa=phi_from_idx(np.asarray(act_ao["phase_idx"],dtype=int),cfg)
        ai=sorted(set(int(x) for x in aa if x>0)); sr=0.0; rt=np.zeros(K)
        for s2 in sig:
            o=rate.compute_sum_rate(aa,Pa,ch,act_ao["w_p"],act_ao["w_c_vec"],C_k=act_ao["C_k"],active_irs_ids=ai,sigma2=s2,use_true=True)
            sr+=float(o["sum_rate"]); rt+=np.asarray(o["R_private"])+np.asarray(o["C_k"])
        aos.append((sr/NAVG,float(np.mean(rt/NAVG>=Dk))))
        for t in TH:
            a=np.asarray(act_ao["assignment"],dtype=int).copy()   # start FEASIBLE (AO)
            cur=sc_at(a,ch,t)
            for _rd in range(2):
                ch_ang=False
                for k in range(K):
                    b0=int(a[k])
                    for c in range(M+1):
                        if c==b0: continue
                        a[k]=c; v=sc_at(a,ch,t)
                        if v>cur: cur,b0,ch_ang=v,c,True
                        a[k]=b0
                    a[k]=b0
                if not ch_ang: break
            pts=evaluate(a,ch,sig); ok=[r for r,q in pts if q>=t-1e-9]
            best[t].append(max(ok) if ok else np.nan)
        env.user_pos=env._walk_users(env.user_pos)
        env.channels=env.channel_model.update_user_channels(env.user_pos,env.irs_pos,env.channels)
A=np.array(aos)
print("  K=%d M=%d R_LoS=%.1f P_S=%.0f" % (cfg.K,cfg.M,cfg.R_LoS_km,cfg.P_S_dBm))
print("  TRUE FRONTIER (routing ascent + beta sweep + oracle phase/Ck, use_true, %d sig)" % NAVG)
print("  QoS >= |  max R_tot | vs AO %.3f" % A[:,0].mean())
for t in TH:
    v=np.nanmean(best[t]); print("  %5.0f%% |  %8.3f | %+.3f" % (t*100,v,v-A[:,0].mean()))
print("  AO tren cung states: R_tot %.3f QoS %.1f%%" % (A[:,0].mean(),A[:,1].mean()*100))
