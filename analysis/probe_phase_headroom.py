"""How much J is really available on the phase axis?

The gap decomposition substitutes `oracle_phase_idx`, which maximises the
coherent gain Sum_g |h_g|^2 — a surrogate, not the objective. It sees no power,
no C_k, no interference denominator and no QoS penalty. So the "+0.130 J from
oracle phase" it reports is what that SURROGATE attains, and the true phase
headroom can only be larger.

This measures three phases on the same state, everything else held at what the
agent chose:

  agent        the deployed PhaseMLP output
  gain-oracle  oracle_phase_idx           (the surrogate the teacher imitates)
  J-oracle     oracle_phase_search(score=J)  (what AO's phase step actually does)

The gap between the last two is the part of the phase axis the current teacher
cannot point at, however strongly it is weighted.
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import params as P
from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from RL import QuantumActor, ClassicalActor, PhaseMLP, PowerMLP, CkMLP
from train import (_build_phase_state, _build_ck_state, _get_active_irs,
                   _get_active_irs_ids, _compute_irs_favored, _build_power_state)
from analysis.oracle_alloc import phi_from_idx
from analysis.phase_oracle import oracle_phase_idx, oracle_phase_search, est_channels

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--K', type=int, required=True)
ap.add_argument('--M', type=int, required=True)
ap.add_argument('--episodes', type=int, default=3)
ap.add_argument('--steps', type=int, default=40)
ap.add_argument('--seed', type=int, default=77)
ap.add_argument('--draws', type=int, default=16)
ap.add_argument('--sweeps', type=int, default=1)
a = ap.parse_args()

cfg = make_config(K=a.K, M=a.M, P_S_dBm=50.0, R_LoS_km=0.5)
rate = RateComputer(cfg)
Dk, eps, lam = cfg.D_k_bps_hz, cfg.epsilon_qp, cfg.lambda_D
d = a.ckpt if os.path.isfile(os.path.join(a.ckpt, 'actor_config.json')) \
    else os.path.join(a.ckpt, 'agents')
ac = json.load(open(os.path.join(d, 'actor_config.json')))
actor = (ClassicalActor if ac.get('mode') == 'classical' else QuantumActor).from_dir(d, seed=0)
if ac.get('mode') != 'classical':
    actor.n_shots = P.n_shots_eval
ph, pw, ck = (PhaseMLP.from_dir(d, seed=0), PowerMLP.from_dir(d, seed=0),
              CkMLP.from_dir(d, seed=0))
hp = os.path.abspath(d)
for _ in range(5):
    hp = os.path.dirname(hp)
    f = os.path.join(hp, 'hyperparameters.json')
    if os.path.isfile(f):
        pn = json.load(open(f)).get('power_net', {})
        pw.power_fairness = float(pn.get('power_fairness', 0.0))
        pw.power_fairness_priv = float(pn.get('power_fairness_priv',
                                              pn.get('power_fairness', 0.0)))
        pw.power_priv_frac = float(pn.get('power_priv_frac', 0.8))
        break

env = ISTNEnv(cfg=cfg, seed=a.seed, n_steps_ep=a.steps, reward_noise_avg=1)
dem = np.full(cfg.K, Dk)
acc = {k: [] for k in ('agent', 'gain', 'jopt')}


def score(phi, pidx, wp, wc, act_ids, C_k, ch, sig):
    Phi = phi_from_idx(pidx, cfg)
    s = 0.0
    for s2 in sig:
        out = rate.compute_sum_rate(assignment=phi, Phi=Phi, channels=ch, w_p=wp,
                                    w_c_vec=wc, C_k=C_k, active_irs_ids=act_ids,
                                    sigma2=s2)
        R = np.asarray(out['R_private']) + np.asarray(out['C_k'])
        sh = np.maximum(0.0, Dk - R) / (Dk + eps)
        s += float(out['sum_rate']) - lam * float((sh ** 2).sum())
    return s / len(sig)


for ep in range(a.episodes):
    obs = env.reset(seed=a.seed + ep)
    for _ in range(a.steps):
        ch = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in env.channels.items()}
        sig = [env.channel_model.sample_noise_sigma2() for _ in range(a.draws)]
        s_t = actor.extract_state(obs, dem, _compute_irs_favored(env))
        phi, _, inf = actor.forward(s_t, greedy=True)
        rep = inf['z_t']
        act_irs, act_ids = _get_active_irs(phi), _get_active_irs_ids(phi)
        p_agent, _, _ = ph.forward(_build_phase_state(ch, phi, cfg, rep),
                                   act_irs, greedy=True)

        def downstream(pidx):
            """Power and C_k re-solved under THIS phase — the phase change must be
            allowed to propagate, or the comparison credits phase for an
            allocation chosen against a different channel."""
            Phi = phi_from_idx(pidx, cfg)
            h = rate.effective_channels_all(phi, Phi, ch)
            wc, wp, _, _ = pw.forward(_build_power_state(h, rep), act_ids)
            part = rate.compute_rates_partial(phi, Phi, ch, wp, wc,
                                              active_irs_ids=act_ids)
            C_k, _, _ = ck.forward(_build_ck_state(dem, part['R_private'],
                                                   part['R_c_group'], phi, cfg),
                                   phi, part['R_c_group'])
            return wp, wc, C_k

        est = est_channels(ch)
        p_gain = oracle_phase_idx(est, phi, cfg)

        def sc(pidx):
            wp, wc, C_k = downstream(pidx)
            return score(phi, pidx, wp, wc, act_ids, C_k, ch, sig)

        p_j, _, _ = oracle_phase_search(est, phi, cfg, sc, n_sweeps=a.sweeps)
        for key, pidx in (('agent', p_agent), ('gain', p_gain), ('jopt', p_j)):
            acc[key].append(sc(pidx))

        obs, _, _, _ = env.step({'assignment': phi, 'phase_idx': p_agent,
                                 'w_p': downstream(p_agent)[0],
                                 'w_c_vec': downstream(p_agent)[1],
                                 'C_k': downstream(p_agent)[2]})

n = len(acc['agent'])
ag, gn, jo = (np.mean(acc[k]) for k in ('agent', 'gain', 'jopt'))
print("=" * 74)
print(f"  PHASE HEADROOM  K={cfg.K} M={cfg.M}  {n} states, {a.draws} sigma^2 draws")
print("=" * 74)
print(f"  agent phase (deployed)            J = {ag:.4f}")
print(f"  gain-oracle  (Sum|h|^2 surrogate) J = {gn:.4f}   +{gn-ag:.4f}")
print(f"  J-oracle     (search on J)        J = {jo:.4f}   +{jo-ag:.4f}")
print("  " + "-" * 70)
print(f"  headroom the CURRENT teacher can point at : {gn-ag:+.4f}")
print(f"  headroom it CANNOT (surrogate vs J-opt)   : {jo-gn:+.4f}")
print(f"  total phase headroom                      : {jo-ag:+.4f}")
print("=" * 74)
