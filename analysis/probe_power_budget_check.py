"""
probe_power_budget_check.py
---------------------------
Does every policy actually spend at most P_S?

Written because PPO-flat retrained at C1 P_S=70 reports R_tot 10.30 at QoS 99.9%
against 3.82 for the DNN and 2.08 for AO at the same budget. That gap is large enough
that the first thing to rule out is a policy emitting more transmit power than it is
allowed, which would invalidate the comparison rather than win it.

Reports, per arm, the total emitted power sum(w_p) + sum(w_c) as a fraction of the
linear budget P_S, plus how the spend splits between private and common streams. A
compliant policy sits at or below 1.0.

Usage:
  python analysis/probe_power_budget_check.py \
      --env-run results/result_399/checkpoints/ep_20004 \
      --arms flat:results/result_399/checkpoints/ep_20004 \
             dnn:results/result_375/checkpoints/ep_02000 \
      --seeds 42 --states 60
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from infer import (load_training_cfg, load_agents, _compute_irs_favored,
                   _get_active_irs, _get_active_irs_ids, _build_phase_state)


def spend(env, cfg, demand, nets, flat_prev):
    actor, phase_net, power_net, ck_net = nets
    obs = env._get_obs()
    blocked = _compute_irs_favored(env)
    is_flat = phase_net is None

    if is_flat:
        h_now = env.rate_computer.effective_channels_all(
            env.assignment, env.Phi, env.channels)
        s_t = actor.extract_state(obs, demand, blocked, h_eff=h_now,
                                  prev_rates=flat_prev, update_norm=False)
        a, _, fi = actor.forward(s_t, greedy=True)
        wp, wc = fi["w_p"], fi["w_c_vec"]
    else:
        s_t = actor.extract_state(obs, demand, blocked)
        phi, _, ainfo = actor.forward(s_t, greedy=True)
        rep = ainfo["z_t"]
        active, ids = _get_active_irs(phi), _get_active_irs_ids(phi)
        if getattr(phase_net, "N", None) != cfg.N:
            phase_net.N = cfg.N
        pidx, _, _ = phase_net.forward(
            _build_phase_state(env.channels, phi, cfg, rep), active, greedy=True)
        Phi = env.phase_model.build_phi(env.phase_model.index_to_phase(pidx))
        h = env.rate_computer.effective_channels_all(phi, Phi, env.channels)
        hs = float(np.mean(np.abs(h))) + 1e-9
        sp = np.concatenate([h.real / hs, h.imag / hs])
        if getattr(power_net, "scale_feats", False):
            from train import _power_scale_feats
            cfg.power_feat_scale = getattr(power_net, "feat_scale", 1.0)
            sp = np.concatenate([sp, _power_scale_feats(h, cfg)])
        if getattr(power_net, "mag_feats", None):
            from train import _power_mag_feats
            sp = np.concatenate([sp, _power_mag_feats(h, power_net.mag_feats)])
        if power_net.d_s > sp.shape[0]:
            sp = np.concatenate([sp, rep])
        wc, wp, _, _ = power_net.forward(sp, ids)

    return float(np.sum(wp)), float(np.sum(np.atleast_1d(wc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-run", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="LABEL:run_dir ...")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--states", type=int, default=60)
    a = ap.parse_args()

    cfg = load_training_cfg(a.env_run)
    P_lin = 10.0 ** (cfg.P_S_dBm / 10.0) / 1000.0
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    print("env %s   K=%d M=%d P_S=%.1f dBm  ->  budget %.4g W"
          % (a.env_run, cfg.K, cfg.M, cfg.P_S_dBm, P_lin))

    for spec in a.arms:
        lab, path = spec.split(":", 1)
        nets = load_agents(path, seed=0)
        priv, com = [], []
        for sd in a.seeds:
            env = ISTNEnv(cfg, seed=sd, n_steps_ep=a.states + 5)
            env.reset(seed=sd)
            for _ in range(a.states):
                p, c = spend(env, cfg, demand, nets, None)
                priv.append(p)
                com.append(c)
                env.user_pos = env._walk_users(env.user_pos)
                env.channels = env.channel_model.update_user_channels(
                    env.user_pos, env.irs_pos, env.channels)
        priv, com = np.array(priv), np.array(com)
        tot = priv + com
        print("  %-6s  private %.4g  common %.4g  total %.4g   "
              "total/budget  mean %.4f  max %.4f   over-budget states %.1f%%"
              % (lab, priv.mean(), com.mean(), tot.mean(),
                 (tot / P_lin).mean(), (tot / P_lin).max(),
                 100.0 * float(np.mean(tot > P_lin * 1.001))))


if __name__ == "__main__":
    sys.exit(main())
