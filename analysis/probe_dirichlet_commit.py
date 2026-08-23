"""Can the power head COMMIT to the allocation it is failing to reach?

The teacher target concentrates private power with a large dynamic range. A
Dirichlet head can miss that in two different ways, and they need opposite fixes:

  (A) its MEAN is already spread but its CONCENTRATION is too low, so the drawn
      allocation is noise around a good intention -> an entropy / concentration
      problem, fixed by annealing or a concentration floor;

  (B) its MEAN is flat, i.e. it never intends the concentrated allocation at all
      -> an information or credit problem, and no entropy schedule helps.

Reports, per state: the dynamic range (max/min) of the head's Dirichlet MEAN, of
the EXECUTED allocation after the fairness blend and the sample, and of the
teacher target; plus the total concentration sum(alpha), which is what sets how
tightly a draw tracks the mean.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from infer import (load_training_cfg, load_agents, _compute_irs_favored,
                   _get_active_irs, _get_active_irs_ids, _build_phase_state)
from train import _power_aux_target


def dr(v):
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 1e-30)
    return float(v.max() / v.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--states", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_training_cfg(a.run)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.states + 5)
    actor, phase_net, power_net, ck_net = load_agents(a.run, seed=a.seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)

    dr_mean, dr_exec, dr_tgt, conc = [], [], [], []
    obs = env.reset(seed=a.seed)
    for _ in range(a.states):
        blocked = _compute_irs_favored(env)
        s_t = actor.extract_state(obs, demand, blocked)
        phi, _, ainfo = actor.forward(s_t, greedy=True)
        rep = ainfo.get("o_hat", ainfo["z_t"]) if getattr(actor, "NO_AE", False) \
            else ainfo["z_t"]
        active, ids = _get_active_irs(phi), _get_active_irs_ids(phi)
        s_ph = _build_phase_state(env.channels, phi, cfg, rep)
        pidx, _, _ = phase_net.forward(s_ph, active, greedy=True)
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

        # raw Dirichlet parameters, before sampling and before the fairness blend
        logits = power_net._full_logits(sp)
        mask = power_net._common_mask(ids)
        _a_s, _a_c, a_p = power_net._conc(logits, mask)
        dr_mean.append(dr(a_p / a_p.sum()))
        conc.append(float(a_p.sum()))

        wc_p, wp_p, _, _ = power_net.forward(sp, ids)
        dr_exec.append(dr(wp_p))

        tgt = _power_aux_target(env.rate_computer, phi, Phi, env.channels,
                               ids, h, cfg)
        if tgt is not None:
            dr_tgt.append(dr(tgt["private"]))

        obs, _r, _d, _i = env.step({"assignment": phi, "phase_idx": pidx,
                                    "w_p": wp_p, "w_c_vec": wc_p,
                                    "C_k": np.zeros(cfg.K)})

    def line(name, v):
        v = np.array(v)
        print(f"  {name:<34}{np.median(v):>12.1f}{np.percentile(v,25):>11.1f}"
              f"{np.percentile(v,75):>11.1f}")

    print("=" * 76)
    print(f"  DIRICHLET COMMITMENT · {a.run}")
    print(f"  K={cfg.K} M={cfg.M} · {a.states} states")
    print("=" * 76)
    print(f"  {'dynamic range (max/min)':<34}{'median':>12}{'p25':>11}{'p75':>11}")
    print("  " + "-" * 66)
    line("head's Dirichlet MEAN", dr_mean)
    line("EXECUTED (blend + sample)", dr_exec)
    if dr_tgt:
        line("teacher TARGET", dr_tgt)
    print("  " + "-" * 66)
    c = np.array(conc)
    print(f"  {'sum(alpha) private':<34}{np.median(c):>12.1f}"
          f"{np.percentile(c,25):>11.1f}{np.percentile(c,75):>11.1f}")
    print("=" * 76)
    if dr_tgt:
        m, t = float(np.median(dr_mean)), float(np.median(dr_tgt))
        if m < 0.2 * t:
            print("  MEAN is far flatter than the target -> (B) the head does not")
            print("  intend the concentrated allocation. Entropy schedules cannot fix")
            print("  an intention that was never formed.")
        else:
            print("  MEAN tracks the target but execution does not -> (A) sampling /")
            print("  fairness blend is destroying a good intention: a concentration")
            print("  floor or a lower alpha_priv is the lever.")
    print("=" * 76)


if __name__ == "__main__":
    main()
