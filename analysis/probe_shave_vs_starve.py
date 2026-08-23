"""
probe_shave_vs_starve.py
------------------------
Two policies, ONE environment, and the question the binary QoS column cannot answer.

Motivating case: at P_S=70 a base policy transferred zero-shot keeps a high QoS while a
policy retrained at 70 shows a lower QoS and a much larger R_tot. Two readings fit
those aggregates equally well:

  SHAVING   both policies hold every user near D_k, and the retrained one simply sits
            a hair lower, so its extra rate is bought from margin nobody needed.
  STARVING  the retrained policy abandons some users outright and spends their power
            elsewhere, so its extra rate is bought from service.

J at lambda_D = 1.5 cannot separate them, because the shortfall penalty is quadratic
and normalised: a user at 0.9*D_k costs about 0.015. The discriminating statistic is
WHERE the failures sit, not how many there are.

Differences from probe_flat_vs_hier.py, both deliberate:

  ONE cfg    that probe calls load_training_cfg per run, so two policies trained at
             different P_S are scored in two different environments. Here every arm is
             scored in the environment of --env-run, which is what makes a zero-shot
             arm comparable to a retrained one.

  OWN C_k    that probe substitutes the oracle C_k rule into every arm, correct when
             comparing against AO. Here both arms are ours and the C_k head is one of
             the three levers that produce threshold-filling in the first place, so
             substituting the oracle would erase the very effect under test.

States are shared, not merely seeded alike: each arm decides on the same obs before the
users are walked once for everybody.

Usage:
  python analysis/probe_shave_vs_starve.py \
      --env-run results/result_382/checkpoints/ep_02900 \
      --arms base:results/result_289 ps70:results/result_382/checkpoints/ep_02900 \
      --seeds 42 0 1 --states 200
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from infer import (load_training_cfg, load_agents, _compute_irs_favored,
                   _get_active_irs, _get_active_irs_ids, _build_phase_state,
                   _build_ck_state)

LAMBDAS = (1.5, 3.0, 5.0, 10.0, 20.0)
BANDS = [(1.10, np.inf, ">1.10"), (1.00, 1.10, "1.00-1.10"),
         (0.90, 1.00, "0.90-1.00"), (0.70, 0.90, "0.70-0.90"),
         (0.50, 0.70, "0.50-0.70"), (-np.inf, 0.50, "<0.50")]


def decide(env, cfg, demand, nets):
    """One greedy decision, scored on the true channel with the policy's own C_k."""
    actor, phase_net, power_net, ck_net = nets
    obs = env._get_obs()
    blocked = _compute_irs_favored(env)

    s_t = actor.extract_state(obs, demand, blocked)
    phi, _, ainfo = actor.forward(s_t, greedy=True)
    z_t = ainfo["z_t"]
    rep = ainfo.get("o_hat", z_t) if getattr(actor, "NO_AE", False) else z_t

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

    rc = env.rate_computer
    partial = rc.compute_rates_partial(phi, Phi, env.channels, wp, wc,
                                       active_irs_ids=ids)
    s_ck = _build_ck_state(demand, partial["R_private"], partial["R_c_group"],
                           phi, cfg)
    C_k, _, _ = ck_net.forward(s_ck, phi, partial["R_c_group"])

    o = rc.compute_sum_rate(phi, Phi, env.channels, wp, wc, C_k=C_k,
                            active_irs_ids=ids, use_true=True)
    R_k = np.asarray(o["R_private"]) + np.asarray(o["C_k"])
    return R_k, float(o["sum_rate"]), float(np.mean(phi > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-run", required=True,
                    help="run whose training_config defines the SHARED environment")
    ap.add_argument("--arms", nargs="+", required=True, help="LABEL:run_dir ...")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1])
    ap.add_argument("--states", type=int, default=200)
    a = ap.parse_args()

    cfg = load_training_cfg(a.env_run)
    Dk = cfg.D_k_bps_hz
    demand = np.full(cfg.K, Dk)
    print("shared env from %s  ->  K=%d M=%d P_S=%.1f dBm  D_k=%.4f  lambda_D=%.2f"
          % (a.env_run, cfg.K, cfg.M, cfg.P_S_dBm, Dk,
             float(getattr(cfg, "lambda_D", 1.5))))
    print("%d states x %d seeds, greedy, decide on g-hat / score on true g, own C_k"
          % (a.states, len(a.seeds)))

    labels, nets = [], {}
    for spec in a.arms:
        lab, path = spec.split(":", 1)
        labels.append(lab)
        nets[lab] = load_agents(path, seed=0)
        print("  arm %-6s <- %s" % (lab, path))

    ratio = {l: [] for l in labels}     # R_k / D_k, flattened over users+states
    rtot = {l: [] for l in labels}
    irs = {l: [] for l in labels}

    for sd in a.seeds:
        env = ISTNEnv(cfg, seed=sd, n_steps_ep=a.states + 5)
        env.reset(seed=sd)
        for _ in range(a.states):
            for l in labels:
                R_k, tot, fr = decide(env, cfg, demand, nets[l])
                ratio[l].append(R_k / Dk)
                rtot[l].append(tot)
                irs[l].append(fr)
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)

    W = 78
    print("\n" + "=" * W)
    print("  RATE vs QoS")
    print("=" * W)
    print("  %-8s %10s %8s %9s %10s" % ("arm", "R_tot", "QoS", "IRS-use", "J(1.5)"))
    Js = {}
    for l in labels:
        r = np.concatenate([np.asarray(x) for x in ratio[l]]).reshape(-1)
        short = np.clip(1.0 - r, 0.0, None)
        rt = float(np.mean(rtot[l]))
        for lam in LAMBDAS:
            Js.setdefault(l, {})[lam] = rt - lam * float(
                np.sum(short ** 2) / len(ratio[l]))
        print("  %-8s %10.4f %7.1f%% %8.1f%% %10.4f"
              % (l, rt, 100 * float(np.mean(r >= 1.0)),
                 100 * float(np.mean(irs[l])), Js[l][1.5]))

    print("\n" + "=" * W)
    print("  SERVICE  ·  share of user-slots by R_k / D_k")
    print("=" * W)
    print("  %-11s %s" % ("band", "".join("%10s" % l for l in labels)))
    for lo, hi, name in BANDS:
        row = ""
        for l in labels:
            r = np.concatenate([np.asarray(x) for x in ratio[l]]).reshape(-1)
            row += "%9.1f%%" % (100 * float(np.mean((r >= lo) & (r < hi))))
        print("  %-11s %s" % (name, row))

    print("\n  %-11s %s" % ("stat", "".join("%10s" % l for l in labels)))
    for name, fn in [
            ("median", lambda r: np.median(r)),
            ("p05", lambda r: np.percentile(r, 5)),
            ("p25", lambda r: np.percentile(r, 25)),
            ("min", lambda r: r.min()),
            ("med|fail", lambda r: np.median(r[r < 1.0]) if (r < 1.0).any() else np.nan),
            ("p05|fail", lambda r: np.percentile(r[r < 1.0], 5) if (r < 1.0).any() else np.nan),
            ("mean|serv", lambda r: r[r >= 1.0].mean() if (r >= 1.0).any() else np.nan)]:
        row = ""
        for l in labels:
            r = np.concatenate([np.asarray(x) for x in ratio[l]]).reshape(-1)
            row += "%10.3f" % fn(r)
        print("  %-11s %s" % (name, row))

    print("\n" + "=" * W)
    print("  J re-scored at other lambda_D")
    print("=" * W)
    print("  %-8s %s" % ("arm", "".join("%11s" % ("l=%g" % l) for l in LAMBDAS)))
    for l in labels:
        print("  %-8s %s" % (l, "".join("%11.3f" % Js[l][lam] for lam in LAMBDAS)))
    print("=" * W)
    print("  SHAVING reads as: failures bunched in 0.90-1.00, med|fail near 1,")
    print("  and mean|serv near 1. STARVING reads as: mass below 0.70, med|fail low,")
    print("  and mean|serv well above 1 where the freed power went.")


if __name__ == "__main__":
    sys.exit(main())
