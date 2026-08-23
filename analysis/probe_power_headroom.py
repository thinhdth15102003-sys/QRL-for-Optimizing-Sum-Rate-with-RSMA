"""Is the fairness prior capping us at AO's own power scheme?

AO's _best_power searches only the common/private SPLIT and then hands every
user an equal share: np.full(K, P_S * f / K). It never allocates private power
unequally. So on the power axis AO is a one-parameter method.

Our recipe runs --power-fairness-priv 0.3, which blends the executed private
distribution toward that same flat allocation. If unequal private power is worth
much, the prior is pulling us toward the baseline's weakness and parity is the
ceiling it buys.

Holds assignment and phase FIXED at what the trained policy chose, so the only
thing varying is power, and scores four allocations on identical states:

  policy   what the trained head actually emits (fairness blend included)
  equal    AO's scheme: best split, flat private  -- the baseline's own ceiling
  oracle   best split + coordinate ascent on the private distribution
  oracle+  same, with the split re-optimised after the private ascent

The gap equal -> oracle is the headroom the fairness prior is standing on. If it
is small, unequal private power is not the lever and the C2 story is not here.
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
from analysis.oracle_alloc import oracle_ck_met
from train import _power_aux_target


def make_scorer(env, cfg, use_true):
    """use_true=False -> the DESIGN view a deployed solver has (estimated ĝ).
    use_true=True     -> what the environment actually pays (true g).

    Searching and scoring on the same channels lets an allocation exploit
    estimation error it will not have at run time: the winner-take-all target
    the teacher grid picks is exactly the shape that breaks when the argmax is
    wrong. Every allocation here is CHOSEN on ĝ and PAID on g.
    """
    lam = float(getattr(cfg, "lambda_D", 1.5))
    eps = float(getattr(cfg, "epsilon_qp", 1e-3))
    Dk = cfg.D_k_bps_hz
    rc = env.rate_computer

    def ck_for(phi, Phi, wp, wc, active):
        # No use_true here, and that is not an oversight: compute_rates_partial
        # has none. C_k is a DESIGN-time choice made from ĝ, in both views -- a
        # deployment cannot pick C_k from a channel it has not measured. Only
        # the achieved sum rate below is evaluated on the true g.
        part = rc.compute_rates_partial(phi, Phi, env.channels, wp, wc,
                                        active_irs_ids=active)
        _, Ck = oracle_ck_met(part["R_private"], part["R_c_group"],
                              part["groups"], Dk)
        for gid, mem in part["groups"].items():
            mem = np.asarray(list(mem), dtype=int)
            if mem.size == 0:
                continue
            left = float(part["R_c_group"].get(int(gid), 0.0)) - float(Ck[mem].sum())
            if left > 1e-12:
                Ck[mem] += left / mem.size
        return Ck

    def J(phi, Phi, wp, wc, active):
        Ck = ck_for(phi, Phi, wp, wc, active)
        out = rc.compute_sum_rate(phi, Phi, env.channels, wp, wc, C_k=Ck,
                                  active_irs_ids=active, use_true=use_true)
        R = np.asarray(out["R_private"]) + np.asarray(out["C_k"])
        short = np.maximum(0.0, Dk - R) / (Dk + eps)
        # QoS of the LAST scored allocation. J alone cannot say whether a high
        # score came from serving everyone or from starving most users -- and an
        # unconstrained ascent on this objective will happily do the latter.
        J.last_qos = float(np.sum(R >= Dk)) / len(R)
        J.last_rtot = float(out["sum_rate"])
        return float(out["sum_rate"]) - lam * float((short ** 2).sum())
    J.last_qos = J.last_rtot = float("nan")
    return J


def best_split_equal(J, phi, Phi, active, cfg, splits):
    """AO's power step: grid on the split, flat private."""
    K, P_S = cfg.K, cfg.P_S
    best, arg = -np.inf, None
    for f in splits:
        wp = np.full(K, P_S * f / K)
        wc = np.full(len(active) + 1, P_S * (1.0 - f) / (len(active) + 1))
        v = J(phi, Phi, wp, wc, active)
        if v > best:
            best, arg = v, (f, wp, wc)
    return best, arg


def ascend_private(J, phi, Phi, active, cfg, f, sweeps=3):
    """Coordinate ascent on the private distribution at a fixed split."""
    K, P_S = cfg.K, cfg.P_S
    W = P_S * f
    wc = np.full(len(active) + 1, P_S * (1.0 - f) / (len(active) + 1))
    x = np.full(K, 1.0 / K)
    best = J(phi, Phi, x * W, wc, active)
    # Multipliers kept mild and the dynamic range capped: with (0.25..4.0) and
    # 3 sweeps the search reached max/min ~2.6e5, i.e. it had emptied nine users
    # to feed one. That is a real optimum of this J -- the shortfall penalty is
    # bounded, so starving a user costs at most lambda*(D/(D+eps))^2 -- but it is
    # not an allocation any deployment would accept, and quoting its J as
    # "headroom" would be quoting the objective's blind spot, not a target.
    for _ in range(sweeps):
        for k in range(K):
            for mul in (0.5, 0.8, 1.25, 2.0):
                t = x.copy()
                t[k] *= mul
                t /= t.sum()
                if t.max() / t.min() > 50.0:
                    continue
                v = J(phi, Phi, t * W, wc, active)
                if v > best:
                    best, x = v, t
    return best, x, wc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-split", dest="ns", type=int, default=11)
    a = ap.parse_args()

    cfg = load_training_cfg(a.run)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.states + 5)
    actor, phase_net, power_net, ck_net = load_agents(a.run, seed=a.seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    J = make_scorer(env, cfg, use_true=False)      # what a solver optimises on
    Jt = make_scorer(env, cfg, use_true=True)      # what the channel actually pays
    splits = np.linspace(0.05, 0.95, a.ns)

    keys = ("policy", "equal", "beta-grid", "oracle", "oracle+")
    res = {k: [] for k in keys}
    qos = {k: [] for k in keys}
    rtot = {k: [] for k in keys}
    flat = []

    def record(k, phi, Phi, wp, wc, ids):
        """Always scores through Jt: the allocation was chosen on ĝ above."""
        res[k].append(Jt(phi, Phi, wp, wc, ids))
        qos[k].append(Jt.last_qos)
        rtot[k].append(Jt.last_rtot)
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
        if power_net.d_s > sp.shape[0]:
            sp = np.concatenate([sp, rep])
        wc_p, wp_p, _, _ = power_net.forward(sp, ids)

        record("policy", phi, Phi, wp_p, wc_p, ids)

        je, (f, wpe, wce) = best_split_equal(J, phi, Phi, ids, cfg, splits)
        record("equal", phi, Phi, wpe, wce, ids)

        # The teacher that ALREADY exists (--power-aux-weight): private power
        # restricted to the one-parameter family |h|^{2b} on a 12-point (b,f)
        # grid. Scored through the SAME scorer as everything else, so the only
        # difference is the allocation and not the C_k convention. If this row
        # reaches oracle+, the old teacher's target was never the problem and a
        # richer target is not the fix.
        # The teacher builds its target on the ESTIMATED channel, exactly as it
        # does inside train.py -- h here is the effective channel the head sees.
        # _power_aux_target's own calls default to use_true=False, i.e. it builds
        # its target on ĝ exactly as it does inside train.py. Nothing to override.
        tgt = _power_aux_target(env.rate_computer, phi, Phi, env.channels,
                                ids, h, cfg)
        fb = float(tgt["split"][1]) if tgt is not None else f
        wpb = (np.asarray(tgt["private"], dtype=float) * cfg.P_S * fb
               if tgt is not None else wpe)
        wcb = np.full(len(ids) + 1, cfg.P_S * (1.0 - fb) / (len(ids) + 1))
        record("beta-grid", phi, Phi, wpb, wcb, ids)

        jo, x, wco = ascend_private(J, phi, Phi, ids, cfg, f)
        record("oracle", phi, Phi, x * cfg.P_S * f, wco, ids)
        flat.append(float(np.max(x) / np.min(x)))

        bestp, barg = jo, (x * cfg.P_S * f, wco)
        for f2 in splits:                       # re-optimise the split at that x
            W2 = cfg.P_S * f2
            wc2 = np.full(len(ids) + 1,
                          cfg.P_S * (1.0 - f2) / (len(ids) + 1))
            v = J(phi, Phi, x * W2, wc2, ids)
            if v > bestp:
                bestp, barg = v, (x * W2, wc2)
        record("oracle+", phi, Phi, barg[0], barg[1], ids)

        # advance on the policy's own action so states stay on-distribution
        part = env.rate_computer.compute_rates_partial(
            phi, Phi, env.channels, wp_p, wc_p, active_irs_ids=ids)
        s_ck = _build_ck_state(demand, part["R_private"], part["R_c_group"],
                               phi, cfg)
        C_k, _, _ = ck_net.forward(s_ck, phi, part["R_c_group"])
        obs, _r, _d, _i = env.step({"assignment": phi, "phase_idx": pidx,
                                    "w_p": wp_p, "w_c_vec": wc_p, "C_k": C_k})

    print("=" * 78)
    print(f"  POWER HEADROOM · {a.run} · K={cfg.K} M={cfg.M} · {a.states} states")
    print("  assignment + phase held FIXED at the policy's choice")
    print("=" * 78)
    base = float(np.mean(res["equal"]))
    print(f"  {'allocation':<28}{'J':>9}{'+/-':>7}{'vs equal':>10}"
          f"{'R_tot':>9}{'QoS':>8}")
    for k in keys:
        v = np.array(res[k])
        m, se = float(v.mean()), float(v.std() / np.sqrt(len(v)))
        tag = {"policy": "trained head (w/ prior)",
               "equal": "AO scheme: best split, flat",
               "beta-grid": "EXISTING teacher |h|^2b grid",
               "oracle": "best split + private ascent",
               "oracle+": "  ... + split re-optimised"}[k]
        print(f"  {tag:<28}{m:>9.4f}{se:>7.4f}{m - base:>+10.4f}"
              f"{float(np.mean(rtot[k])):>9.4f}"
              f"{100*float(np.mean(qos[k])):>7.1f}%")
    print("-" * 78)
    print(f"  private-weight dynamic range at the optimum "
          f"(max/min): mean {np.mean(flat):.1f}x  median {np.median(flat):.1f}x")
    print(f"  trained head vs AO's power scheme : "
          f"{float(np.mean(res['policy'])) - base:+.4f} J   "
          f"QoS {100*(float(np.mean(qos['policy'])) - float(np.mean(qos['equal']))):+.1f} pt")
    print("=" * 78)


if __name__ == "__main__":
    main()
