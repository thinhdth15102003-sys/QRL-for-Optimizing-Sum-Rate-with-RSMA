"""
probe_vs_ao.py
--------------
Head-to-head: trained policies against AO on identical states, over several env
seeds.

Split out from probe_gap_split.py, which also runs the three oracle swaps. Those
dominate its cost (per-element phase search, routing ascent) and are not needed
when the question is only "how far from AO". Dropping them buys enough speed to
run the several env seeds this comparison actually requires: measured spread
across env seeds is +-0.043 J, which is LARGER than the differences between the
policies being compared, so a single-seed table would rank them by noise.

Same conventions as the rest of analysis/ (see PROBES.md):
  * scored on J, not R_tot
  * every decision made on the estimated channel, every score paid on the true one
  * the user walk is independent of the action, so all policies replay the SAME
    states rather than each drifting into its own trajectory
  * AO at its measured plateau setting (rounds=1, phase_sweeps=1, n_split=11);
    deeper settings buy no J and would inflate our margin

Usage:
  python analysis/probe_vs_ao.py \
      --runs VQC:results/result_278/checkpoints/ep_01000 \
             DNN:results/result_280/checkpoints/ep_09100 \
             DNN-base:results/result_232 \
      --seeds 42 7 21 --states 100
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from infer import (load_training_cfg, load_agents, _compute_irs_favored,
                   _get_active_irs, _get_active_irs_ids, _build_phase_state,
                   _build_ck_state)
from analysis.oracle_alloc import oracle_ck_met, phi_from_idx
from analysis.probe_ao_rate import AORatePolicy


def make_J(env, cfg, use_true):
    rc, Dk = env.rate_computer, cfg.D_k_bps_hz
    lam = float(getattr(cfg, "lambda_D", 1.5))
    eps = float(getattr(cfg, "epsilon_qp", 1e-3))

    def J(phi, Phi, wp, wc, ids, Ck=None):
        p = rc.compute_rates_partial(phi, Phi, env.channels, wp, wc,
                                     active_irs_ids=ids)
        # Default C_k is `oracle_ck_met`, which maximises #met (QoS), NOT J.
        # AO's own _ck calls exactly this, so AO is locked QoS-first on the
        # common-rate split. Applying it to the policy too makes the other three
        # heads comparable, but it also DELETES the one lever the policy has and
        # AO does not. Pass Ck to score the policy's own CkMLP instead.
        if Ck is None:
            _, Ck = oracle_ck_met(p["R_private"], p["R_c_group"], p["groups"], Dk)
            for g, m in p["groups"].items():
                m = np.asarray(list(m), dtype=int)
                if m.size:
                    left = float(p["R_c_group"].get(int(g), 0.0)) - float(Ck[m].sum())
                    if left > 1e-12:
                        Ck[m] += left / m.size
        o = rc.compute_sum_rate(phi, Phi, env.channels, wp, wc, C_k=Ck,
                                active_irs_ids=ids, use_true=use_true)
        R = np.asarray(o["R_private"]) + np.asarray(o["C_k"])
        s = np.maximum(0.0, Dk - R) / (Dk + eps)
        J.qos = float(np.sum(R >= Dk)) / len(R)
        J.rtot = float(o["sum_rate"])
        # Kept for the flat actor's rate-feedback state, which needs last step's
        # (R_private, mean C_k) exactly as train_flat.rollout feeds it back.
        J.rpriv = np.asarray(o["R_private"])
        J.ckmean = float(np.mean(np.asarray(o["C_k"])))
        return float(o["sum_rate"]) - lam * float((s ** 2).sum())
    J.qos = J.rtot = float("nan")
    J.rpriv, J.ckmean = None, 0.0
    return J


# AO's plateau setting, re-measured 2026-08-06 with probe_ao_saturate.py on 80
# paired states per case, grid extended to sweeps=3 (a grid that stops at the
# value you publish cannot show that value saturates).
#
#   K=5   (1,1,11) J 1.8231 @ 25.4ms  · best 1.8275 @ 53.4ms  (+0.0044, SE 0.0094)
#   K=10  (1,1,11) J 1.6400 @ 52.5ms  · best 1.6436 @ 194ms   (+0.0036, SE 0.0057)
#   K=15  (1,1,11) J 1.5745 @ 95.5ms  · best 1.5799 @ 382ms   (+0.0054, SE 0.0042)
#
# Every gain beyond sweeps=1 is at or inside the standard error while costing
# 2-4x the time, so sweeps=1 is the honest publish point everywhere.
#
# ⛔ FIXED: this used to pin K=15 to sweeps=2, on a note claiming sweeps=1 fell
# short of the plateau. The re-measurement says the opposite -- at fixed rounds,
# 1 -> 2 sweeps LOWERS J at K=15 (rounds 1: 1.5745 -> 1.5697; rounds 2: 1.5770
# -> 1.5730). The old setting therefore scored AO at a configuration that was
# both weaker and 1.6x slower, understating the baseline in every Case-3 number.
_AO_SWEEPS = {5: 1, 10: 1, 15: 1}          # by K


def score_one(run_dir, seed, n_states, ao_sweeps=None, own_ck=False):
    cfg = load_training_cfg(run_dir)
    env = ISTNEnv(cfg, seed=seed, n_steps_ep=n_states + 5)
    actor, phase_net, power_net, ck_net = load_agents(run_dir, seed=seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    Jt = make_J(env, cfg, True)
    sw = _AO_SWEEPS.get(cfg.K, 1) if ao_sweeps is None else int(ao_sweeps)
    ao = AORatePolicy(cfg, RateComputer(cfg), rounds=1, n_split=11,
                      phase_sweeps=sw)

    # Flat-PPO baseline: one network emits the whole action, so there is no z_t
    # and no phase/power/Ck nets. C_k is NOT taken from the actor here — make_J
    # applies the same oracle C_k rule to every policy and to AO, so the flat
    # arm is scored on exactly the footing as the hierarchical one.
    is_flat = phase_net is None
    flat_prev = None

    pj, pq, aj, aq = [], [], [], []
    env.reset(seed=seed)
    for _ in range(n_states):
        obs = env._get_obs()
        blocked = _compute_irs_favored(env)
        if is_flat:
            h_now = env.rate_computer.effective_channels_all(
                env.assignment, env.Phi, env.channels)
            s_t = actor.extract_state(obs, demand, blocked, h_eff=h_now,
                                      prev_rates=flat_prev, update_norm=False)
            act_f, _, fi = actor.forward(s_t, greedy=True)
            phi, ids = act_f["phi"], act_f["active_ids"]
            Phi_f = env.phase_model.build_phi(
                env.phase_model.index_to_phase(act_f["phase_idx"]))
            pj.append(Jt(phi, Phi_f, fi["w_p"], fi["w_c_vec"], ids))
            pq.append(Jt.qos)
            flat_prev = (Jt.rpriv, Jt.ckmean)
        else:
            pj.append(_score_hier(env, cfg, actor, phase_net, power_net,
                                  ck_net, obs, demand, blocked, Jt, own_ck))
            pq.append(Jt.qos)

        act = ao.act(obs, env)
        aid = sorted({int(x) for x in act["assignment"] if x > 0})
        aj.append(Jt(act["assignment"], phi_from_idx(act["phase_idx"], cfg),
                     act["w_p"], act["w_c_vec"], aid))
        aq.append(Jt.qos)

        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)
    return (float(np.mean(pj)), float(np.mean(pq)),
            float(np.mean(aj)), float(np.mean(aq)))


def _score_hier(env, cfg, actor, phase_net, power_net, ck_net, obs, demand,
                blocked, Jt, own_ck=False):
        s_t = actor.extract_state(obs, demand, blocked)
        phi, _, ainfo = actor.forward(s_t, greedy=True)
        rep = ainfo["z_t"]
        active, ids = _get_active_irs(phi), _get_active_irs_ids(phi)
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
        wc_p, wp_p, _, _ = power_net.forward(sp, ids)
        Ck = None
        if own_ck:
            part = env.rate_computer.compute_rates_partial(
                phi, Phi, env.channels, wp_p, wc_p, active_irs_ids=ids)
            s_ck = _build_ck_state(demand, part["R_private"],
                                   part["R_c_group"], phi, cfg)
            Ck, _, _ = ck_net.forward(s_ck, phi, part["R_c_group"])
        return Jt(phi, Phi, wp_p, wc_p, ids, Ck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="LABEL:run_dir ...")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 21])
    ap.add_argument("--states", type=int, default=100)
    ap.add_argument("--own-ck", dest="own_ck", action="store_true",
                    help="score the policy with its OWN CkMLP instead of the QoS-maximising oracle C_k that AO is locked to")
    ap.add_argument("--ao-sweeps", dest="ao_sweeps", type=int, default=None,
                    help="override AO phase_sweeps; default is the measured "
                         "per-case plateau (K=5,10 -> 1; K=15 -> 2)")
    a = ap.parse_args()

    specs = [s.split(":", 1) for s in a.runs]
    print("=" * 78)
    print(f"  POLICY vs AO · {a.states} paired states x {len(a.seeds)} env seeds "
          f"{a.seeds}")
    print("  decide on g-hat, score on true g · AO at its measured "
        "per-case J-plateau (phase_sweeps 1 for K=5/10, 2 for K=15)")
    print("=" * 78)

    tab, ao_all, ao_qos = {}, [], []
    for lbl, d in specs:
        per = []
        for s in a.seeds:
            pj, pq, aj, aq = score_one(d, s, a.states, a.ao_sweeps, a.own_ck)
            per.append((s, pj, pq, aj, aq))
            ao_all.append(aj)
            ao_qos.append(aq)
            print(f"    {lbl:<10} seed {s:<4} J {pj:.4f}  QoS {100*pq:5.1f}%"
                  f"   AO {aj:.4f} (QoS {100*aq:5.1f}%)  d {pj-aj:+.4f}",
                  flush=True)
        tab[lbl] = per

    print("\n" + "=" * 78)
    print(f"  {'policy':<12}{'J':>9}{'sd':>8}{'QoS':>8}{'vs AO':>10}{'sd':>8}")
    print("  " + "-" * 55)
    for lbl, per in tab.items():
        j = np.array([p[1] for p in per])
        q = np.array([p[2] for p in per])
        d = np.array([p[1] - p[3] for p in per])
        print(f"  {lbl:<12}{j.mean():>9.4f}{j.std(ddof=1) if len(j)>1 else 0:>8.4f}"
              f"{100*q.mean():>7.1f}%{d.mean():>+10.4f}"
              f"{d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else 0:>8.4f}")
    ao, aoq = np.array(ao_all), np.array(ao_qos)
    print(f"  {'AO':<12}{ao.mean():>9.4f}{ao.std(ddof=1):>8.4f}"
          f"{100*aoq.mean():>7.1f}%{'—':>10}")
    print("=" * 78)
    print("  'vs AO' sd is the standard error over env seeds. A difference between")
    print("  two policies smaller than that is not a ranking.")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
