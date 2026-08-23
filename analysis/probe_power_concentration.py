"""
probe_power_concentration.py
----------------------------
How concentrated is the allocation, for every method, as the power budget moves?

Companion to tab:abl_genshift. That table shows PPO-flat reporting R_tot ~10.2 at
P_S=70 in BOTH cases against ~2.3 for AO/DNN/VQC. The budget check already showed
no over-spending (total/budget = 1.0000), so the question is where the allocation
goes, not how much of it there is.

Two quantities per (method, P_S):

  * share of sum(w_p) held by the top-ranked user. This is well defined per user
    and is the DECISION. AO is structurally pinned at 1/K here because it splits
    private power equally (CSI/baselines: wp = full(K, P_S*f/K)).
  * share of R_tot held by the top-ranked user. This is the CONSEQUENCE, and it
    is the quantity tab:abl_genshift sums.

Per-user *total* power is deliberately NOT reported: the common stream is a
per-group quantity (w_c_vec has one entry per group), so attributing it to
individual users would need an arbitrary convention, and at P_S=70 the common
share is ~50% of the budget — half the figure would rest on that convention.

Usage:
  python analysis/probe_power_concentration.py --K 10 --M 2 --P 70 \
      --hier VQC=results/result_291/checkpoints/ep_13700 \
      --flat flat=results/result_402/checkpoints/ep_40008
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P
from params import make_config
from CSI.env import ISTNEnv
from RL import QuantumActor, ClassicalActor, PhaseMLP, PowerMLP, CkMLP
from train import (_build_phase_state, _build_ck_state, _get_active_irs,
                   _get_active_irs_ids, _build_power_state, _compute_irs_favored)
from analysis.probe_ao_rate import AORatePolicy


def _stats(W, R, K):
    W = np.vstack(W); R = np.vstack(R)
    wtop = np.sort(W, axis=1)[:, ::-1][:, 0]
    rtop = np.sort(R, axis=1)[:, ::-1][:, 0]
    wsum = W.sum(1); rsum = R.sum(1)
    ok = wsum > 1e-12

    # Lorenz curve of the private-power allocation: per state sort the shares
    # ascending and accumulate, then average the curves. Equal split is the
    # diagonal, so a curve's sag below it is the concentration.
    sh = W[ok] / wsum[ok][:, None]
    lor = np.cumsum(np.sort(sh, axis=1), axis=1).mean(0)      # (K,), ends at 1
    x = np.arange(1, K + 1) / K
    area = float(np.trapezoid(np.r_[0.0, lor], np.r_[0.0, x])
                 if hasattr(np, "trapezoid") else
                 np.trapz(np.r_[0.0, lor], np.r_[0.0, x]))

    return dict(w_top=100 * float(np.mean(wtop[ok] / wsum[ok])),
                r_top=100 * float(np.mean(rtop / rsum)),
                R_tot=float(rsum.mean()),
                equal=100.0 / K,
                w_lorenz=[float(v) for v in lor],
                gini=float(1.0 - 2.0 * area))


def run_hier(run_dir, cfg, seeds, n_steps):
    d = os.path.join(run_dir, "agents")
    is_q = os.path.isfile(os.path.join(d, "actor_config.json")) and \
        "n_qubits" in json.load(open(os.path.join(d, "actor_config.json")))
    actor = (QuantumActor if is_q else ClassicalActor).from_dir(d, seed=0)
    ph = PhaseMLP.from_dir(d, seed=0)
    pw = PowerMLP.from_dir(d, seed=0)
    ck = CkMLP.from_dir(d, seed=0)
    hp_dir = os.path.abspath(d)
    for _ in range(5):
        hp_dir = os.path.dirname(hp_dir)
        hp = os.path.join(hp_dir, "hyperparameters.json")
        if os.path.isfile(hp):
            pn = json.load(open(hp)).get("power_net", {})
            pw.power_fairness = float(pn.get("power_fairness", 0.0))
            pw.power_fairness_priv = float(pn.get("power_fairness_priv",
                                                  pn.get("power_fairness", 0.0)))
            pw.power_priv_frac = float(pn.get("power_priv_frac", 0.8))
            break
    W, R = [], []
    dem = np.full(cfg.K, cfg.D_k_bps_hz)
    for s in seeds:
        env = ISTNEnv(cfg=cfg, seed=s, n_steps_ep=n_steps,
                      reward_noise_avg=getattr(P, "reward_noise_avg", 1))
        obs = env.reset(seed=s)
        for _ in range(n_steps):
            s_t = actor.extract_state(obs, dem, _compute_irs_favored(env))
            phi, _, inf = actor.forward(s_t, greedy=True)
            rep = inf["z_t"]
            ai, aid = _get_active_irs(phi), _get_active_irs_ids(phi)
            pidx, _, _ = ph.forward(
                _build_phase_state(env.channels, phi, cfg, rep), ai, greedy=True)
            Phi = env.phase_model.build_phi(env.phase_model.index_to_phase(pidx))
            h = env.rate_computer.effective_channels_all(phi, Phi, env.channels)
            wc, wp, _, _ = pw.forward(_build_power_state(h, rep), aid)
            part = env.rate_computer.compute_rates_partial(
                phi, Phi, env.channels, wp, wc, active_irs_ids=aid)
            C_k, _, _ = ck.forward(
                _build_ck_state(dem, part["R_private"], part["R_c_group"], phi, cfg),
                phi, part["R_c_group"])
            obs, _, _, info = env.step({"assignment": phi, "phase_idx": pidx,
                                        "w_p": wp, "w_c_vec": wc, "C_k": C_k})
            W.append(np.asarray(wp, float))
            R.append(np.asarray(info["R_private"], float) + np.asarray(C_k, float))
    return _stats(W, R, cfg.K)


def run_flat(run_dir, cfg, seeds, n_steps):
    from RL.flat_actor import FlatActor
    fa = FlatActor.from_dir(os.path.join(run_dir, "agents"), seed=0)
    W, R = [], []
    dem = np.full(cfg.K, cfg.D_k_bps_hz)
    for s in seeds:
        env = ISTNEnv(cfg=cfg, seed=s, n_steps_ep=n_steps,
                      reward_noise_avg=getattr(P, "reward_noise_avg", 1))
        obs = env.reset(seed=s)
        prev = None
        for _ in range(n_steps):
            h_now = env.rate_computer.effective_channels_all(
                env.assignment, env.Phi, env.channels)
            st = fa.extract_state(obs, dem, _compute_irs_favored(env),
                                  h_eff=h_now, prev_rates=prev, update_norm=False)
            act, _, fi = fa.forward(st, greedy=True)
            Phi = env.phase_model.build_phi(
                env.phase_model.index_to_phase(act["phase_idx"]))
            part = env.rate_computer.compute_rates_partial(
                act["phi"], Phi, env.channels, fi["w_p"], fi["w_c_vec"],
                active_irs_ids=act["active_ids"])
            C_k, _, _, _ = fa.sample_ck(fi["logits"], act["phi"],
                                        part["R_c_group"], greedy=True)
            obs, _, _, info = env.step({"assignment": act["phi"],
                                        "phase_idx": act["phase_idx"],
                                        "w_p": fi["w_p"],
                                        "w_c_vec": fi["w_c_vec"], "C_k": C_k})
            Rp = np.asarray(info["R_private"], float)
            W.append(np.asarray(fi["w_p"], float))
            R.append(Rp + np.asarray(C_k, float))
            prev = (Rp, float(np.mean(C_k)))
    return _stats(W, R, cfg.K)


def run_ao(cfg, seeds, n_steps):
    W, R = [], []
    for s in seeds:
        env = ISTNEnv(cfg=cfg, seed=s, n_steps_ep=n_steps,
                      reward_noise_avg=getattr(P, "reward_noise_avg", 1))
        obs = env.reset(seed=s)
        pol = AORatePolicy(cfg, env.rate_computer, rounds=2, n_split=11,
                           route_ascent=True, phase_sweeps=1)
        for _ in range(n_steps):
            a = pol.act(obs, env)
            obs, _, _, info = env.step(a)
            W.append(np.asarray(a["w_p"], float))
            R.append(np.asarray(info["R_private"], float)
                     + np.asarray(a["C_k"], float))
    return _stats(W, R, cfg.K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--P", dest="ps", type=float, required=True)
    ap.add_argument("--hier", nargs="*", default=[], help="LABEL=run_dir")
    ap.add_argument("--flat", nargs="*", default=[], help="LABEL=run_dir")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--skip-ao", action="store_true")
    a = ap.parse_args()

    cfg = make_config(K=a.K, M=a.M, P_S_dBm=a.ps, R_LoS_km=0.5)
    print("K=%d M=%d P_S=%.0f dBm  %d seeds x %d steps   equal split = %.1f%%"
          % (cfg.K, cfg.M, a.ps, len(a.seeds), a.steps, 100.0 / cfg.K))
    print("  %-8s %10s %12s %10s" % ("method", "R_tot", "top-1 w_p%", "top-1 R%"))
    rows = {}
    if not a.skip_ao:
        rows["AO"] = run_ao(cfg, a.seeds, a.steps)
    for spec in a.flat:
        lbl, d = spec.split("=", 1)
        rows[lbl] = run_flat(d, cfg, a.seeds, a.steps)
    for spec in a.hier:
        lbl, d = spec.split("=", 1)
        rows[lbl] = run_hier(d, cfg, a.seeds, a.steps)
    for lbl, r in rows.items():
        print("  %-8s %10.4f %11.1f%% %9.1f%%"
              % (lbl, r["R_tot"], r["w_top"], r["r_top"]))
    src = {}
    for spec in list(a.flat) + list(a.hier):
        lbl, d = spec.split("=", 1)
        src[lbl] = d
    print("JSON " + json.dumps({"K": cfg.K, "M": cfg.M, "P_S": a.ps,
                                "seeds": a.seeds, "steps": a.steps,
                                "ckpt": src, "rows": rows}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
