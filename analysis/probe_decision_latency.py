"""
probe_decision_latency.py
-------------------------
Wall-clock cost of producing ONE complete action, for every method that appears
in the main tables: AO, the DNN pipeline, and the VQC pipeline.

Differs from probe_inference_latency.py, which times only `actor.forward` (the
assignment head). That isolates the VQC-vs-DNN core, but it is not what a
deployment pays: the deployed decision runs assignment -> phase -> power -> Ck,
and the chaining itself costs two rate-kernel evaluations (`effective_channels_all`
feeds the power head, `compute_rates_partial` feeds the Ck head). Those belong to
the policy's bill. This probe mirrors infer.py's loop exactly and times the whole
block.

Protocol follows Li et al. (arXiv:2204.13372), who state machine, common
initialisation and an explicit stopping rule for every algorithm compared:

  * one thread   -- OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1, no GPU
  * same seed    -- every method replays the same env seed
  * AO stopping  -- reported at the cheapest setting reaching its J plateau
                    (rounds=1, phase_sweeps=1, n_split=11; see ao_sat_*.log),
                    NOT at a deeper setting whose extra time buys no J, which
                    would inflate the baseline's cost and our own advantage
  * median [IQR] -- not mean: one scheduler preemption ruins a mean

Also tallies rate-kernel evaluations per decision. That count is a property of
the algorithm, so unlike the milliseconds it survives any reimplementation --
the objection Sun et al. (arXiv:1705.09412) documented in their own Table III,
where a 127x DNN speedup over Python WMMSE fell to 1.68x over the same algorithm
in C.

Usage (machine MUST be otherwise idle):
  python analysis/probe_decision_latency.py \
      --dnn C1:results/result_234 C2:results/result_232 \
      --vqc C1:results/result_260 C2:results/result_261 \
      --cases C1:5,1 C2:10,2 C3:15,3 --n-time 300
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from infer import (load_training_cfg, load_agents, _compute_irs_favored,
                   _get_active_irs, _get_active_irs_ids, _build_phase_state,
                   _build_ck_state)
from analysis.probe_ao_rate import AORatePolicy


class KernelCounter:
    """Wraps a RateComputer so every objective-level call is tallied.

    Gated by `on`, because env.step() scores the chosen action through the same
    RateComputer. Those calls are the simulator's, not the decision's -- leaving
    the counter open across the whole loop credits every method with the env's
    own evaluations and makes the policy look like it searches when it does not.
    """

    def __init__(self, rate):
        self.rate, self.n, self.on = rate, 0, False
        for name in ("compute_sum_rate", "compute_rates_partial",
                     "effective_channels_all"):
            orig = getattr(rate, name)

            def wrap(*a, _o=orig, **kw):
                if self.on:
                    self.n += 1
                return _o(*a, **kw)
            setattr(rate, name, wrap)


def time_policy(run_dir, seed, n_warm, n_time):
    """Time the full four-head decision, exactly as infer.py assembles it."""
    cfg = load_training_cfg(run_dir)
    env = ISTNEnv(cfg, seed=seed, n_steps_ep=n_warm + n_time + 10)
    actor, phase_net, power_net, ck_net = load_agents(run_dir, seed=seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    ctr = KernelCounter(env.rate_computer)

    # PPO-flat emits the whole action from one network, so there are no
    # phase/power/Ck nets to chain. Timing it through the hierarchical path
    # would crash; timing only its forward would omit the rate-feedback state
    # and the C_k draw, both of which infer.py performs per decision.
    is_flat = phase_net is None
    flat_prev = None

    obs, ts = env.reset(seed=seed), []
    for i in range(n_warm + n_time):
        if i == n_warm:
            ctr.n = 0                                  # count the timed span only
        ctr.on = True
        t0 = time.perf_counter()

        blocked = _compute_irs_favored(env)
        if is_flat:
            h_now = env.rate_computer.effective_channels_all(
                env.assignment, env.Phi, env.channels)
            s_t = actor.extract_state(obs, demand, blocked, h_eff=h_now,
                                      prev_rates=flat_prev, update_norm=False)
            act_f, _, fi = actor.forward(s_t, greedy=True)
            phi, phase_idx = act_f["phi"], act_f["phase_idx"]
            w_p, w_c_vec = fi["w_p"], fi["w_c_vec"]
            Phi = env.phase_model.build_phi(
                env.phase_model.index_to_phase(phase_idx))
            part = env.rate_computer.compute_rates_partial(
                phi, Phi, env.channels, w_p, w_c_vec,
                active_irs_ids=act_f["active_ids"])
            C_k, _, _, _ = actor.sample_ck(fi["logits"], phi,
                                           part["R_c_group"], greedy=True)
            action = {"assignment": phi, "phase_idx": phase_idx, "w_p": w_p,
                      "w_c_vec": w_c_vec, "C_k": C_k}
            dt = time.perf_counter() - t0
            ctr.on = False
            if i >= n_warm:
                ts.append(dt)
            obs, _r, _d, info = env.step(action)
            flat_prev = (np.asarray(info["R_private"]),
                         float(np.mean(C_k)))
            continue

        s_t = actor.extract_state(obs, demand, blocked)
        phi, _, ainfo = actor.forward(s_t, greedy=True)
        z_t = ainfo["z_t"]
        rep = ainfo.get("o_hat", z_t) if getattr(actor, "NO_AE", False) else z_t
        active_irs, active_ids = _get_active_irs(phi), _get_active_irs_ids(phi)

        s_phase = _build_phase_state(env.channels, phi, cfg, rep)
        phase_idx, _, _ = phase_net.forward(s_phase, active_irs, greedy=True)
        Phi = env.phase_model.build_phi(env.phase_model.index_to_phase(phase_idx))

        h_eff = env.rate_computer.effective_channels_all(phi, Phi, env.channels)
        _hs = float(np.mean(np.abs(h_eff))) + 1e-9
        s_power = np.concatenate([h_eff.real / _hs, h_eff.imag / _hs])
        if getattr(power_net, "scale_feats", False):
            from train import _power_scale_feats
            s_power = np.concatenate([s_power, _power_scale_feats(h_eff, cfg)])
        if power_net.d_s > s_power.shape[0]:
            s_power = np.concatenate([s_power, rep])
        w_c_vec, w_p, _, _ = power_net.forward(s_power, active_ids)

        part = env.rate_computer.compute_rates_partial(
            phi, Phi, env.channels, w_p, w_c_vec, active_irs_ids=active_ids)
        s_ck = _build_ck_state(demand, part["R_private"], part["R_c_group"],
                               phi, cfg)
        C_k, _, _ = ck_net.forward(s_ck, phi, part["R_c_group"])
        action = {"assignment": phi, "phase_idx": phase_idx, "w_p": w_p,
                  "w_c_vec": w_c_vec, "C_k": C_k}

        dt = time.perf_counter() - t0
        ctr.on = False
        if i >= n_warm:
            ts.append(dt)
        obs, _r, _d, _i = env.step(action)

    tn = type(actor).__name__
    mode = "flat" if is_flat else ("DNN" if "Classical" in tn else "VQC")
    return np.array(ts) * 1e3, ctr.n / float(n_time), cfg, mode


def time_ao(K, M, seed, n_warm, n_time, rounds, sweeps, n_split):
    cfg = make_config(K=K, M=M)
    rate = RateComputer(cfg)
    ctr = KernelCounter(rate)
    env = ISTNEnv(cfg, seed=seed, n_steps_ep=n_warm + n_time + 10)
    pol = AORatePolicy(cfg, rate, rounds=rounds, n_split=n_split,
                       phase_sweeps=sweeps)

    obs, ts = env.reset(seed=seed), []
    for i in range(n_warm + n_time):
        if i == n_warm:
            ctr.n = 0
        ctr.on = True
        t0 = time.perf_counter()
        action = pol.act(obs, env)
        dt = time.perf_counter() - t0
        ctr.on = False
        if i >= n_warm:
            ts.append(dt)
        obs, _r, _d, _i = env.step(action)
    return np.array(ts) * 1e3, ctr.n / float(n_time)


def fmt(ts):
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    return med, q3 - q1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dnn", nargs="*", default=[], help="LABEL:run_dir ...")
    ap.add_argument("--vqc", nargs="*", default=[], help="LABEL:run_dir ...")
    ap.add_argument("--flat", nargs="*", default=[], help="LABEL:run_dir ...")
    ap.add_argument("--cases", nargs="+", default=["C1:5,1", "C2:10,2",
                                                  "C3:15,3"])
    ap.add_argument("--n-warm", dest="n_warm", type=int, default=30)
    ap.add_argument("--n-time", dest="n_time", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ao-rounds", dest="ao_r", type=int, default=1)
    ap.add_argument("--ao-sweeps", dest="ao_s", type=int, default=1)
    ap.add_argument("--ao-split", dest="ao_n", type=int, default=11)
    a = ap.parse_args()

    print("=" * 84)
    print(f"  DECISION LATENCY — full four-head action · warm={a.n_warm} "
          f"timed={a.n_time} · seed {a.seed}")
    print(f"  AO at its J-plateau setting: rounds={a.ao_r} "
          f"phase_sweeps={a.ao_s} n_split={a.ao_n}")
    print(f"  threads: OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS','?')} "
          f"OMP={os.environ.get('OMP_NUM_THREADS','?')} "
          f"CUDA={os.environ.get('CUDA_VISIBLE_DEVICES','?')!r}")
    print("=" * 84)
    print(f"  {'case':<6}{'method':<8}{'K,M':<8}{'median ms':>11}{'IQR':>9}"
          f"{'kernel evals':>14}{'dec/core-s':>12}")
    print("  " + "-" * 78)

    runs = {}
    for spec in a.dnn:
        lbl, d = spec.split(":", 1)
        runs.setdefault(lbl, {})["DNN"] = d
    for spec in a.vqc:
        lbl, d = spec.split(":", 1)
        runs.setdefault(lbl, {})["VQC"] = d
    for spec in a.flat:
        lbl, d = spec.split(":", 1)
        runs.setdefault(lbl, {})["flat"] = d

    table = {}
    for spec in a.cases:
        lbl, km = spec.split(":", 1)
        K, M = (int(x) for x in km.split(","))
        ts, nev = time_ao(K, M, a.seed, a.n_warm, a.n_time, a.ao_r, a.ao_s,
                          a.ao_n)
        med, iqr = fmt(ts)
        table.setdefault(lbl, {})["AO"] = med
        print(f"  {lbl:<6}{'AO':<8}{f'{K},{M}':<8}{med:>11.3f}{iqr:>9.3f}"
              f"{nev:>14.0f}{1000.0/med:>12.1f}", flush=True)

        for mode in ("DNN", "VQC", "flat"):
            d = runs.get(lbl, {}).get(mode)
            if d is None or not os.path.isdir(os.path.join(d, "agents")):
                print(f"  {lbl:<6}{mode:<8}{f'{K},{M}':<8}"
                      f"{'-- no checkpoint --':>34}", flush=True)
                continue
            ts, nev, cfg, got = time_policy(d, a.seed, a.n_warm, a.n_time)
            if got != mode:
                print(f"  {lbl:<6}{mode:<8}{'':8}"
                      f"{'!! run is ' + got + ', skipped':>34}", flush=True)
                continue
            med, iqr = fmt(ts)
            table.setdefault(lbl, {})[mode] = med
            print(f"  {lbl:<6}{mode:<8}{f'{cfg.K},{cfg.M}':<8}{med:>11.3f}"
                  f"{iqr:>9.3f}{nev:>14.1f}{1000.0/med:>12.1f}", flush=True)

    print("=" * 84)
    print("  Ratio vs AO (median):")
    for lbl, d in table.items():
        if "AO" not in d:
            continue
        parts = [f"AO/{m} = {d['AO']/d[m]:6.1f}x" for m in ("DNN", "VQC")
                 if m in d]
        if parts:
            print(f"    {lbl}:  " + "   ".join(parts))
    print("=" * 84)
    print("  CAVEATS to carry into the paper:")
    print("   * VQC timing is a CLASSICAL STATEVECTOR SIMULATION. Real quantum")
    print("     hardware at n_shots_eval would be far slower; this is not a")
    print("     hardware-latency claim.")
    print("   * Both pipelines are NumPy and dispatch-bound at these sizes, so")
    print("     the ratio is not an artefact of one side being compiled.")
    print("   * Kernel-eval counts are language-free; the milliseconds are not.")
    print("=" * 84)


if __name__ == "__main__":
    main()
