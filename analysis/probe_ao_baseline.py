"""
probe_ao_baseline.py
--------------------
Per-state ALTERNATING-OPTIMIZATION (AO) baseline, in the spirit of the convex/AO
pipeline of Tan et al., assembled from the closed-form / greedy sub-solvers already
present in analysis/:

    routing (blocked -> its IRS)  ->  dominant phase  ->  power/Ck
                                  ->  multi-user phase -> power/Ck        (one round)

Unlike the trained policy, this solver runs the full optimization **for every channel
realization**. The probe therefore reports both the achieved quality (R_tot, QoS) and
the per-state SOLVE TIME, so the amortized-inference contrast is quantitative.

Actions are pushed through the same `env.step()` path used to evaluate the RL actors,
so quality numbers are directly comparable to `infer.py` results.

Usage:
  python analysis/probe_ao_baseline.py --run results/result_95 \
      --episodes 50 --steps 200 --seed 42 --rounds 1
"""
import sys, os, argparse, json
from time import perf_counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from infer import load_training_cfg
from params import make_config
from analysis.probe_assignment_quality import _oracle_assignment
from train import _irs_favored_target
from analysis.phase_oracle import oracle_phase_idx
from analysis.oracle_alloc import (phi_from_idx, _est_channels,
                                   multiuser_phase_idx, greedy_power_ck,
                                   oracle_ck_met)


def _cfg_from_run(run_dir, r_los=None):
    """Build cfg from the run's OWN hyperparameters.json. load_training_cfg walks up
    and can silently fall back to params.py defaults (wrong R_LoS); this does not."""
    hp = json.load(open(os.path.join(run_dir, 'hyperparameters.json')))
    s = hp['system']
    return make_config(K=s['K'], M=s['M'], N=s['N'], P_S_dBm=s['P_S_dBm'],
                       R_LoS_km=(s['R_LoS_km'] if r_los is None else r_los),
                       D_k_bps_hz=s['D_k_bps_hz'])


class AOPolicy:
    """Per-state alternating optimization over (assignment, phase, power, C_k)."""

    def __init__(self, cfg, rate, rounds=1, genie=False):
        self.cfg, self.rate, self.rounds = cfg, rate, rounds
        self.genie = genie
        self.Dk = cfg.D_k_bps_hz

    def act(self, obs, env):
        cfg, rate, Dk = self.cfg, self.rate, self.Dk
        ch = env.channels
        # (1) routing.
        #   fair  : decide from ESTIMATED CSI only -- the same information the
        #           learned policy receives (best IRS path vs direct, on g_hat).
        #   genie : uses the TRUE blockage flag su_blocked -> upper bound only.
        if self.genie:
            a = _oracle_assignment(ch, env.user_pos, env.irs_pos)
        else:
            tgt, msk = _irs_favored_target(env)
            a = np.where(msk > 0, tgt, 0).astype(int)
        active = sorted(set(int(x) for x in a if x > 0))
        # (2) initial phase: per-IRS dominant-user closed form
        pidx = oracle_phase_idx(_est_channels(ch), a, cfg)
        Phi = phi_from_idx(pidx, cfg)
        # (3) alternate power <-> phase
        _, wp, wc = greedy_power_ck(a, Phi, ch, cfg, rate, active, Dk)
        for _ in range(self.rounds):
            pidx = multiuser_phase_idx(a, ch, cfg, rate, active, Dk, wp, wc)
            Phi = phi_from_idx(pidx, cfg)
            _, wp, wc = greedy_power_ck(a, Phi, ch, cfg, rate, active, Dk)
        # (4) common-rate split: greedy demand fill
        part = rate.compute_rates_partial(np.asarray(a), Phi, ch, wp, wc,
                                          active_irs_ids=active)
        _, C_k = oracle_ck_met(part['R_private'], part['R_c_group'],
                               part['groups'], Dk)
        return {'assignment': a, 'phase_idx': pidx,
                'w_p': wp, 'w_c_vec': wc, 'C_k': C_k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run dir, used only for its cfg/case')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rounds', type=int, default=1, help='phase<->power AO rounds')
    ap.add_argument('--R-LoS', dest='r_los', type=float, default=None,
                    help='override coverage radius (km); default = run native')
    ap.add_argument('--genie', action='store_true', help='use TRUE blockage for routing (upper bound)')
    a = ap.parse_args()

    cfg = _cfg_from_run(a.run, a.r_los)
    rate = RateComputer(cfg)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.steps)
    pol = AOPolicy(cfg, rate, rounds=a.rounds, genie=a.genie)

    print("=" * 74)
    print(f"  AO BASELINE (per-state solve) · K={cfg.K} M={cfg.M} "
          f"R_LoS={cfg.R_LoS_km} · rounds={a.rounds} · routing={'GENIE' if a.genie else 'fair(est-CSI)'}")
    print(f"  {a.episodes} ep x {a.steps} steps, seed {a.seed}")
    print("=" * 74)

    solve_ms, srate, qos = [], [], []
    for ep in range(a.episodes):
        obs = env.reset()
        for _ in range(a.steps):
            t0 = perf_counter()
            action = pol.act(obs, env)
            solve_ms.append((perf_counter() - t0) * 1e3)
            obs, _r, _d, info = env.step(action)
            srate.append(float(info['sum_rate']))
            qos.append(float(np.sum(info['R_tot'] >= cfg.D_k_bps_hz)) / cfg.K)
        if (ep + 1) % max(1, a.episodes // 5) == 0:
            print(f"    ep {ep+1}/{a.episodes}: R_tot={np.mean(srate):.4f} "
                  f"QoS={100*np.mean(qos):.1f}% solve={np.mean(solve_ms):.1f} ms",
                  flush=True)

    solve_ms = np.array(solve_ms)
    print("-" * 74)
    print(f"  R_tot           : {np.mean(srate):.4f}")
    print(f"  QoS             : {100*np.mean(qos):.1f}%")
    print(f"  solve time/state: {solve_ms.mean():.1f} ms  "
          f"(median {np.median(solve_ms):.1f}, p95 {np.percentile(solve_ms,95):.1f})")
    print("=" * 74)


if __name__ == '__main__':
    main()
