"""
probe_ao_rate.py
----------------
Per-state alternating optimization that maximizes the PAPER'S objective

        J = sum_rate  -  lambda_D * sum_k ( max(0, D_k - R_tot_k) / D_k )^2

rather than the number of QoS-met users. This is the fair per-state-solver baseline
for a sum-rate-maximization paper (probe_ao_baseline.py maximizes #met instead, which
over-delivers QoS and under-delivers rate).

Three differences from the #met version:
  1. every sub-step is scored by J, not by #met;
  2. the leftover common-rate budget is actually spent (oracle_ck_met deliberately
     leaves it unallocated as "QoS-irrelevant slack" -- but it is pure sum-rate);
  3. routing gets a coordinate-ascent pass on J (this is the combinatorial part that
     Tan et al. attack with simulated annealing).

All decisions are made on the ESTIMATED channel (design view, use_true=False), the
same information the learned policy receives.

Usage:
  python analysis/probe_ao_rate.py --run results/result_117 --R-LoS 0.5 \
      --episodes 50 --steps 200 --seed 42
"""
import sys, os, json, argparse
from time import perf_counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from params import make_config
from train import _irs_favored_target
from analysis.phase_oracle import oracle_phase_idx, oracle_phase_search
from analysis.oracle_alloc import phi_from_idx, _est_channels, oracle_ck_met


def _cfg_from_run(run_dir, r_los=None):
    hp = json.load(open(os.path.join(run_dir, 'hyperparameters.json')))
    s = hp['system']
    return make_config(K=s['K'], M=s['M'], N=s['N'], P_S_dBm=s['P_S_dBm'],
                       R_LoS_km=(s['R_LoS_km'] if r_los is None else r_los),
                       D_k_bps_hz=s['D_k_bps_hz'])


class AORatePolicy:
    def __init__(self, cfg, rate, rounds=2, n_split=11, route_ascent=True,
                 phase_sweeps=1):
        self.cfg, self.rate = cfg, rate
        self.rounds, self.route_ascent = rounds, route_ascent
        self.phase_sweeps = phase_sweeps
        self.Dk = cfg.D_k_bps_hz
        self.lamD = float(getattr(cfg, 'lambda_D', 1.5))
        self.eps = float(getattr(cfg, 'epsilon_qp', 1e-3))
        self.L = len(cfg.phase_levels)
        self.splits = np.linspace(0.05, 0.95, n_split)

    # ── objective ────────────────────────────────────────────────────────────
    def _obj(self, a, Phi, ch, wp, wc, Ck, active):
        out = self.rate.compute_sum_rate(np.asarray(a), Phi, ch, wp, wc,
                                         C_k=Ck, active_irs_ids=active)
        Rt = np.asarray(out['R_private']) + np.asarray(out['C_k'])
        short = np.maximum(0.0, self.Dk - Rt) / (self.Dk + self.eps)
        return float(out['sum_rate']) - self.lamD * float((short ** 2).sum())

    # ── C_k: meet D_k first, then SPEND the leftover common budget ───────────
    def _ck(self, a, Phi, ch, wp, wc, active):
        part = self.rate.compute_rates_partial(np.asarray(a), Phi, ch, wp, wc,
                                               active_irs_ids=active)
        Rp, Rcg, groups = part['R_private'], part['R_c_group'], part['groups']
        _, Ck = oracle_ck_met(Rp, Rcg, groups, self.Dk)
        for gid, mem in groups.items():
            mem = np.asarray(list(mem), dtype=int)
            if mem.size == 0:
                continue
            left = float(Rcg.get(int(gid), 0.0)) - float(Ck[mem].sum())
            if left > 1e-12:
                Ck[mem] += left / mem.size          # split is sum-rate-neutral
        return Ck

    def _best_power(self, a, Phi, ch, active):
        K, P_S = self.cfg.K, self.cfg.P_S
        best, arg = -np.inf, None
        for f in self.splits:
            wp = np.full(K, P_S * f / K)
            wc = np.full(len(active) + 1, P_S * (1.0 - f) / (len(active) + 1))
            Ck = self._ck(a, Phi, ch, wp, wc, active)
            o = self._obj(a, Phi, ch, wp, wc, Ck, active)
            if o > best:
                best, arg = o, (wp, wc, Ck)
        return arg

    def act(self, obs, env):
        cfg, ch = self.cfg, env.channels
        tgt, msk = _irs_favored_target(env)
        a = np.where(msk > 0, tgt, 0).astype(int)
        active = sorted(set(int(x) for x in a if x > 0))
        pidx = np.asarray(oracle_phase_idx(_est_channels(ch), a, cfg), dtype=int)
        wp, wc, Ck = self._best_power(a, phi_from_idx(pidx, cfg), ch, active)

        for _ in range(self.rounds):
            # (a) phase: PER-ELEMENT search on J, warm-started from the current
            #     pidx (so successive rounds refine rather than restart).
            #     ⚠ was `pidx[m-1,:] = lv` — one uniform level per IRS, scored
            #     without ever evaluating the per-element seed, so it DISCARDED
            #     the closed-form alignment every round. Correct only under the
            #     old scalar g_RU.
            def _score_J(ph):
                Ph = phi_from_idx(ph, cfg)
                Ck2 = self._ck(a, Ph, ch, wp, wc, active)
                return self._obj(a, Ph, ch, wp, wc, Ck2, active)

            pidx, _, _ = oracle_phase_search(_est_channels(ch), a, cfg, _score_J,
                                             n_sweeps=self.phase_sweeps,
                                             seed_pidx=pidx)
            Phi = phi_from_idx(pidx, cfg)
            # (b) power/Ck re-solve under the new phase
            wp, wc, Ck = self._best_power(a, Phi, ch, active)
            # (c) routing: coordinate ascent on J (the combinatorial step)
            if self.route_ascent:
                for k in range(cfg.K):
                    best, bc = -np.inf, int(a[k])
                    for c in range(cfg.M + 1):
                        trial = a.copy(); trial[k] = c
                        act2 = sorted(set(int(x) for x in trial if x > 0))
                        wc2 = np.full(len(act2) + 1, float(wc.sum()) / (len(act2) + 1))
                        Ck2 = self._ck(trial, Phi, ch, wp, wc2, act2)
                        o = self._obj(trial, Phi, ch, wp, wc2, Ck2, act2)
                        if o > best:
                            best, bc = o, c
                    a[k] = bc
                active = sorted(set(int(x) for x in a if x > 0))
                wp, wc, Ck = self._best_power(a, Phi, ch, active)

        Phi = phi_from_idx(pidx, cfg)
        Ck = self._ck(a, Phi, ch, wp, wc, active)
        return {'assignment': a, 'phase_idx': pidx,
                'w_p': wp, 'w_c_vec': wc, 'C_k': Ck}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--R-LoS', dest='r_los', type=float, default=None)
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--no-route-ascent', dest='route', action='store_false')
    ap.add_argument('--phase-sweeps', dest='phase_sweeps', type=int, default=1,
                    help='per-element phase coordinate-ascent sweeps per round '
                         '(0 = seed + uniform restarts only — much faster, but a '
                         'weaker phase solution)')
    a = ap.parse_args()

    cfg = _cfg_from_run(a.run, a.r_los)
    rate = RateComputer(cfg)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.steps)
    pol = AORatePolicy(cfg, rate, rounds=a.rounds, route_ascent=a.route,
                       phase_sweeps=a.phase_sweeps)

    print("=" * 74)
    print(f"  AO (sum-rate objective) · K={cfg.K} M={cfg.M} R_LoS={cfg.R_LoS_km} "
          f"P_S={cfg.P_S_dBm}dBm · rounds={a.rounds} route-ascent={a.route} "
          f"phase-sweeps={a.phase_sweeps}")
    print("=" * 74)

    ms, sr, qos = [], [], []
    for ep in range(a.episodes):
        obs = env.reset()
        for _ in range(a.steps):
            t0 = perf_counter(); action = pol.act(obs, env)
            ms.append((perf_counter() - t0) * 1e3)
            obs, _r, _d, info = env.step(action)
            sr.append(float(info['sum_rate']))
            qos.append(float(np.sum(info['R_tot'] >= cfg.D_k_bps_hz)) / cfg.K)
        if (ep + 1) % max(1, a.episodes // 5) == 0:
            print(f"    ep {ep+1}/{a.episodes}: R_tot={np.mean(sr):.4f} "
                  f"QoS={100*np.mean(qos):.1f}% solve={np.mean(ms):.1f} ms", flush=True)

    ms = np.array(ms)
    print("-" * 74)
    print(f"  R_tot           : {np.mean(sr):.4f}")
    print(f"  QoS             : {100*np.mean(qos):.1f}%")
    print(f"  solve time/state: {ms.mean():.1f} ms (median {np.median(ms):.1f})")
    print("=" * 74)


if __name__ == '__main__':
    main()
