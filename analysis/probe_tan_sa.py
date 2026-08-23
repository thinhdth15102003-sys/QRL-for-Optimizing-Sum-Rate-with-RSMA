"""
analysis/probe_tan_sa.py
------------------------
Reproduce Tan et al.'s IRS-user selection (their Algorithm 4, simulated
annealing) and compare it against our coordinate-ascent routing, to quantify how
much of the AO-vs-Tan sum-rate difference is explained by the ROUTING search.

Why only this piece of their pipeline is reproduced
---------------------------------------------------
Their Algorithm 1 lifts the precoder to W = w·w^H and enforces Rank(W) ≤ 1 (SDR
+ rank-1 extraction). Under the single-antenna, no-transmit-beamforming
assumption of THIS work, w is a SCALAR, so W is 1x1, rank-1 holds automatically
and the relaxation is exact — there is no relaxation gap to reproduce, and an
SDP solver would return what a direct power search already returns. Likewise
Algorithm 2 (RMCG) solves the CONTINUOUS unit-modulus phase; we measured 2-bit
vs 6-bit quantisation to be worth <0.001 bps/Hz here, so that is not the gap
either. Algorithm 4 is the one component whose search quality we can vary and
measure.

Their Algorithm 4 (as written in the paper):
    phi^(0) random; beta = change rate; gamma_f = reduction factor
    repeat:
        k = mod(q, K) + 1;  phi^new = phi^q with phi_k flipped   (single-bit flip)
        evaluate R_tot^new via Algorithm 3
        if R_tot^new > R_tot^old: accept
        else: accept with probability P_cha = exp((R_new - R_old)/beta)   (eq 54)
        q += 1;  beta = gamma_f * beta
    until converged (N_eps5 iterations)

Note phi_k is BINARY there (eq 15b/53b), i.e. "user k uses the IRS or not",
which matches Case 1 (M=1). For M>1 the flip is generalised to a random IRS id.

Everything else (phase = per-element oracle, power = (beta,f) grid, C_k =
demand-fill) is held IDENTICAL across the compared routing searches, so the only
difference measured is the routing search itself. Scored like env.step
(decide on ĝ, score on true g, reward_noise_avg σ² draws).

Usage:
  python analysis/probe_tan_sa.py --K 5 --M 1 --iters 20
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import numpy as np
import params as P
from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.phase_oracle import oracle_phase_idx, est_channels
from analysis.oracle_alloc import oracle_ck_met, phi_from_idx

ap = argparse.ArgumentParser()
ap.add_argument('--K', type=int, default=5)
ap.add_argument('--M', type=int, default=1)
ap.add_argument('--P', type=float, default=50.0)
ap.add_argument('--R-LoS', dest='rlos', type=float, default=0.5)
ap.add_argument('--episodes', type=int, default=2)
ap.add_argument('--steps', type=int, default=15)
ap.add_argument('--iters', type=int, default=20,
                help="Tan Fig.4 x-axis runs to ~20 iterations")
ap.add_argument('--beta0', type=float, default=0.5, help='SA initial change rate')
ap.add_argument('--gamma-f', dest='gf', type=float, default=0.9, help='SA reduction factor')
ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()

cfg  = make_config(K=a.K, M=a.M, P_S_dBm=a.P, R_LoS_km=a.rlos)
rate = RateComputer(cfg)
NAVG = int(getattr(P, 'reward_noise_avg', 16))
env  = ISTNEnv(cfg=cfg, seed=a.seed, n_steps_ep=a.steps, reward_noise_avg=NAVG)
Dk   = cfg.D_k_bps_hz
BET  = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SPL  = np.linspace(0.05, 0.95, 11)
rng  = np.random.default_rng(a.seed)


def inner(assign, ch):
    """Algorithm 1+2+3 stand-in, identical for every routing search:
    per-element oracle phase, (beta,f) power grid, demand-fill C_k.
    Returns the best DESIGN-view sum-rate for this routing."""
    assign = np.asarray(assign, dtype=int)
    if not np.any(assign > 0):
        act, G = [], 0
        Phi = np.zeros((cfg.M, cfg.N, cfg.N), dtype=complex)
        di = np.arange(cfg.N); Phi[:, di, di] = 1.0
    else:
        Phi = phi_from_idx(oracle_phase_idx(est_channels(ch), assign, cfg), cfg)
        act = sorted(set(int(x) for x in assign if x > 0)); G = len(act)
    g = np.abs(rate.effective_channels_all(assign, Phi, ch)) ** 2 + 1e-30
    best = -np.inf
    for b in BET:
        w = g ** b
        for f in SPL:
            wp = w / w.sum() * (cfg.P_S * f)
            wc = np.full(G + 1, cfg.P_S * (1 - f) / (G + 1))
            part = rate.compute_rates_partial(assign, Phi, ch, wp, wc,
                                              active_irs_ids=act)
            _, Ck = oracle_ck_met(part['R_private'], part['R_c_group'],
                                  part['groups'], Dk)
            for gid, mem in part['groups'].items():
                mem = np.asarray(list(mem), dtype=int)
                if mem.size:
                    left = float(part['R_c_group'].get(int(gid), 0.0)) - float(Ck[mem].sum())
                    if left > 1e-12:
                        Ck[mem] += left / mem.size
            o = rate.compute_sum_rate(assign, Phi, ch, wp, wc, C_k=Ck,
                                      active_irs_ids=act, sigma2=cfg.sigma2)
            Rt = np.asarray(o['R_private']) + np.asarray(o['C_k'])
            # Tan (49d)/(50b): the min-rate constraint holds for EVERY user.
            # Without it the search runs to the rate-heavy corner (R_tot 5.26 at
            # QoS 20%) and every routing scores the same, which tells us nothing.
            short = float(np.sum(np.maximum(0.0, Dk - Rt) / Dk))
            best = max(best, float(o['sum_rate']) - 10.0 * short)
    return best


def achieved(assign, ch, sigmas):
    """Score the final routing on the TRUE channel, like env.step."""
    assign = np.asarray(assign, dtype=int)
    if not np.any(assign > 0):
        act, G = [], 0
        Phi = np.zeros((cfg.M, cfg.N, cfg.N), dtype=complex)
        di = np.arange(cfg.N); Phi[:, di, di] = 1.0
    else:
        Phi = phi_from_idx(oracle_phase_idx(est_channels(ch), assign, cfg), cfg)
        act = sorted(set(int(x) for x in assign if x > 0)); G = len(act)
    g = np.abs(rate.effective_channels_all(assign, Phi, ch)) ** 2 + 1e-30
    bestJ, keep = -np.inf, None
    for b in BET:
        w = g ** b
        for f in SPL:
            wp = w / w.sum() * (cfg.P_S * f)
            wc = np.full(G + 1, cfg.P_S * (1 - f) / (G + 1))
            part = rate.compute_rates_partial(assign, Phi, ch, wp, wc, active_irs_ids=act)
            _, Ck = oracle_ck_met(part['R_private'], part['R_c_group'], part['groups'], Dk)
            for gid, mem in part['groups'].items():
                mem = np.asarray(list(mem), dtype=int)
                if mem.size:
                    left = float(part['R_c_group'].get(int(gid), 0.0)) - float(Ck[mem].sum())
                    if left > 1e-12:
                        Ck[mem] += left / mem.size
            o = rate.compute_sum_rate(assign, Phi, ch, wp, wc, C_k=Ck,
                                      active_irs_ids=act, sigma2=cfg.sigma2)
            Rt = np.asarray(o['R_private']) + np.asarray(o['C_k'])
            short = float(np.sum(np.maximum(0.0, Dk - Rt) / Dk))
            val = float(o['sum_rate']) - 10.0 * short   # same rule as inner()
            if val > bestJ:
                bestJ, keep = val, (wp, wc, Ck)
    wp, wc, Ck = keep
    sr, rt = 0.0, np.zeros(cfg.K)
    for s2 in sigmas:
        o = rate.compute_sum_rate(assign, Phi, ch, wp, wc, C_k=Ck,
                                  active_irs_ids=act, sigma2=s2, use_true=True)
        sr += float(o['sum_rate'])
        rt += np.asarray(o['R_private']) + np.asarray(o['C_k'])
    n = len(sigmas)
    return sr / n, float(np.mean(rt / n >= Dk))


# ── routing searches ─────────────────────────────────────────────────────────

def route_tan_sa(ch, iters):
    """Tan Algorithm 4: single-bit flip + Metropolis accept, cooling beta."""
    phi = rng.integers(0, cfg.M + 1, size=cfg.K)
    R_old = inner(phi, ch)
    best, R_best = phi.copy(), R_old
    beta = a.beta0
    for q in range(iters):
        k = q % cfg.K                                   # mod(q,K)+1 in 1-based
        new = phi.copy()
        if cfg.M == 1:
            new[k] = 1 - new[k]                         # binary flip (eq 53b)
        else:
            alt = [c for c in range(cfg.M + 1) if c != phi[k]]
            new[k] = alt[rng.integers(len(alt))]
        R_new = inner(new, ch)
        if R_new > R_old:
            phi, R_old = new, R_new
            if R_new > R_best:
                best, R_best = new.copy(), R_new
        else:
            # eq (54): P_cha = exp((R_new - R_old)/beta)
            if rng.random() < np.exp((R_new - R_old) / max(beta, 1e-12)):
                phi, R_old = new, R_new
        beta *= a.gf
    return best


def route_coord_ascent(ch, rounds=2):
    """Our AO routing: full coordinate ascent over all M+1 choices per user."""
    phi = np.zeros(cfg.K, dtype=int)
    cur = inner(phi, ch)
    for _ in range(rounds):
        changed = False
        for k in range(cfg.K):
            b0 = int(phi[k])
            for c in range(cfg.M + 1):
                if c == b0:
                    continue
                phi[k] = c
                v = inner(phi, ch)
                if v > cur:
                    cur, b0, changed = v, c, True
                phi[k] = b0
            phi[k] = b0
        if not changed:
            break
    return phi


def route_exhaustive(ch):
    """Ground truth (only tractable for small (M+1)^K)."""
    best, R = None, -np.inf
    for c in itertools.product(range(cfg.M + 1), repeat=cfg.K):
        v = inner(np.array(c, dtype=int), ch)
        if v > R:
            R, best = v, np.array(c, dtype=int)
    return best


n_comb = (cfg.M + 1) ** cfg.K
do_exh = n_comb <= 4096
rows = {k: [] for k in ['tan_sa', 'coord', 'exh']}
for ep in range(a.episodes):
    env.reset(seed=a.seed + ep)
    for _ in range(a.steps):
        ch = env.channels
        sigmas = [env.channel_model.sample_noise_sigma2() for _ in range(NAVG)]
        rows['tan_sa'].append(achieved(route_tan_sa(ch, a.iters), ch, sigmas))
        rows['coord'].append(achieved(route_coord_ascent(ch), ch, sigmas))
        if do_exh:
            rows['exh'].append(achieved(route_exhaustive(ch), ch, sigmas))
        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)

print('=' * 74)
print(f'  ROUTING SEARCH COMPARISON  ·  K={cfg.K} M={cfg.M} R_LoS={cfg.R_LoS_km} '
      f'P_S={cfg.P_S_dBm}dBm')
print(f'  {len(rows["tan_sa"])} states · SA iters={a.iters} beta0={a.beta0} '
      f'gamma_f={a.gf} · everything except routing held identical')
print('=' * 74)
print(f"  {'routing search':34s} {'R_tot':>8} {'QoS':>8}")
lbl = {'tan_sa': f'Tan Alg-4 simulated annealing ({a.iters} it)',
       'coord':  'our AO coordinate ascent',
       'exh':    f'EXHAUSTIVE over {n_comb} assignments'}
base = None
for k in ['tan_sa', 'coord', 'exh']:
    if not rows[k]:
        continue
    A = np.array(rows[k])
    r, q = A[:, 0].mean(), A[:, 1].mean() * 100
    if base is None:
        base = r
    print(f'  {lbl[k]:34s} {r:>8.3f} {q:>7.1f}%')
print('-' * 74)
if rows['tan_sa'] and rows['coord']:
    t, c = np.array(rows['tan_sa'])[:, 0].mean(), np.array(rows['coord'])[:, 0].mean()
    print(f'  coordinate ascent − Tan SA = {c - t:+.3f}  ({100*(c-t)/t:+.1f}%)')
if rows['exh']:
    e = np.array(rows['exh'])[:, 0].mean()
    print(f'  exhaustive − coordinate ascent = {e - c:+.3f}  '
          f'· exhaustive − Tan SA = {e - t:+.3f}')
print('=' * 74)
