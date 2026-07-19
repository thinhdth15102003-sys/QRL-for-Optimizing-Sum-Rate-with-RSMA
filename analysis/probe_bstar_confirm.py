"""
probe_bstar_confirm.py — §7 CONFIRMATION TEST (docs/IRS-Group-Capacity-Ablation.md)
-----------------------------------------------------------------------------------
Controlled rig, Case-2 topology (K=10, M=2) with **IRS 2 DISABLED** — never assigned,
never a group, no power slice => zero interference with the experiment (user re-spec
2026-07-10; replaces the parked-user variant).  Weights PINNED across every config:
2 groups (direct + IRS 1), beta_g = 1/2, alpha_k = 1/K, rho = 0.2.

Per draw (positions + fading = ONE realization, shared by all B -> paired CRN):
  - A = 3 blocked users nearest IRS 1 pinned on it (extra blocked stay dead in
    group 0 -> constant background, mirrored by theory).
  - free users admitted BEST-FIRST (nearest IRS 1), B = 0..n_free-1 (>=1 direct
    kept: no cliff row; cliff already validated 0.141-vs-0.137).
  - measured (CODE, CSI/rate.py): R_tot(B), marginal, QoS floor (equal-share
    all-members-ok) + QoS general (sum-shortfall <= R_c^g).
  - theory (Γ-form §1, SAME realized channels; Γ per route extracted via two
    all-direct / all-IRS1 calls): R_tot(B), B*_rate (cumulative argmax),
    B*_QoS floor/general, B* = min.
  -> per-draw classification theory-vs-code + rig B* distribution (the answer).

Usage:  python analysis/probe_bstar_confirm.py [--N 24] [--draws 50]
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import make_config
from CSI.env import ISTNEnv

RHO, A_FIX, B_CAP = 0.2, 3, 5


def theory_R(gam, x, y):
    """Private/common rates from the §1 Γ-form at power fractions x (private), y (common)."""
    gp = x * gam / (gam * (1.0 - x - y) + 1.0)
    gc = y * gam / (gam * (1.0 - y) + 1.0)
    return np.log2(1.0 + gp), np.log2(1.0 + gc)


def run(P_dBm, N, n_draws, seed=13):
    cfg = make_config(K=10, M=2, N=N, P_S_dBm=P_dBm)
    env = ISTNEnv(cfg, seed=seed, n_steps_ep=10)
    K   = cfg.K
    Dk  = cfg.D_k_bps_hz
    x   = (1.0 - RHO) / K          # private power fraction per user
    y   = RHO / 2.0                # common fraction per group (2 groups, PINNED)
    w_p = np.full(K, x * cfg.P_S)
    w_c = np.full(2, y * cfg.P_S)  # [group0, IRS1] — IRS 2 has NO slice

    stats = dict(mrg=[], b_rate=[], b_qfl=[], b_qgen=[], b_star=[],
                 cls_rate=0, cls_qfl=0, cls_qgen=0, cls_star=0, n=0, wire=0.0)

    d = 0
    while d < n_draws:
        env.reset()
        ch      = env.channels
        blocked = ch['su_blocked']
        blk_idx = np.where(blocked)[0]
        fre_idx = np.where(~blocked)[0]
        if len(blk_idx) < A_FIX or len(fre_idx) < 3:
            continue
        d += 1
        pidx = np.zeros((cfg.M, cfg.N), dtype=int)
        Phi  = env.phase_model.build_phi(env.phase_model.index_to_phase(pidx))

        # pinned membership: A = 3 blocked nearest IRS 1; free sorted best-first
        blk_order = blk_idx[np.argsort(ch['d_IRS_U'][0, blk_idx])]
        fre_order = fre_idx[np.argsort(ch['d_IRS_U'][0, fre_idx])]
        A_set  = blk_order[:A_FIX]
        B_pool = fre_order[:min(len(fre_order) - 1, B_CAP)]   # keep >=1 direct
        Bmax   = len(B_pool)

        # per-route Γ for THEORY: two probe calls on the same channels
        phi0 = np.zeros(K, dtype=int)
        out0 = env.rate_computer.compute_sum_rate(phi0, Phi, ch, w_p, w_c[:1],
                                                  active_irs_ids=[])
        g_dir = (np.abs(out0['h_eff']) ** 2) * cfg.P_S / cfg.sigma2
        phi1 = np.ones(K, dtype=int)
        out1 = env.rate_computer.compute_sum_rate(phi1, Phi, ch, w_p, w_c,
                                                  active_irs_ids=[1])
        g_irs = (np.abs(out1['h_eff']) ** 2) * cfg.P_S / cfg.sigma2

        R_code, R_th, qfl, qgen = [], [], [], []
        for B in range(Bmax + 1):
            phi = np.zeros(K, dtype=int)
            phi[A_set] = 1
            phi[B_pool[:B]] = 1
            out = env.rate_computer.compute_sum_rate(phi, Phi, ch, w_p, w_c,
                                                     active_irs_ids=[1])
            R_code.append(out['sum_rate'])
            grp = phi == 1
            ok  = out['R_private'] + out['C_k'] >= Dk          # equal-share fill
            qfl.append(bool(ok[grp].all()))
            Rc_g = float(out['R_common_group'].get(1, 0.0))
            s = np.clip(Dk - out['R_private'][grp], 0.0, None).sum()
            qgen.append(bool(s <= Rc_g))
            # ---- theory R_tot(B) from Γ-form on the same realization ----
            gam = np.where(grp, g_irs, g_dir)
            Rp, Rc = theory_R(gam, x, y)
            R_th.append(Rp.sum() + Rc[grp].min() + Rc[~grp].min())

        R_code, R_th = np.array(R_code), np.array(R_th)
        stats['wire'] = max(stats['wire'], float(np.max(np.abs(R_code - R_th))))
        if Bmax >= 1:
            stats['mrg'].append(np.diff(R_code))

        def maxfeas(q):
            q = np.asarray(q); idx = np.where(q)[0]
            return int(idx.max()) if len(idx) else 0
        b_rate_c, b_rate_t = int(R_code.argmax()), int(R_th.argmax())
        b_qfl_c            = maxfeas(qfl)
        b_qgen_c           = maxfeas(qgen)
        # theory QoS from Γ-form
        qfl_t, qgen_t = [], []
        for B in range(Bmax + 1):
            grp = np.zeros(K, bool); grp[A_set] = True; grp[B_pool[:B]] = True
            gam = np.where(grp, g_irs, g_dir)
            Rp, Rc = theory_R(gam, x, y)
            Rc_g   = Rc[grp].min()
            n_g    = grp.sum()
            qfl_t.append(bool((Rp[grp] + Rc_g / n_g >= Dk).all()))
            qgen_t.append(bool(np.clip(Dk - Rp[grp], 0, None).sum() <= Rc_g))
        b_qfl_t, b_qgen_t = maxfeas(qfl_t), maxfeas(qgen_t)
        b_star_c, b_star_t = min(b_rate_c, b_qgen_c), min(b_rate_t, b_qgen_t)

        stats['b_rate'].append(b_rate_c); stats['b_qfl'].append(b_qfl_c)
        stats['b_qgen'].append(b_qgen_c); stats['b_star'].append(b_star_c)
        stats['cls_rate'] += int(b_rate_c == b_rate_t)
        stats['cls_qfl']  += int(b_qfl_c == b_qfl_t)
        stats['cls_qgen'] += int(b_qgen_c == b_qgen_t)
        stats['cls_star'] += int(b_star_c == b_star_t)
        stats['n'] += 1

    n = stats['n']
    mrg = np.full((len(stats['mrg']), max(len(m) for m in stats['mrg'])), np.nan)
    for i, m in enumerate(stats['mrg']):
        mrg[i, :len(m)] = m
    def dist(v):
        v = np.array(v)
        return f"mode={np.bincount(v).argmax()} mean={v.mean():.2f}"
    print(f"\n===== P_S={P_dBm} dBm · N={N} · K=10 M=2 (IRS2 OFF) · A={A_FIX} · {n} draws =====")
    print(f"  wiring |R_code−R_theory| max      : {stats['wire']:.2e}")
    print(f"  mean marginal ΔR_tot per admit    : " +
          " ".join(f"{np.nanmean(mrg[:, j]):+.4f}" for j in range(mrg.shape[1])))
    print(f"  B*_rate  {dist(stats['b_rate'])}   | theory match {100*stats['cls_rate']/n:.1f}%")
    print(f"  B*_QoS floor {dist(stats['b_qfl'])} | match {100*stats['cls_qfl']/n:.1f}%   "
          f"· general {dist(stats['b_qgen'])} | match {100*stats['cls_qgen']/n:.1f}%")
    print(f"  ⭐ B* = min(rate, QoS-gen)  {dist(stats['b_star'])} | theory match {100*stats['cls_star']/n:.1f}%")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=24)
    ap.add_argument('--draws', type=int, default=50)
    args = ap.parse_args()
    for P in (50, 40):
        run(P, args.N, args.draws)
