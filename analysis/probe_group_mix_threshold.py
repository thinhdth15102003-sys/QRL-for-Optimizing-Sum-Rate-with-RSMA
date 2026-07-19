"""
probe_group_mix_threshold.py
----------------------------
GROUP-MIX THRESHOLD heatmap + closed-form validation (docs/IRS-Group-Capacity-
Ablation.md SS6).  Single-IRS setting (K=10, M=1) so (A, B) semantics are clean:

    A = #blocked users routed onto the IRS   (nearest/strongest first)
    B = #free    users routed onto the IRS   (nearest first)
    remaining users -> direct (group 0); un-routed blocked users sit dead in
    group 0 (beta_blk^2-attenuated) exactly as in the real env.

Equal ref power (rho=0.2 split, uniform within), oracle phase (all elements
same index -> |sum phi| = N), C_k = equal share.  For each (A, B) cell:

  measured : R_tot, QoS% of IRS-group members, overall QoS%   (code)
  predicted: group-members-all-OK  <=>  A+B <= n*_mix,
             n*_mix = f_c(G_min) / (D_k - f_p(G_wk))          (SS6 closed form)

Prints per-P_S grids + classification accuracy of the closed form.

Usage:  python analysis/probe_group_mix_threshold.py [--N 24] [--draws 40]
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import make_config
from CSI.env import ISTNEnv

RHO = 0.2


def f_p(gam, b, a):
    return np.log2(1.0 + b * gam / (gam * (1 - b - a) + 1.0))


def f_c(gam, a):
    return np.log2(1.0 + a * gam / (gam * (1 - a) + 1.0))


def run(P_dBm, N, n_draws, a_max=6, b_max=4, seed=13):
    cfg = make_config(K=10, M=1, N=N, P_S_dBm=P_dBm)
    env = ISTNEnv(cfg, seed=seed, n_steps_ep=10)
    K, M = cfg.K, cfg.M
    Dk   = cfg.D_k_bps_hz

    cells = {}            # (A,B) -> list of (R_tot, qos_grp, qos_all)
    cls_ok, cls_n = 0, 0  # closed-form classification accuracy

    d = 0
    while d < n_draws:
        env.reset()
        ch = env.channels
        blocked = ch['su_blocked']
        blk_idx = np.where(blocked)[0]
        fre_idx = np.where(~blocked)[0]
        if len(blk_idx) == 0 or len(fre_idx) < 2:
            continue
        d += 1
        # routing order: strongest cascade first (nearest IRS distance)
        blk_order = blk_idx[np.argsort(ch['d_IRS_U'][0, blk_idx])]
        fre_order = fre_idx[np.argsort(ch['d_IRS_U'][0, fre_idx])]
        pidx = np.zeros((M, cfg.N), dtype=int)
        Phi  = env.phase_model.build_phi(env.phase_model.index_to_phase(pidx))

        for A in range(0, min(len(blk_order), a_max) + 1):
            for B in range(0, min(len(fre_order), b_max) + 1):
                phi = np.zeros(K, dtype=int)
                phi[blk_order[:A]] = 1
                phi[fre_order[:B]] = 1
                act_ids = [1] if (A + B) > 0 else []
                G = len(act_ids)
                b_frac, a_frac = (1 - RHO) / K, RHO / (G + 1)
                w_p = np.full(K, b_frac * cfg.P_S)
                w_c = np.full(G + 1, a_frac * cfg.P_S)
                out = env.rate_computer.compute_sum_rate(
                    phi, Phi, ch, w_p, w_c, active_irs_ids=act_ids)
                ok      = out['R_private'] + out['C_k'] >= Dk
                grp     = phi == 1
                qos_all = float(np.mean(ok))
                qos_grp = float(np.mean(ok[grp])) if grp.any() else np.nan
                cells.setdefault((A, B), []).append(
                    (out['sum_rate'], qos_grp, qos_all))

                # ---- closed-form prediction for the IRS group ----
                if grp.any():
                    gam  = (np.abs(out['h_eff']) ** 2) * cfg.P_S / cfg.sigma2
                    gmin = gam[grp].min()
                    n    = A + B
                    fp_w = f_p(gmin, b_frac, a_frac)   # weakest member binds
                    fc_m = f_c(gmin, a_frac)
                    if Dk > fp_w:
                        pred_all_ok = n <= fc_m / (Dk - fp_w)
                    else:
                        pred_all_ok = True
                    meas_all_ok = bool(ok[grp].all())
                    cls_ok += int(pred_all_ok == meas_all_ok)
                    cls_n  += 1

    def grid(idx, fmt):
        As = sorted({k[0] for k in cells}); Bs = sorted({k[1] for k in cells})
        print("A\\B " + " ".join(f"{b:>6}" for b in Bs))
        for a in As:
            row = []
            for b in Bs:
                v = cells.get((a, b))
                row.append(fmt(np.nanmean([x[idx] for x in v])) if v else "     -")
            print(f"{a:>3} " + " ".join(row))

    print(f"\n===== P_S={P_dBm} dBm · N={N} · K=10 M=1 · {n_draws} draws =====")
    print(f"--- R_tot(A,B) ---");            grid(0, lambda v: f"{v:6.3f}")
    print(f"--- QoS% of IRS-group members ---"); grid(1, lambda v: f"{100*v:6.1f}")
    print(f"--- QoS% overall ---");          grid(2, lambda v: f"{100*v:6.1f}")
    print(f"closed-form n*_mix classification accuracy: "
          f"{100*cls_ok/max(cls_n,1):.1f}%  ({cls_ok}/{cls_n} cells)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=24)
    ap.add_argument('--draws', type=int, default=40)
    args = ap.parse_args()
    for P in (50, 40):
        run(P, args.N, args.draws)
