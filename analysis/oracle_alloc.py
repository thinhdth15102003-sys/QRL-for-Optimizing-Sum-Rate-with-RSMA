"""
analysis/oracle_alloc.py
------------------------
Closed-form / search ORACLE allocations for the POWER and Ck (common-rate-split)
sub-problems — the candidate WARM-UP TARGETS that bake the binding levers into the
agent's start state instead of forcing them live (live --power-fairness blend hammers
CkMLP; see Common-Knowledge [K] + result_44 ck-collapse).

Three targets, all consistent with CSI/rate.py:

  oracle_ck_met   — within each group the common rate R_c_group is bottlenecked by the
                    WEAKEST member (sinr_c.min). Given private rates R_p, each user needs
                    common rate d_k = max(0, D_k − R_p) to clear QoS. Greedy demand-fill
                    (cheapest-d_k first, up to the group budget R_c_group) MAXIMISES the
                    group's #met. Closed-form. This is exactly "fill the weak users",
                    the thing the live CkMLP keeps failing to learn.

  multiuser_phase_idx — a user's effective channel on IRS m depends ONLY on Φ[m], so the
                    groups stay INDEPENDENT given the assignment and each IRS is optimised
                    alone. Under PER-ELEMENT g_RU the per-IRS choice is L^N, not L, so this
                    is no longer exhaustive: it seeds from the per-element closed form
                    (analysis/phase_oracle.oracle_phase_idx, which maximises coherent gain)
                    and coordinate-ascends on the #met objective, always scoring the
                    incumbent so it can only improve on that seed.

  greedy_power_ck — coordinate-ascent over the private/common split + per-user private
                    power (move a quantum from the most-surplus met user to the closest
                    unmet user), scored by oracle_ck #met. Approximate "oracle power".

Used by analysis/probe_oracle_alloc.py to measure the QoS CEILING of each target under
the agent's LIVE routing+phase, before committing a fresh warm-up run.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from analysis.phase_oracle import oracle_phase_idx


# ── helpers ──────────────────────────────────────────────────────────────────

def phi_from_idx(pidx, cfg):
    """(M,N) int phase index → (M,N,N) diagonal complex Φ."""
    M, N = cfg.M, cfg.N
    Phi = np.zeros((M, N, N), dtype=complex)
    di = np.arange(N)
    Phi[:, di, di] = np.exp(1j * cfg.phase_levels[np.asarray(pidx, dtype=int)])
    return Phi


def _est_channels(ch):
    """Map the env channel dict to the non-_hat keys oracle_phase_idx expects."""
    return {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
            'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}


# ── oracle Ck (greedy demand-fill) ───────────────────────────────────────────

def oracle_ck_met(R_private, R_c_group, groups, Dk):
    """Greedy within-group common-rate fill that maximises #met.

    Returns (met_vec (K,) bool, C_k (K,) float).  Per group, fund users with the
    SMALLEST common-rate demand d_k=max(0,Dk−R_p) first, up to the group budget
    R_c_group[gid]; a user is met iff fully funded (or already met by private).
    No rescale — Σ_{k∈g} C_k ≤ R_c_group (leftover is QoS-irrelevant slack).
    """
    R_private = np.asarray(R_private, dtype=float)
    K = R_private.shape[0]
    C_k = np.zeros(K)
    met = R_private >= Dk
    for gid, members in groups.items():
        members = np.asarray(list(members), dtype=int)
        if members.size == 0:
            continue
        budget = float(R_c_group.get(int(gid), 0.0))
        demand = np.maximum(0.0, Dk - R_private[members])
        spent = 0.0
        for j in np.argsort(demand):          # cheapest demand first
            d = float(demand[j])
            if d <= 1e-12:
                continue                       # already met by private alone
            if spent + d <= budget + 1e-12:
                C_k[members[j]] = d
                spent += d
                met[members[j]] = True
            # else: cannot fully fund → leave unmet, save budget for cheaper users
    return met, C_k


# ── per-recipe #met evaluator ────────────────────────────────────────────────

def met_under_recipe(assignment, Phi, ch, cfg, rate, w_p, w_c_vec, active, Dk,
                     sigma2=None, oracle_ck=True):
    """met_vec (K,) for a given power recipe under fixed assignment+phase.

    oracle_ck=True → greedy demand-fill; False → equal within-group split
    (matches RateComputer.compute_sum_rate C_k=None semantics).
    """
    part = rate.compute_rates_partial(np.asarray(assignment), Phi, ch, w_p, w_c_vec,
                                      active_irs_ids=active, sigma2=sigma2)
    R_p = np.asarray(part['R_private'], dtype=float)
    R_cg, groups = part['R_c_group'], part['groups']
    if oracle_ck:
        met, _ = oracle_ck_met(R_p, R_cg, groups, Dk)
    else:
        C_k = np.zeros(cfg.K)
        for gid, members in groups.items():
            members = np.asarray(list(members), dtype=int)
            if members.size:
                C_k[members] = R_cg.get(int(gid), 0.0) / members.size
        met = (R_p + C_k) >= Dk
    return met, R_p


# ── multi-user phase oracle on the #met objective (per-element) ──────────────

def multiuser_phase_idx(assignment, ch, cfg, rate, active, Dk, w_p, w_c_vec,
                        sigma2=None, refine_sweeps: int = 1):
    """
    Per active IRS, maximise THAT group's #met (oracle Ck) with power fixed.
    Groups are independent given the assignment, so each IRS is optimised alone.

    Starts from the PER-ELEMENT closed-form oracle (which maximises coherent
    gain, not #met), then improves it on the #met objective:
      • candidate sweep over the L uniform-phase rows (cheap, L evals);
      • per-element coordinate ascent, `refine_sweeps` passes (N·L evals each).
    The incumbent is always scored, so the result can never be worse than the
    per-element init.

    ⚠ Pre-refactor this function OVERWROTE the whole row with one uniform level
    (`pidx[mi,:] = l`) — correct only while g_RU was per-IRS scalar and every
    element shared one optimal phase. Under per-element g_RU that discards the
    alignment entirely; hence the rewrite.
    """
    assignment = np.asarray(assignment, dtype=int)
    L = len(cfg.phase_levels)
    pidx = oracle_phase_idx(_est_channels(ch), assignment, cfg).copy()   # (M,N) init

    def _n_met(trial, members):
        met, _ = met_under_recipe(assignment, phi_from_idx(trial, cfg), ch, cfg,
                                  rate, w_p, w_c_vec, active, Dk, sigma2, True)
        return int(met[members].sum())

    for m in active:                                    # m is 1-based IRS gid
        mi = m - 1
        members = np.where(assignment == m)[0]
        if members.size == 0:
            continue

        best_row = pidx[mi].copy()
        best_n   = _n_met(pidx, members)                # score the incumbent

        # ① uniform-phase candidates (the old search, kept as cheap restarts)
        for l in range(L):
            trial = pidx.copy(); trial[mi, :] = l
            n = _n_met(trial, members)
            if n > best_n:
                best_n, best_row = n, trial[mi].copy()
        pidx[mi] = best_row

        # ② per-element coordinate ascent on #met
        for _ in range(refine_sweeps):
            changed = False
            for el in range(cfg.N):
                cur = int(pidx[mi, el])
                for l in range(L):
                    if l == cur:
                        continue
                    trial = pidx.copy(); trial[mi, el] = l
                    n = _n_met(trial, members)
                    if n > best_n:
                        best_n, pidx[mi, el] = n, l
                        changed = True
            if not changed:
                break
    return pidx


# ── approximate oracle power (coordinate ascent) ─────────────────────────────

def greedy_power_ck(assignment, Phi, ch, cfg, rate, active, Dk, sigma2=None,
                    pf_grid=(0.7, 0.8, 0.9), iters=None):
    """Search the private/common split + per-user private power to maximise #met
    (oracle Ck). Returns (met_vec, w_p, w_c_vec). Approximate ('greedy') oracle."""
    assignment = np.asarray(assignment, dtype=int)
    K, P_S, G = cfg.K, cfg.P_S, len(active)
    iters = K if iters is None else iters
    best = (-1, None, None)
    for pf in pf_grid:
        priv_tot, comm_tot = P_S * pf, P_S * (1.0 - pf)
        wc = np.full(G + 1, comm_tot / (G + 1))
        wp = np.full(K, priv_tot / K)
        met, _ = met_under_recipe(assignment, Phi, ch, cfg, rate, wp, wc, active,
                                  Dk, sigma2, True)
        delta = priv_tot * 0.08
        for _ in range(iters):
            part = rate.compute_rates_partial(assignment, Phi, ch, wp, wc,
                                              active_irs_ids=active, sigma2=sigma2)
            R_p = np.asarray(part['R_private'], dtype=float)
            m2, _ = oracle_ck_met(R_p, part['R_c_group'], part['groups'], Dk)
            unmet, metset = np.where(~m2)[0], np.where(m2)[0]
            if unmet.size == 0 or metset.size == 0:
                break
            u = unmet[np.argmin(Dk - R_p[unmet])]      # closest-to-threshold unmet
            d = metset[np.argmax(R_p[metset])]         # most-surplus met (donor)
            if wp[d] <= delta:
                break
            wp[d] -= delta; wp[u] += delta
            m3, _ = met_under_recipe(assignment, Phi, ch, cfg, rate, wp, wc, active,
                                     Dk, sigma2, True)
            if int(m3.sum()) >= int(m2.sum()):
                met = m3                                # keep (≥ allows lateral unlock)
            else:
                wp[d] += delta; wp[u] -= delta          # revert + stop
                break
        nmet = int(met.sum())
        if nmet > best[0]:
            best = (nmet, wp.copy(), wc.copy())
    met, _ = met_under_recipe(assignment, Phi, ch, cfg, rate, best[1], best[2],
                              active, Dk, sigma2, True)
    return met, best[1], best[2]
