"""Rate-QoS frontier by EXHAUSTIVE assignment search, both cases.

Replaces probe_v_star / probe_true_frontier / probe_pareto_frontier, which
disagree with each other because each enforces a different constraint (per-state
QoS >= t, "n_met >= n on the estimate", "serve-all") and each searches the
assignment differently. One definition, one output.

Per state the achievable set of (R_tot, QoS) pairs is built as

    assignment : EXHAUSTIVE over (M+1)^K
    phase      : per-element oracle, exact given the assignment
    power      : w_p ∝ |h_est|^(2*beta) at split f   (grid; a full-vector
                 refinement is a separate, later step)
    C_k        : demand-fill oracle, then leftover group budget spread evenly

decided on the estimate ĝ and scored on the true channel g, averaged over the
same reward_noise_avg sigma^2 draws env.step uses. The frontier itself is
assembled from those point sets by aggregate_pareto.py.

Two reductions make the exhaustive search affordable:
  * the oracle phase of panel m depends only on the SET of users on m, so it is
    cached over the M*(2^K - 1) (panel, subset) pairs instead of recomputed per
    assignment;
  * the rate kernel is closed form, so every assignment is evaluated in one
    vectorised pass rather than one call each.

The vectorised kernel is checked against CSI.rate.RateComputer on random draws
before any measurement runs; --validate does only that.

    python analysis/probe_pareto_exhaustive.py --K 5  --M 1 --validate
    python analysis/probe_pareto_exhaustive.py --K 10 --M 2 --pilot 3
    python analysis/probe_pareto_exhaustive.py --K 10 --M 2 --shard 0/8
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P                                              # noqa: E402
from params import make_config                                  # noqa: E402
from CSI.env import ISTNEnv                                     # noqa: E402
from CSI.rate import RateComputer                               # noqa: E402
from analysis.phase_oracle import oracle_phase_idx, est_channels  # noqa: E402
from analysis.oracle_alloc import phi_from_idx                  # noqa: E402

BETA = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
SPLIT = np.linspace(0.05, 0.95, 13)
KEEP_PER_QOS = 32          # survivors carried into the 16-draw rescoring


# ----------------------------------------------------------------- geometry
def enumerate_assignments(K, M):
    """(A,K) int array of every assignment, A = (M+1)^K."""
    A = (M + 1) ** K
    d = np.empty((A, K), dtype=np.int8)
    idx = np.arange(A)
    for k in range(K):
        d[:, k] = (idx // (M + 1) ** k) % (M + 1)
    return d


def panel_bitmasks(digits, m):
    """(A,) int: bitmask of the users assigned to panel m."""
    bits = (digits == m).astype(np.int64)
    return bits @ (1 << np.arange(digits.shape[1], dtype=np.int64))


def build_h2(digits, ch, cfg, rate):
    """|h|^2 on the estimate and on the truth for every assignment.

    The oracle phase of a panel depends only on the set of users it carries, so
    it is computed once per (panel, subset) and broadcast to the assignments
    sharing that subset.
    """
    K, M = cfg.K, cfg.M
    A = digits.shape[0]
    est = est_channels(ch)
    H2e = np.zeros((A, K))
    H2t = np.zeros((A, K))

    # direct users: the effective channel is g_SU, independent of any panel
    zero = np.zeros(K, dtype=int)
    Phi0 = phi_from_idx(np.zeros((M, cfg.N), dtype=int), cfg)
    he0 = rate.effective_channels_all(zero, Phi0, ch, use_true=False)
    ht0 = rate.effective_channels_all(zero, Phi0, ch, use_true=True)
    direct = (digits == 0)
    H2e[direct] = np.broadcast_to(np.abs(he0) ** 2, (A, K))[direct]
    H2t[direct] = np.broadcast_to(np.abs(ht0) ** 2, (A, K))[direct]

    n_phase = 0
    for m in range(1, M + 1):
        bm = panel_bitmasks(digits, m)
        order = np.argsort(bm, kind="stable")
        bs = bm[order]
        edges = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
        for e0, e1 in zip(edges, np.r_[edges[1:], len(bs)]):
            mask = int(bs[e0])
            if mask == 0:
                continue                       # panel unused by these rows
            members = np.flatnonzero(
                (mask >> np.arange(K, dtype=np.int64)) & 1)
            a = np.zeros(K, dtype=int)
            a[members] = m
            Phi = phi_from_idx(oracle_phase_idx(est, a, cfg), cfg)
            n_phase += 1
            he = rate.effective_channels_all(a, Phi, ch, use_true=False)
            ht = rate.effective_channels_all(a, Phi, ch, use_true=True)
            rows = order[e0:e1]
            H2e[np.ix_(rows, members)] = np.abs(he[members]) ** 2
            H2t[np.ix_(rows, members)] = np.abs(ht[members]) ** 2
    return H2e, H2t, n_phase


# ------------------------------------------------------------ vector kernel
def group_stream_index(digits, M):
    """Mirror RateComputer._make_wc_map: one common stream per active panel in
    ascending gid order, plus the direct group. Returns (A,K) stream index and
    (A,) stream count."""
    A, K = digits.shape
    active = np.stack([(digits == m).any(1) for m in range(1, M + 1)], 1)  # (A,M)
    # stream 0 is the direct group; active panel j takes the next free slot
    slot = np.cumsum(active, axis=1) * active                    # (A,M), 1-based
    idx = np.zeros((A, K), dtype=np.int64)
    for m in range(1, M + 1):
        sel = digits == m
        idx[sel] = np.broadcast_to(slot[:, m - 1:m], (A, K))[sel]
    n_stream = active.sum(1) + 1
    return idx, n_stream


def eval_grid(H2e, H2t, digits, stream_idx, n_stream, cfg, s2_list, keep):
    """Achievable (R_tot, QoS) over the (beta, split) grid.

    Searches on the nominal sigma^2, keeps the strongest `keep` points at each
    attainable QoS level, and rescores only those over `s2_list`.
    """
    K, M, PS, Dk = cfg.K, cfg.M, cfg.P_S, cfg.D_k_bps_hz  # noqa: F841
    A = digits.shape[0]
    n_str_max = M + 1
    pts = []

    for beta in BETA:
        w = H2e ** beta if beta else np.ones_like(H2e)
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
        for f in SPLIT:
            wp = w * (PS * f)
            wc_each = PS * (1.0 - f) / n_stream                  # (A,)
            own_wc = np.broadcast_to(wc_each[:, None], (A, K))
            tot_wc = wc_each * n_stream                          # (A,)
            r, Rk = _score(H2e, H2t, wp, own_wc, tot_wc, digits, cfg, cfg.sigma2)
            pts.append((r, (Rk >= cfg.D_k_bps_hz).mean(1), beta, f))

    R = np.stack([p[0] for p in pts])        # (G, A)
    Q = np.stack([p[1] for p in pts])
    grid = [(p[2], p[3]) for p in pts]

    # survivors: the best `keep` by rate at each attainable QoS level
    flatR, flatQ = R.ravel(), Q.ravel()
    levels = np.unique(flatQ)
    sel = []
    for lv in levels:
        w_lv = np.flatnonzero(flatQ == lv)
        if w_lv.size > keep:
            w_lv = w_lv[np.argpartition(-flatR[w_lv], keep)[:keep]]
        sel.append(w_lv)
    sel = np.unique(np.concatenate(sel))

    out = np.empty((sel.size, 3))
    for i, fl in enumerate(sel):
        gi, ai = divmod(int(fl), A)
        beta, f = grid[gi]
        w = H2e[ai:ai + 1] ** beta if beta else np.ones((1, K))
        w = w / max(w.sum(), 1e-300)
        wp = w * (cfg.P_S * f)
        ns = int(n_stream[ai])
        wc_each = cfg.P_S * (1.0 - f) / ns
        own = np.full((1, K), wc_each)
        rs = js = 0.0
        rk_acc = np.zeros(K)
        tw = np.array([wc_each * ns])
        dec = _decide_ck(H2e[ai:ai + 1], wp, own, tw, digits[ai:ai + 1],
                         cfg, cfg.sigma2)
        for s2 in s2_list:
            r, Rk = _apply(H2t[ai:ai + 1], wp, own, tw, digits[ai:ai + 1],
                           cfg, s2, dec)
            short = np.maximum(0.0, Dk - Rk[0])
            js += r[0] - cfg.lambda_D * float(
                np.sum((short / (Dk + P.epsilon_qp)) ** 2))
            rs += r[0]; rk_acc += Rk[0]
        n = len(s2_list)
        out[i] = (rs / n, float(np.mean(rk_acc / n >= Dk)), js / n)
    return out


def _rates(h2, wp, own_wc, tot_wc, s2):
    tot_wp = wp.sum(1, keepdims=True)
    tw = np.asarray(tot_wc).reshape(-1, 1)
    s2 = np.asarray(s2)
    sinr_p = h2 * wp / (h2 * (tot_wp - wp + tw - own_wc) + s2)
    sinr_c = h2 * own_wc / (h2 * (tot_wp + tw - own_wc) + s2)
    return np.log2(1.0 + sinr_p), sinr_c


def _decide_ck(h2e, wp, own_wc, tot_wc, digits, cfg, s2):
    """The common-rate split the policy commits to for this state.

    Decided once, on the estimate at the nominal noise level, exactly as the
    caller of RateComputer does: cheapest demand first within each group, then
    the leftover group budget spread evenly, then the 1e-10 floor that
    _finish_sum_rate applies before it rescales.
    """
    M, Dk = cfg.M, cfg.D_k_bps_hz
    Rp_e, sinr_c_e = _rates(h2e, wp, own_wc, tot_wc, s2)
    dec = np.zeros_like(Rp_e)
    for gid in range(M + 1):
        mem = digits == gid
        if not mem.any():
            continue
        has = mem.any(1)
        Rc_e = np.where(has, np.log2(1.0 + np.where(mem, sinr_c_e, np.inf).min(1)), 0.0)
        dem = np.where(mem, np.maximum(0.0, Dk - Rp_e), np.inf)
        order = np.argsort(dem, axis=1)
        dsort = np.take_along_axis(dem, order, 1)
        dsort = np.where(np.isinf(dsort), 0.0, dsort)
        got = np.where(np.cumsum(dsort, axis=1) <= Rc_e[:, None] + 1e-12, dsort, 0.0)
        d = np.take_along_axis(got, np.argsort(order, axis=1), 1) * mem
        n_mem = mem.sum(1)
        left = np.maximum(Rc_e - got.sum(1), 0.0)
        d = d + np.where(n_mem > 0, left / np.maximum(n_mem, 1), 0.0)[:, None] * mem
        dec += np.where(mem, np.maximum(d, 1e-10), 0.0)
    return dec


def _apply(h2t, wp, own_wc, tot_wc, digits, cfg, s2, dec):
    """Score a committed allocation on the true channel at noise level s2.

    _finish_sum_rate renormalises the C_k it is handed so that each group's share
    sums to the common rate of the SCORING channel, so the decision fixes only
    the split and the total follows the true channel, per draw.
    """
    M = cfg.M
    Rp_t, sinr_c_t = _rates(h2t, wp, own_wc, tot_wc, s2)
    Ck = np.zeros_like(Rp_t)
    for gid in range(M + 1):
        mem = digits == gid
        if not mem.any():
            continue
        has = mem.any(1)
        Rc_t = np.where(has, np.log2(1.0 + np.where(mem, sinr_c_t, np.inf).min(1)), 0.0)
        d = np.where(mem, dec, 0.0)
        tot = d.sum(1)
        scale = np.where((Rc_t >= 1e-12) & (tot > 0), Rc_t / np.maximum(tot, 1e-300), 0.0)
        Ck += d * scale[:, None]
    Rk = Rp_t + Ck
    return Rk.sum(1), Rk


def _score(h2e, h2t, wp, own_wc, tot_wc, digits, cfg, s2):
    """Decide and score at the same noise level (the search pass)."""
    dec = _decide_ck(h2e, wp, own_wc, tot_wc, digits, cfg, s2)
    return _apply(h2t, wp, own_wc, tot_wc, digits, cfg, s2, dec)


# -------------------------------------------------------------- validation
def validate(cfg, rate, env, n=200, seed=0, n_draw=4):
    """Vectorised kernel vs RateComputer on random assignment/power draws."""
    rng = np.random.default_rng(seed)
    K, M = cfg.K, cfg.M
    env.reset(seed=42)
    ch = env.channels
    worst_r = worst_q = worst_j = 0.0
    for _ in range(n):
        a = rng.integers(0, M + 1, size=K)
        act = sorted(set(int(x) for x in a if x > 0))
        Phi = phi_from_idx(oracle_phase_idx(est_channels(ch), a, cfg), cfg)
        beta = float(rng.choice(BETA)); f = float(rng.choice(SPLIT))
        he = rate.effective_channels_all(a, Phi, ch, use_true=False)
        w = np.abs(he) ** (2 * beta) if beta else np.ones(K)
        w = w / w.sum()
        wp = w * (cfg.P_S * f)
        G = len(act)
        wc = np.full(G + 1, cfg.P_S * (1.0 - f) / (G + 1))

        draws = [env.channel_model.sample_noise_sigma2() for _ in range(n_draw)]
        part = rate.compute_rates_partial(a, Phi, ch, wp, wc,
                                          active_irs_ids=act, sigma2=cfg.sigma2)
        # reference C_k: demand-fill + even leftover, as the earlier probes do
        from analysis.oracle_alloc import oracle_ck_met
        _, Ckr = oracle_ck_met(part["R_private"], part["R_c_group"],
                               part["groups"], cfg.D_k_bps_hz)
        for gid, mem in part["groups"].items():
            mem = np.asarray(list(mem), dtype=int)
            if mem.size:
                left = float(part["R_c_group"].get(int(gid), 0.0)) - float(Ckr[mem].sum())
                if left > 1e-12:
                    Ckr[mem] += left / mem.size
        # mirror env.step: average the per-draw reward, take QoS from the
        # averaged rates
        acc_r = acc_j = 0.0
        acc_rk = np.zeros(K)
        for s2 in draws:
            oo = rate.compute_sum_rate(a, Phi, ch, wp, wc, C_k=Ckr,
                                       active_irs_ids=act, sigma2=s2,
                                       use_true=True)
            rk = np.asarray(oo["R_private"]) + np.asarray(oo["C_k"])
            sh = np.maximum(0.0, cfg.D_k_bps_hz - rk)
            acc_j += float(oo["sum_rate"]) - cfg.lambda_D * float(
                np.sum((sh / (cfg.D_k_bps_hz + P.epsilon_qp)) ** 2))
            acc_r += float(oo["sum_rate"]); acc_rk += rk
        ref_r = acc_r / n_draw
        ref_q = float(np.mean(acc_rk / n_draw >= cfg.D_k_bps_hz))
        ref_j = acc_j / n_draw

        ht = rate.effective_channels_all(a, Phi, ch, use_true=True)
        h2t = (np.abs(ht) ** 2)[None, :]
        h2e = (np.abs(he) ** 2)[None, :]
        d = a[None, :].astype(np.int8)
        own = np.full((1, K), cfg.P_S * (1.0 - f) / (G + 1))
        mr = mj = 0.0
        mrk = np.zeros(K)
        twv = np.array([cfg.P_S * (1.0 - f)])
        dec = _decide_ck(h2e, wp[None, :], own, twv, d, cfg, cfg.sigma2)
        for s2 in draws:
            r, Rk = _apply(h2t, wp[None, :], own, twv, d, cfg, s2, dec)
            sh = np.maximum(0.0, cfg.D_k_bps_hz - Rk[0])
            mj += r[0] - cfg.lambda_D * float(
                np.sum((sh / (cfg.D_k_bps_hz + P.epsilon_qp)) ** 2))
            mr += r[0]; mrk += Rk[0]
        worst_r = max(worst_r, abs(mr / n_draw - ref_r))
        worst_q = max(worst_q, abs(float(np.mean(mrk / n_draw >= cfg.D_k_bps_hz)) - ref_q))
        worst_j = max(worst_j, abs(mj / n_draw - ref_j))
    return worst_r, worst_q, worst_j


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--P", type=float, default=50.0)
    ap.add_argument("--R-LoS", dest="rlos", type=float, default=0.5)
    ap.add_argument("--env-seeds", default="42,43,44")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--keep", type=int, default=KEEP_PER_QOS)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = make_config(K=a.K, M=a.M, P_S_dBm=a.P, R_LoS_km=a.rlos)
    rate = RateComputer(cfg)
    env = ISTNEnv(cfg=cfg, seed=42, n_steps_ep=a.steps,
                  reward_noise_avg=P.reward_noise_avg)

    print("K=%d M=%d N=%d P_S=%.0f R_LoS=%.1f D_k=%.3f  assignments=%d"
          % (cfg.K, cfg.M, cfg.N, cfg.P_S_dBm, cfg.R_LoS_km,
             cfg.D_k_bps_hz, (a.M + 1) ** a.K))

    t0 = time.time()
    dr, dq, dj = validate(cfg, rate, env, n=200)
    print("kernel check vs RateComputer: max|dR_tot|=%.3e  max|dQoS|=%.3e  "
          "max|dJ|=%.3e  (%.1fs)" % (dr, dq, dj, time.time() - t0))
    if dr > 1e-9 or dq > 1e-12 or dj > 1e-9:
        sys.exit("VECTORISED KERNEL DISAGREES — refusing to measure")
    if a.validate:
        return

    digits = enumerate_assignments(a.K, a.M)
    stream_idx, n_stream = group_stream_index(digits, a.M)
    seeds = [int(s) for s in a.env_seeds.split(",")]
    si, sn = (int(x) for x in a.shard.split("/"))

    todo = [(s, t) for s in seeds for t in range(a.steps)]
    todo = todo[si::sn]
    if a.pilot:
        todo = todo[:a.pilot]

    sets, meta, t0 = [], [], time.time()
    cur = None
    for (s, t) in todo:
        if cur != s:
            env.reset(seed=s); cur = s; step_at = 0
        while step_at < t:
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)
            step_at += 1
        ch = env.channels
        # The noise draws are keyed by (env seed, step) rather than taken from
        # the env's own generator, so a sharded run reproduces a single-process
        # one exactly. Same distribution as ChannelModel.sample_noise_sigma2.
        nrng = np.random.default_rng(1000003 * s + t)
        s2s = [10.0 ** (nrng.normal(cfg.noise_mean_dBW,
                                    np.sqrt(cfg.noise_var_dBW), size=cfg.K) / 10.0)
               for _ in range(P.reward_noise_avg)]
        ts = time.time()
        H2e, H2t, n_phase = build_h2(digits, ch, cfg, rate)
        tp = time.time()
        pts = eval_grid(H2e, H2t, digits, stream_idx, n_stream, cfg, s2s,
                        a.keep)
        sets.append(pts)
        meta.append((s, t, pts.shape[0]))
        print("  seed %d step %3d | phase-cache %4d (%.1fs) | grid %.1fs | "
              "points %d | best R_tot %.4f"
              % (s, t, n_phase, tp - ts, time.time() - tp, pts.shape[0],
                 pts[:, 0].max()))

    dt = time.time() - t0
    print("\n%d states in %.1fs  -> %.2f s/state" % (len(todo), dt, dt / max(len(todo), 1)))
    if a.pilot:
        full = len(seeds) * a.steps
        print("projected full run (%d states, 1 core): %.1f h" % (full, dt / len(todo) * full / 3600))
        return

    out = a.out or ("analysis/data/pareto_exh_K%dM%d_shard%dof%d.npz"
                    % (a.K, a.M, si, sn))
    np.savez_compressed(out, meta=np.array(meta),
                        **{"s%d" % i: p for i, p in enumerate(sets)})
    print("wrote", out)


if __name__ == "__main__":
    main()
