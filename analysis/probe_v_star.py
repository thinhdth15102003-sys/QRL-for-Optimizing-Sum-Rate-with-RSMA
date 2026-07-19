"""
probe_v_star.py
---------------
PER-STATE 4-VARIABLE ORACLE V* + (QoS, R_tot) PARETO FRONTIER for Case-1-sized
problems — the "solve-per-state" ceiling (Tan-2026-style) that both actors
(VQC r45, DNN r106) are amortizing. Gap = V*(λ) − V_agent(λ from infer).

Per sampled env state, optimizes ALL FOUR decision variables:
  1. ASSIGNMENT — exhaustive (M+1)^K enumeration (Case 1: 2^5 = 32).
  2. PHASE      — exhaustive L levels per IRS. EXACT under the scalar-φ channel
                  model (eff_phi = Σ_n e^{jθ} → all elements share one index, and
                  groups are independent given the assignment).
  3. POWER      — two-stage grid: private w ∝ |h_eff|^(2β) (log-domain, β up to 16)
                  × common-pool fraction s × common split {equal, ∝members}.
  4. Ck         — CLOSED-FORM optimal, two flavours (Σ_g C_k = R_c,g is spent in
                  full by env rescale → sum-rate is Ck-INVARIANT; Ck only moves
                  the QoS penalty):
                    · waterfill  — exact minimizer of the env's quadratic penalty
                                   Σ((D_k−R_tot)+/(D_k+ε))²  → reward-optimal
                    · greedy-fill — oracle_ck_met, maximizes #met → QoS endpoint
Selection is done on NOMINAL σ² (the same information the agent has: the noise
draw happens AFTER the action in env.step); the selected action is then re-scored
under --noise-draws σ² realisations via compute_sum_rate (env-exact, including
the C_k per-group rescale) → honest E[reward], E[QoS].

Outputs
  · V*(λ) sweep — the scalarized frontier: per λ the (reward, R_tot, QoS, feas)
    of the per-state argmax. λ=1.5 row = the infer reward metric (infer does NOT
    restore the run's λ_D; params.py λ=1.5 scored BOTH r45 and r106).
  · #met-constrained frontier — max R_tot s.t. nominal #met ≥ n (greedy-fill Ck).
  · sanity row — heuristic oracle-assignment + 80/20 equal power (≈ probe canon).

NOTE: power is a (rich) grid, not a solver → V* is a certified ACHIEVABLE lower
bound of the true ceiling; the true gap can only be ≥ the reported one on the
power axis (assignment+phase are exact, Ck exact given the rest).

Usage:
  python analysis/probe_v_star.py --ckpt results/result_106/checkpoints/ep_02000 \
      --episodes 50 --stride 20 --seed 42
"""
import sys, os, json, argparse, itertools, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.phase_oracle import oracle_phase_idx
from analysis.probe_assignment_quality import _oracle_assignment
from analysis.oracle_alloc import oracle_ck_met, phi_from_idx

LAM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 1e6]
DELTA_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]  # robust margin δ (bps/Hz over D_k)


# ── closed-form Ck ────────────────────────────────────────────────────────────

def waterfill_ck(R_p, R_cg, groups, Dk, eps):
    """Exact minimizer of Σ((D_k−R_p−c_k)+/(D_k+ε))² s.t. Σ_{k∈g} c_k ≤ R_c,g.
    Returns (C_k shape (K,), min quadratic penalty, #met under this allocation)."""
    K = R_p.shape[0]
    C = np.zeros(K); pen = 0.0; nmet = 0
    for gid, members in groups.items():
        m = np.asarray(list(members), dtype=int)
        if m.size == 0:
            continue
        B = float(R_cg.get(int(gid), 0.0))
        s = np.maximum(0.0, Dk - R_p[m])
        tot = float(s.sum())
        if tot <= B + 1e-12:                      # budget covers all shortfalls
            C[m] = s
            nmet += m.size
            continue
        ss = np.sort(s)[::-1]                     # water level t: Σ(s_i−t)+ = B
        csum = np.cumsum(ss)
        t = ss[0]
        for j in range(ss.size):
            tj = (csum[j] - B) / (j + 1)
            nxt = ss[j + 1] if j + 1 < ss.size else 0.0
            if nxt - 1e-15 <= tj <= ss[j] + 1e-15:
                t = max(tj, 0.0); break
        c = np.maximum(0.0, s - t)
        C[m] = c
        resid = np.minimum(s, t)
        pen += float(np.sum((resid / (Dk + eps)) ** 2))
        nmet += int(np.sum(s <= 1e-12))
    return C, pen, nmet


def greedy_ck(R_p, R_cg, groups, Dk, eps):
    """oracle_ck_met wrapper → (C_k, quadratic penalty of that allocation, #met)."""
    met, C = oracle_ck_met(R_p, R_cg, groups, Dk)
    short = np.maximum(0.0, Dk - (R_p + C))
    pen = float(np.sum((short / (Dk + eps)) ** 2))
    return C, pen, int(met.sum())


def margin_ck(R_p, R_cg, groups, Dk):
    """Maximin-margin Ck: fund every shortfall, spread the leftover equally within
    the group (raises the weakest). Returns (C_k, min-user margin R_tot−D_k;
    −maxresid if some group can't cover its shortfalls)."""
    K = R_p.shape[0]
    C = np.zeros(K); margin = np.inf
    for gid, members in groups.items():
        m = np.asarray(list(members), dtype=int)
        if m.size == 0:
            continue
        B = float(R_cg.get(int(gid), 0.0))
        s = np.maximum(0.0, Dk - R_p[m])
        tot = float(s.sum())
        if tot > B + 1e-12:
            # infeasible group: waterfill, margin = −(worst residual shortfall)
            ss = np.sort(s)[::-1]; csum = np.cumsum(ss); t = ss[0]
            for j in range(ss.size):
                tj = (csum[j] - B) / (j + 1)
                nxt = ss[j + 1] if j + 1 < ss.size else 0.0
                if nxt - 1e-15 <= tj <= ss[j] + 1e-15:
                    t = max(tj, 0.0); break
            C[m] = np.maximum(0.0, s - t)
            margin = min(margin, -t)
        else:
            share = (B - tot) / m.size
            C[m] = s + share
            grp_margin = float(np.min(R_p[m] + C[m] - Dk))
            margin = min(margin, grp_margin)
    return C, (0.0 if margin is np.inf else float(margin))


# ── candidate machinery ───────────────────────────────────────────────────────

def private_weights(h_eff, beta):
    """w ∝ |h|^(2β), log-domain (β=16-safe); beta=None → exact one-hot on the
    strongest user (winner-take-all endpoint the finite-β grid cannot express:
    x_rest=0 → zero interference → SINR = Γ_max exactly). Returns weights (K,)."""
    g = np.abs(h_eff) ** 2
    if beta is None:
        w = np.zeros(g.shape[0]); w[int(np.argmax(g))] = 1.0
        return w
    lg = np.log(g + 1e-30)
    w = np.exp(beta * (lg - lg.max()))
    return w / w.sum()


class Cand:
    __slots__ = ('a', 'pidx', 'w_p', 'w_c', 'active', 'sumR', 'pen_wf', 'C_wf',
                 'nmet_g', 'C_g', 'nmet_wf', 'C_m', 'margin')

def eval_candidate(a, Phi, pidx, ch, cfg, rate, active, s_frac, beta, dist, Dk, eps):
    K = cfg.K
    h_eff = rate.effective_channels_all(a, Phi, ch)
    w_p = private_weights(h_eff, beta) * (cfg.P_S * (1.0 - s_frac))
    G = len(active)
    groups_sizes = None
    if dist == 'prop':
        # common power ∝ #members per group (slot 0 = direct, i+1 = active[i])
        sizes = np.zeros(G + 1)
        sizes[0] = float(np.sum(a == 0))
        for i, gid in enumerate(active):
            sizes[i + 1] = float(np.sum(a == gid))
        sizes = np.maximum(sizes, 1e-6)
        w_c = sizes / sizes.sum() * (cfg.P_S * s_frac)
    else:
        w_c = np.full(G + 1, cfg.P_S * s_frac / (G + 1))
    part = rate.compute_rates_partial(a, Phi, ch, w_p, w_c,
                                      active_irs_ids=active, sigma2=None)
    R_p = np.asarray(part['R_private'], dtype=float)
    R_cg, groups = part['R_c_group'], part['groups']
    sumR = float(R_p.sum()) + float(sum(R_cg.get(int(g), 0.0) for g in groups))
    C_wf, pen_wf, nmet_wf = waterfill_ck(R_p, R_cg, groups, Dk, eps)
    C_g, _, nmet_g = greedy_ck(R_p, R_cg, groups, Dk, eps)
    C_m, margin = margin_ck(R_p, R_cg, groups, Dk)
    c = Cand()
    c.a, c.pidx, c.w_p, c.w_c, c.active = a, pidx, w_p, w_c, active
    c.sumR, c.pen_wf, c.C_wf = sumR, pen_wf, C_wf
    c.nmet_g, c.C_g, c.nmet_wf = nmet_g, C_g, nmet_wf
    c.C_m, c.margin = C_m, margin
    return c


def _est(ch):
    return {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
            'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}


def _cand_at(a, pidx, ch, cfg, rate, sf, b, dist, Dk, eps):
    active = sorted(set(int(x) for x in a if x > 0))
    return eval_candidate(a, phi_from_idx(pidx, cfg), pidx, ch, cfg, rate,
                          active, sf, b, dist, Dk, eps)


def hill_climb(a0, ch, cfg, rate, score_fn, power, Dk, eps, max_passes=5):
    """Best-improvement 1-flip local search over the assignment. Phase per
    candidate = heuristic dominant-user oracle (cheap); exact per-IRS scan is
    applied to the FINAL assignments only. Returns the locally-optimal a."""
    K, M = cfg.K, cfg.M
    sf, b, dist = power
    a = np.asarray(a0, dtype=int).copy()

    def sc(x):
        pidx = oracle_phase_idx(_est(ch), x, cfg)
        return score_fn(_cand_at(x, pidx, ch, cfg, rate, sf, b, dist, Dk, eps))

    best = sc(a)
    for _ in range(max_passes):
        improved = False
        best_move, best_val = None, best
        for k in range(K):
            for g in range(M + 1):
                if g == a[k]:
                    continue
                trial = a.copy(); trial[k] = g
                v = sc(trial)
                if v > best_val + 1e-9:
                    best_val, best_move = v, (k, g)
        if best_move is not None:
            a[best_move[0]] = best_move[1]
            best = best_val; improved = True
        if not improved:
            break
    return a


def exact_phase_scan(a, ch, cfg, rate, score_fn, power, Dk, eps):
    """Per-IRS exhaustive level scan with a GLOBAL objective. Exact given the
    assignment+power: a user's rates depend only on its own group's Φ (scalar
    model) → per-IRS choices are separable. One pass suffices."""
    L = len(cfg.phase_levels)
    sf, b, dist = power
    active = sorted(set(int(x) for x in a if x > 0))
    pidx = oracle_phase_idx(_est(ch), np.asarray(a, dtype=int), cfg).copy()
    for m in active:
        best_v, best_l = -np.inf, int(pidx[m - 1, 0])
        for l in range(L):
            trial = pidx.copy(); trial[m - 1, :] = l
            v = score_fn(_cand_at(a, trial, ch, cfg, rate, sf, b, dist, Dk, eps))
            if v > best_v:
                best_v, best_l = v, l
        pidx[m - 1, :] = best_l
    return pidx


def rescore_noisy(cand, C_k, ch, cfg, rate, sig_list, Dk, eps):
    """Env-exact re-eval of a fixed action under the pre-drawn σ² list.
    Returns dict(sumR, pen, qos, feas) of per-draw means."""
    Phi = phi_from_idx(cand.pidx, cfg)
    sR = pen = qos = feas = 0.0
    for s2 in sig_list:
        out = rate.compute_sum_rate(assignment=cand.a, Phi=Phi, channels=ch,
                                    w_p=cand.w_p, w_c_vec=cand.w_c, C_k=C_k,
                                    active_irs_ids=cand.active, sigma2=s2)
        R_tot = np.asarray(out['R_private']) + np.asarray(out['C_k'])
        short = np.maximum(0.0, Dk - R_tot)
        sR += out['sum_rate']
        pen += float(np.sum((short / (Dk + eps)) ** 2))
        qos += float(np.mean(R_tot >= Dk))
        feas += float(out['feasible'])
    n = len(sig_list)
    return {'sumR': sR / n, 'pen': pen / n, 'qos': qos / n, 'feas': feas / n}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None, help='checkpoint dir; cfg from its hyperparameters.json')
    ap.add_argument('--K', type=int, default=None, help='env override (no-ckpt mode / on top of ckpt cfg)')
    ap.add_argument('--M', type=int, default=None)
    ap.add_argument('--R-LoS', dest='r_los', type=float, default=None)
    ap.add_argument('--P-S', dest='p_s', type=float, default=None)
    ap.add_argument('--irs-frac', dest='irs_frac', type=float, default=None)
    ap.add_argument('--user-frac', dest='user_frac', type=float, default=None)
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--stride', type=int, default=20, help='evaluate every Nth step')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--noise-draws', type=int, default=16)
    args = ap.parse_args()

    # cfg from the ckpt's own case + curriculum geometry (safe vs params.py);
    # --ckpt optional: K-sweep mode builds cfg from params.py + CLI overrides.
    ov = {}
    if args.ckpt is not None:
        d = os.path.abspath(args.ckpt); hp = None
        for _ in range(4):
            d = os.path.dirname(d)
            cand_p = os.path.join(d, 'hyperparameters.json')
            if os.path.isfile(cand_p):
                hp = json.load(open(cand_p)); break
        if hp is None:
            raise SystemExit(f"hyperparameters.json not found near {args.ckpt}")
        s = hp['system']
        keys = ['K', 'M', 'N', 'P_S_dBm', 'D_k_bps_hz', 'R_LoS_km',
                'irs_spawn_radius_frac', 'user_free_radius_frac', 'kappa',
                'noise_var_dBW', 'epsilon_qp', 'beta_blocking', 'beta_IRS',
                'd_block_km', 'path_loss_exp', 'user_speed_mps',
                'balanced_blocked_spawn', 'h_IRS_km']
        ov = {k: s[k] for k in keys if k in s}
    if args.K is not None:     ov['K'] = args.K
    if args.M is not None:     ov['M'] = args.M
    if args.r_los is not None: ov['R_LoS_km'] = args.r_los
    if args.p_s is not None:   ov['P_S_dBm'] = args.p_s
    if args.irs_frac is not None:  ov['irs_spawn_radius_frac'] = args.irs_frac
    if args.user_frac is not None: ov['user_free_radius_frac'] = args.user_frac
    cfg = make_config(**ov)
    rate = RateComputer(cfg)
    K, M, Dk, eps = cfg.K, cfg.M, cfg.D_k_bps_hz, cfg.epsilon_qp
    L = len(cfg.phase_levels)
    n_asg = (M + 1) ** K
    large = n_asg * (L ** M) > 2100    # 2^9·4=2048 (K9 M1) still exhaustive → K-sweep uniform mode
    N_LEVELS = [K, K - 1, K - 2]
    print(f"⚙ env: K={K} M={M} P_S={cfg.P_S_dBm}dBm R_LoS={cfg.R_LoS_km} "
          f"radius={cfg.irs_spawn_radius_frac}/{cfg.user_free_radius_frac} "
          f"D_k={Dk} L={L} | assignments={n_asg} "
          f"[{'LOCAL-SEARCH' if large else 'EXHAUSTIVE'}]", flush=True)

    env = ISTNEnv(cfg=cfg, seed=args.seed + 777, n_steps_ep=args.steps, reward_noise_avg=1)

    assignments = [np.array(a, dtype=int) for a in
                   itertools.product(range(M + 1), repeat=K)]
    coarse = [(0.0, 0.1, 'eq'), (0.0, 0.3, 'eq'), (2.0, 0.1, 'eq'),
              (2.0, 0.3, 'eq'), (8.0, 0.1, 'eq'), (8.0, 0.3, 'eq')]
    fine_s = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
    fine_b = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    fine_d = ['eq', 'prop']

    # accumulators
    lam_acc = {lam: {'rew': [], 'sumR': [], 'qos': [], 'feas': [], 'nom_rew': []}
               for lam in LAM_GRID}
    con_acc = {n: {'sumR': [], 'qos': [], 'ok': []} for n in N_LEVELS}
    del_acc = {dl: {'sumR': [], 'qos': [], 'feas': [], 'rew15': [], 'ok': []}
               for dl in DELTA_GRID}
    san_acc = {'sumR': [], 'qos': [], 'rew': []}

    t0 = time.time(); n_states = 0
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for st in range(args.steps):
            if st % args.stride == 0:
                ch = {k: (v.copy() if hasattr(v, 'copy') else v)
                      for k, v in env.channels.items()}
                sig_list = [env.channel_model.sample_noise_sigma2()
                            for _ in range(args.noise_draws)]

                if not large:
                    # ── stage 1: coarse power over ALL assignment×phase ──────
                    combos = []      # (a, pidx, active, best coarse cands)
                    stage1 = []
                    for a in assignments:
                        active = sorted(set(int(x) for x in a if x > 0))
                        levels = range(L) if active else [0]
                        for l in levels:
                            pidx = np.full((M, cfg.N), l, dtype=int)
                            Phi = phi_from_idx(pidx, cfg)
                            best = None
                            for (b, sf, dist) in coarse:
                                c = eval_candidate(a, Phi, pidx, ch, cfg, rate,
                                                   active, sf, b, dist, Dk, eps)
                                stage1.append(c)
                                if best is None or (c.sumR - 1.5 * c.pen_wf) > \
                                                   (best.sumR - 1.5 * best.pen_wf):
                                    best = c
                            combos.append((a, pidx, active, best))
                    # prune: union of top by reward@1.5 / sumR / nmet
                    by_rew = sorted(combos, key=lambda t: t[3].sumR - 1.5 * t[3].pen_wf,
                                    reverse=True)[:6]
                    by_rate = sorted(combos, key=lambda t: t[3].sumR, reverse=True)[:3]
                    by_met = sorted(combos, key=lambda t: (t[3].nmet_g, t[3].sumR),
                                    reverse=True)[:3]
                    pruned, seen = [], set()
                    for (a, pidx, active, _) in by_rew + by_rate + by_met:
                        key = (tuple(a), int(pidx[0, 0]) if M else 0)
                        if key not in seen:
                            seen.add(key); pruned.append((a, pidx, active))
                    cands = list(stage1)
                else:
                    # ── LOCAL-SEARCH mode (large (M+1)^K, e.g. Case 2) ────────
                    # 1-flip hill-climbs from the heuristic oracle assignment
                    # under 3 objectives (rate / reward@1.5 / reward@8-QoS),
                    # then exact per-IRS phase scans on the final assignments.
                    a_heur = np.asarray(_oracle_assignment(
                        ch, env.user_pos, env.irs_pos), dtype=int)
                    climbs = [
                        (lambda c: c.sumR,                  (0.05, 8.0, 'eq')),
                        (lambda c: c.sumR - 1.5 * c.pen_wf, (0.1, 2.0, 'eq')),
                        (lambda c: c.sumR - 8.0 * c.pen_wf, (0.2, 0.0, 'eq')),
                    ]
                    finals = {tuple(a_heur)}
                    for fn, pw in climbs:
                        finals.add(tuple(hill_climb(a_heur, ch, cfg, rate, fn,
                                                    pw, Dk, eps, max_passes=12)))
                    pruned, seenp = [], set()
                    for at in finals:
                        a = np.array(at, dtype=int)
                        active = sorted(set(int(x) for x in a if x > 0))
                        for fn, pw in [(lambda c: c.sumR, (0.05, 8.0, 'eq')),
                                       (lambda c: c.sumR - 8.0 * c.pen_wf,
                                        (0.2, 0.0, 'eq'))]:
                            pidx = exact_phase_scan(a, ch, cfg, rate, fn, pw, Dk, eps)
                            key = (at, tuple(pidx[:, 0]))
                            if key not in seenp:
                                seenp.add(key); pruned.append((a, pidx, active))
                    cands = []

                # ── stage 2: fine power grid on pruned set ───────────────────
                for (a, pidx, active) in pruned:
                    Phi = phi_from_idx(pidx, cfg)
                    for sf in fine_s:
                        for b in fine_b:
                            for dist in fine_d:
                                cands.append(eval_candidate(a, Phi, pidx, ch, cfg,
                                                            rate, active, sf, b,
                                                            dist, Dk, eps))
                    # winner-take-all endpoint: all power → 1 private stream
                    cands.append(eval_candidate(a, Phi, pidx, ch, cfg, rate,
                                                active, 0.0, None, 'eq', Dk, eps))

                # ── selections + noisy re-eval (cached) ──────────────────────
                cache = {}
                def noisy(c, which):
                    key = (id(c), which)
                    if key not in cache:
                        C_k = {'wf': c.C_wf, 'g': c.C_g, 'm': c.C_m}[which]
                        cache[key] = rescore_noisy(c, C_k, ch, cfg, rate,
                                                   sig_list, Dk, eps)
                    return cache[key]

                for lam in LAM_GRID:
                    best = max(cands, key=lambda c: c.sumR - lam * c.pen_wf)
                    r = noisy(best, 'wf')
                    lam_acc[lam]['rew'].append(r['sumR'] - lam * r['pen'])
                    lam_acc[lam]['sumR'].append(r['sumR'])
                    lam_acc[lam]['qos'].append(r['qos'])
                    lam_acc[lam]['feas'].append(r['feas'])
                    lam_acc[lam]['nom_rew'].append(best.sumR - lam * best.pen_wf)

                for n in N_LEVELS:
                    sub = [c for c in cands if c.nmet_g >= n]
                    con_acc[n]['ok'].append(1.0 if sub else 0.0)
                    if sub:
                        best = max(sub, key=lambda c: c.sumR)
                        r = noisy(best, 'g')
                        con_acc[n]['sumR'].append(r['sumR'])
                        con_acc[n]['qos'].append(r['qos'])

                # robust QoS end: max ΣR s.t. nominal maximin margin ≥ δ
                for dl in DELTA_GRID:
                    sub = [c for c in cands if c.margin >= dl]
                    del_acc[dl]['ok'].append(1.0 if sub else 0.0)
                    if sub:
                        best = max(sub, key=lambda c: c.sumR)
                        r = noisy(best, 'm')
                        del_acc[dl]['sumR'].append(r['sumR'])
                        del_acc[dl]['qos'].append(r['qos'])
                        del_acc[dl]['feas'].append(r['feas'])
                        del_acc[dl]['rew15'].append(r['sumR'] - 1.5 * r['pen'])

                # sanity: heuristic oracle-assignment + dominant-user phase + 80/20
                a_or = _oracle_assignment(ch, env.user_pos, env.irs_pos)
                act_or = sorted(set(int(x) for x in a_or if x > 0))
                est = {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
                       'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}
                p_or = oracle_phase_idx(est, np.asarray(a_or, dtype=int), cfg)
                c_or = eval_candidate(np.asarray(a_or, dtype=int),
                                      phi_from_idx(p_or, cfg), p_or, ch, cfg, rate,
                                      act_or, 0.2, 0.0, 'eq', Dk, eps)
                r = rescore_noisy(c_or, c_or.C_wf, ch, cfg, rate, sig_list, Dk, eps)
                san_acc['sumR'].append(r['sumR']); san_acc['qos'].append(r['qos'])
                san_acc['rew'].append(r['sumR'] - 1.5 * r['pen'])

                n_states += 1
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1:>3}/{args.episodes}  states={n_states}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    def mse(x):
        x = np.asarray(x, dtype=float)
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(max(len(x), 2)))

    W = 96
    print("\n" + "=" * W)
    print(f"  V* λ-SWEEP (4-var oracle, {n_states} states, {args.noise_draws}-draw σ² re-eval)")
    print(f"  reward = ΣR − λ·Σ((D_k−R_tot)+/(D_k+ε))²   [infer metric = λ=1.5 row]")
    print("=" * W)
    print(f"  {'λ':>8} | {'E[reward]/step':>14} ± SE     | {'R_tot':>7} | {'QoS%':>6} | {'feas%':>6} | {'nom-rew':>8}")
    print("  " + "-" * (W - 4))
    for lam in LAM_GRID:
        m_rew, se = mse(lam_acc[lam]['rew'])
        m_sr, _ = mse(lam_acc[lam]['sumR'])
        m_q, _ = mse(lam_acc[lam]['qos'])
        m_f, _ = mse(lam_acc[lam]['feas'])
        m_nr, _ = mse(lam_acc[lam]['nom_rew'])
        tag = '∞' if lam >= 1e5 else f"{lam:g}"
        print(f"  {tag:>8} | {m_rew:>14.4f} ± {se:.4f} | {m_sr:>7.4f} | "
              f"{m_q*100:>5.1f}% | {m_f*100:>5.1f}% | {m_nr:>8.4f}")
    print("\n  #MET-CONSTRAINED FRONTIER (greedy-fill Ck, nominal selection):")
    print(f"  {'#met ≥ n':>9} | {'attainable%':>11} | {'R_tot*':>7} | {'noisy QoS%':>10}")
    for n in N_LEVELS:
        ok, _ = mse(con_acc[n]['ok'])
        if con_acc[n]['sumR']:
            m_sr, se = mse(con_acc[n]['sumR']); m_q, _ = mse(con_acc[n]['qos'])
            print(f"  {n:>9} | {ok*100:>10.1f}% | {m_sr:>7.4f} | {m_q*100:>9.1f}%")
        else:
            print(f"  {n:>9} | {ok*100:>10.1f}% |       — |          —")
    print("\n  ROBUST-MARGIN FRONTIER (max ΣR s.t. nominal maximin margin ≥ δ; margin-Ck deploy):")
    print(f"  {'δ (bps/Hz)':>11} | {'attain%':>7} | {'R_tot*':>7} | {'noisy QoS%':>10} | {'feas%':>6} | {'rew@1.5':>8}")
    for dl in DELTA_GRID:
        ok, _ = mse(del_acc[dl]['ok'])
        if del_acc[dl]['sumR']:
            m_sr, se = mse(del_acc[dl]['sumR']); m_q, _ = mse(del_acc[dl]['qos'])
            m_f, _ = mse(del_acc[dl]['feas']); m_r15, _ = mse(del_acc[dl]['rew15'])
            print(f"  {dl:>11.3f} | {ok*100:>6.1f}% | {m_sr:>7.4f} | {m_q*100:>9.1f}% | "
                  f"{m_f*100:>5.1f}% | {m_r15:>8.4f}")
        else:
            print(f"  {dl:>11.3f} | {ok*100:>6.1f}% |       — |          — |      — |        —")

    m_sr, _ = mse(san_acc['sumR']); m_q, _ = mse(san_acc['qos']); m_r, _ = mse(san_acc['rew'])
    print(f"\n  sanity (heuristic oracle-asg + dom-user phase + 80/20 eq + wf-Ck): "
          f"reward@1.5 {m_r:.4f} · R_tot {m_sr:.4f} · QoS {m_q*100:.1f}%")
    print(f"  runtime {time.time()-t0:.0f}s")
    print("=" * W)


if __name__ == "__main__":
    main()
