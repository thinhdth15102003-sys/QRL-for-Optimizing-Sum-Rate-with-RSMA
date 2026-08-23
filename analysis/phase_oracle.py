"""
analysis/phase_oracle.py
------------------------
Closed-form ORACLE phase index per IRS ELEMENT, for EXP-3 supervised PhaseMLP
warmup and as the phase ceiling used by every oracle probe.

Channel model (CSI/rate.py:effective_channels_all), PER-ELEMENT:

    h_irs[m, k] = beta[m] · conj(g_SR[m]) · Σ_{n=1..N} e^{jθ_{m,n}} · g_RU[m, n, k]
                = Σ_n e^{jθ_{m,n}} · c_{m,n,k},
      with      c_{m,n,k} = beta[m] · conj(g_SR[m]) · g_RU[m, n, k]

Every element n sees its OWN Rayleigh draw g_RU[m,n,k], so each element needs
its OWN phase to co-phase its contribution. For a single routed user k the
optimum is closed-form — align every element with the direct link g_SU[k]:

    θ_{m,n}^*  =  arg(g_SU[k])  −  arg(c_{m,n,k})            (mod 2π)
               =  arg(g_SU[k])  +  arg(g_SR[m])  −  arg(g_RU[m,n,k])

quantised to the nearest of the L = 2^bits discrete levels. This reaches the
full coherent-combining ceiling |h_irs| → Σ_n |c_{n,k}| up to the quantisation
loss (for L=4, ≈ sinc(1/4) ≈ 0.90 of the ideal).

For an IRS carrying G > 1 routed users the exact optimum over L^N is
intractable, so we run cheap per-element COORDINATE ASCENT (each element
re-picks its best level with the others held fixed) seeded from the
dominant-user closed form. Cost O(sweeps · N · L · G) — a few hundred flops.

  ⚠ HISTORY: before the per-element refactor, g_RU was (M,K) — one scalar per
  IRS — so every element shared one optimal phase and the search collapsed to
  L candidates per IRS. Reports/logs predating that (e.g. "ORACLE alignment
  100%" @result_14) were measured under the scalar model and do NOT transfer.

  ⚠ NOTE: targets are built from whatever channel dict the caller passes.
  train.py passes env.channels (TRUE g), so the warm-up teacher is a
  perfect-CSI oracle while the deployed policy decides on ĝ. Intentional for a
  supervised teacher, but it does inflate the warm-up ceiling — see the CSI
  split protocol before reading warm-up numbers as achievable.

Usage as a library (train.py --phase-warmup pretraining):

    from analysis.phase_oracle import oracle_phase_idx
    target = oracle_phase_idx(channels, phi, cfg)   # (M, N) int, per-element

Usage as a CLI (compare live PhaseMLP alignment vs oracle on a checkpoint):

    python analysis/phase_oracle.py --ckpt results/result_N/checkpoints/ep_00XXX --episodes 15

Reports oracle coherent-gain ceiling + live-vs-oracle gap (= EXP-3 headroom).
"""

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv

_TWO_PI = 2.0 * np.pi


# ── helpers ──────────────────────────────────────────────────────────────────

def _irs_coeffs(channels: dict, m: int, assigned_k: np.ndarray, cfg) -> np.ndarray:
    """
    Per-element IRS-leg coefficients for IRS m and the routed users.

        c[n, g] = beta[m] · conj(g_SR[m]) · g_RU[m, n, assigned_k[g]]

    Accepts the legacy (M, K) g_RU shape by broadcasting the same value across
    all N elements — reproduces the old per-IRS-scalar behaviour exactly.
    """
    g_RU = channels['g_RU']
    if g_RU.ndim == 2:                                    # legacy (M, K)
        g_col = np.broadcast_to(g_RU[m, assigned_k],
                                (cfg.N, assigned_k.size))
    else:                                                 # (M, N, K)
        g_col = g_RU[m][:, assigned_k]                    # (N, G)
    return channels['beta'][m] * np.conj(channels['g_SR'][m]) * g_col


def est_channels(ch: dict) -> dict:
    """
    Map an env channel dict to the non-_hat key names the oracle helpers expect,
    carrying the ESTIMATED CSI ĝ — the view every DECISION is made on.
    (Achieved rate is scored on the true g elsewhere; see the CSI split.)
    """
    return {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
            'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}


def _nearest_level(theta: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Wrap-aware nearest discrete phase index for each angle in `theta`."""
    diff = np.abs(((theta[..., None] - levels[None, :] + np.pi) % _TWO_PI) - np.pi)
    return np.argmin(diff, axis=-1).astype(int)


def _dominant_init(c: np.ndarray, d: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    Per-element closed form on the dominant routed user (the one with the
    largest total IRS-leg magnitude Σ_n |c[n,g]|): rotate every element so its
    contribution lands on the direct-link angle arg(d[k_dom]).
    """
    k_dom = int(np.argmax(np.abs(c).sum(axis=0)))
    theta = (np.angle(d[k_dom]) - np.angle(c[:, k_dom])) % _TWO_PI    # (N,)
    return _nearest_level(theta, levels)


def _coord_ascent(idx: np.ndarray, c: np.ndarray, d: np.ndarray,
                  E: np.ndarray, n_sweeps: int) -> np.ndarray:
    """
    Per-element coordinate ascent on  Σ_g |d[g] + Σ_n E[idx_n]·c[n,g]|².
    Each element re-picks its best level with the others frozen; stops early
    once a full sweep changes nothing. Monotone non-decreasing by construction.
    """
    N = c.shape[0]
    S = (E[idx][:, None] * c).sum(axis=0)                 # (G,) current IRS leg
    for _ in range(n_sweeps):
        changed = False
        for n in range(N):
            c_n  = c[n]                                   # (G,)
            base = S - E[idx[n]] * c_n                    # (G,) others' sum
            cand = d[None, :] + base[None, :] + E[:, None] * c_n[None, :]
            obj  = (np.abs(cand) ** 2).sum(axis=1)        # (L,)
            best = int(np.argmax(obj))
            if best != idx[n]:
                idx[n] = best
                changed = True
            S = base + E[idx[n]] * c_n
        if not changed:
            break
    return idx


# ── Core library function ────────────────────────────────────────────────────

def oracle_phase_idx(channels: dict,
                     assignment: np.ndarray,
                     cfg,
                     n_sweeps: int = 3) -> np.ndarray:
    """
    Per-ELEMENT oracle phase index.

    Parameters
    ----------
    channels   : dict with TRUE (or estimated) channels — must contain
                 g_SR (M,), g_RU (M, N, K) [or legacy (M, K)], g_SU (K,), beta (M,)
    assignment : (K,) int  IRS assignment per user
                 0  = direct  (user routes through g_SU only)
                 1..M = IRS m  (1-based IRS id, matches env action format)
    cfg        : SystemConfig  (uses cfg.M, cfg.N, cfg.phase_levels)
    n_sweeps   : coordinate-ascent sweeps for multi-user IRS (0 → closed form
                 only). Ignored when an IRS carries a single user, where the
                 closed form is already exactly optimal.

    Returns
    -------
    phase_idx : (M, N) int in {0, …, L-1}, one index PER ELEMENT.
                Inactive IRS rows (no users assigned) default to 0.
    """
    M, N   = cfg.M, cfg.N
    levels = cfg.phase_levels                 # (L,) rad in [0, 2π)
    E      = np.exp(1j * levels)              # (L,)
    g_SU   = channels['g_SU']                 # (K,) complex

    phase_idx = np.zeros((M, N), dtype=int)

    for m in range(M):
        assigned_k = np.where(assignment == (m + 1))[0]
        if assigned_k.size == 0:
            continue                          # inactive IRS — leave index 0

        c = _irs_coeffs(channels, m, assigned_k, cfg)      # (N, G)
        d = g_SU[assigned_k]                               # (G,)

        idx = _dominant_init(c, d, levels)                 # (N,)
        if assigned_k.size > 1 and n_sweeps > 0:
            idx = _coord_ascent(idx, c, d, E, n_sweeps)
        phase_idx[m, :] = idx

    return phase_idx


# ── Convenience: pure closed form, no search ─────────────────────────────────

def oracle_phase_idx_dominant(channels: dict,
                              assignment: np.ndarray,
                              cfg) -> np.ndarray:
    """
    Closed-form-only variant — per-element dominant-user alignment with NO
    coordinate ascent. Exactly optimal when each IRS carries ≤ 1 user (then it
    equals oracle_phase_idx); a single-user surrogate otherwise. Kept as the
    fast path / sanity reference.
    """
    return oracle_phase_idx(channels, assignment, cfg, n_sweeps=0)


# ── Generic phase search for an ARBITRARY objective ─────────────────────────

def oracle_phase_search(channels: dict,
                        assignment: np.ndarray,
                        cfg,
                        score_fn,
                        n_sweeps: int = 1,
                        include_uniform: bool = True,
                        seed_pidx=None):
    """
    Best (M,N) phase index for a caller-supplied scalar objective.

    Probes need the phase that maximises THEIR metric (reward, R_tot, #met),
    not coherent gain — but the per-element closed form is still the right
    starting point. This does:
      ① seed  : oracle_phase_idx (per-element, maximises coherent gain),
                ALWAYS scored, so the result never underperforms it;
      ② restarts: the L^|active| uniform-row patterns (what these probes used
                to enumerate on their own — cheap, kept as extra candidates);
      ③ ascent: per-element coordinate ascent, `n_sweeps` passes, early-exit
                when a full sweep changes nothing.

    ⚠ Under per-element g_RU the true space is L^N PER IRS, so ② alone is not
    an oracle — it was exhaustive only while every element shared one level.

    Parameters
    ----------
    score_fn : callable (M,N) int array -> float. Higher is better. May return
               None / NaN for an infeasible candidate; treated as -inf.
    n_sweeps : 0 disables ③ (use when each score_fn call is expensive — it costs
               |active|·N·(L-1) evaluations per sweep).

    Returns (best_pidx, best_score, n_evals).
    """
    import itertools

    M, N = cfg.M, cfg.N
    L      = len(cfg.phase_levels)
    active = [m for m in range(M) if np.any(assignment == (m + 1))]

    def _score(p):
        v = score_fn(p)
        if v is None:
            return -np.inf
        v = float(v)
        return -np.inf if np.isnan(v) else v

    pidx = (oracle_phase_idx(channels, assignment, cfg) if seed_pidx is None
            else np.asarray(seed_pidx, dtype=int).copy())
    best_p, best_s, n_ev = pidx.copy(), _score(pidx), 1
    if not active:
        return best_p, best_s, n_ev

    # ② uniform-row restarts (over ACTIVE IRS only — inactive rows are inert)
    if include_uniform:
        for combo in itertools.product(range(L), repeat=len(active)):
            trial = pidx.copy()
            for mi, lev in zip(active, combo):
                trial[mi, :] = lev
            s = _score(trial); n_ev += 1
            if s > best_s:
                best_s, best_p = s, trial.copy()

    # ③ per-element coordinate ascent
    pidx = best_p.copy()
    for _ in range(n_sweeps):
        changed = False
        for mi in active:
            for el in range(N):
                cur = int(pidx[mi, el])
                for l in range(L):
                    if l == cur:
                        continue
                    trial = pidx.copy(); trial[mi, el] = l
                    s = _score(trial); n_ev += 1
                    if s > best_s:
                        best_s, pidx[mi, el] = s, l
                        best_p, changed = pidx.copy(), True
        if not changed:
            break
    return best_p, best_s, n_ev


# ── IRS-path ceiling (shared by every probe that scores IRS vs direct) ───────

def irs_optimal_gain_mag(channels: dict, hat: bool = True) -> np.ndarray:
    """
    (M, K) best achievable IRS-path amplitude, perfectly co-phased:

        |h_irs[m,k]|_opt = beta[m] · |g_SR[m]| · Σ_n |g_RU[m,n,k]|

    Ignores the 2-bit quantisation loss (≈ sinc(1/4) = 0.90), so it is an upper
    bound — the same convention the probes used before the refactor.

    ⚠ Under the OLD scalar g_RU this was `beta·|g_SR|·N·|g_RU[m,k]|`, since every
    element carried the same channel. Per-element it is the SUM of N independent
    magnitudes, which is ≈ N·E|g| but no longer a clean N× factor. Probes that
    still write `N * abs(g_RU[m,k])` are wrong — call this instead.

    hat : use the ESTIMATED channels (default, the decision view) or the TRUE g.
    """
    sfx  = '_hat' if hat else ''
    g_SR = channels['g_SR' + sfx]                     # (M,)
    g_RU = channels['g_RU' + sfx]                     # (M,N,K) or legacy (M,K)
    if g_RU.ndim != 3:
        raise ValueError(
            f"g_RU must be per-element (M,N,K); got {g_RU.shape}. A 2-D g_RU is "
            "the pre-refactor scalar model — the N-fold ceiling is not recoverable "
            "from it without knowing N, so fix the caller rather than guessing.")
    coeff = channels['beta'] * np.abs(g_SR)           # (M,)
    amp   = np.abs(g_RU).sum(axis=1)                  # (M,K) coherent sum over n
    return coeff[:, None] * amp                       # (M,K)


# ── Coherent-gain quality metric ─────────────────────────────────────────────

def coherent_gain_quality(channels: dict, assignment: np.ndarray,
                          phase_idx: np.ndarray, cfg, m: int):
    """
    Per-routed-user combining efficiency for IRS m under `phase_idx`.

        q[g]      = |Σ_n e^{jθ_n} c[n,g]|  /  Σ_n |c[n,g]|          ∈ [0, 1]
        q_rand[g] = sqrt(π/4 · Σ_n |c[n,g]|²) / Σ_n |c[n,g]|

    q = 1 is perfect co-phasing (unreachable with L levels); q_rand is the
    expected efficiency of uniformly random phases — the honest zero point.
    Returns (q, q_rand, h_irs, d) with h_irs the achieved IRS leg (G,).

    Replaces the pre-refactor metric |Σ_n e^{jθ_n}|/N, which measured how much
    the ELEMENTS AGREE. That was right only while all elements shared one
    optimal phase; per-element they SHOULD disagree, so element spread no
    longer says anything about alignment.
    """
    assigned_k = np.where(assignment == (m + 1))[0]
    if assigned_k.size == 0:
        z = np.zeros(0)
        return z, z, z.astype(complex), z.astype(complex)

    c     = _irs_coeffs(channels, m, assigned_k, cfg)          # (N, G)
    E_sel = np.exp(1j * cfg.phase_levels[phase_idx[m]])        # (N,)
    h_irs = (E_sel[:, None] * c).sum(axis=0)                   # (G,)

    a_sum = np.abs(c).sum(axis=0)                              # (G,)
    a_sq  = (np.abs(c) ** 2).sum(axis=0)                       # (G,)
    safe  = np.maximum(a_sum, 1e-300)
    q      = np.abs(h_irs) / safe
    q_rand = np.sqrt(np.pi / 4.0 * a_sq) / safe
    return q, q_rand, h_irs, channels['g_SU'][assigned_k]


# ── CLI: evaluate live PhaseMLP alignment vs oracle on a checkpoint ──────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_11/checkpoints/ep_01400')
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260605)
    ap.add_argument('--sampled', action='store_true',
                    help='use sampled phase (live) instead of greedy (default greedy)')
    args = ap.parse_args()

    cfg = make_config(); K = cfg.K; N = cfg.N
    greedy = not args.sampled
    env = ISTNEnv(cfg=cfg, seed=args.seed,
                  n_steps_ep=args.steps, reward_noise_avg=1)

    ckpt = args.ckpt
    if (os.path.isdir(os.path.join(ckpt, 'agents'))
            and not os.path.isfile(os.path.join(ckpt, 'actor_config.json'))):
        ckpt = os.path.join(ckpt, 'agents')
    from RL import QuantumActor, PhaseMLP
    from train import _build_phase_state, _get_active_irs
    actor     = QuantumActor.from_dir(ckpt, seed=0)
    phase_net = PhaseMLP.from_dir(ckpt, seed=0)
    actor.n_shots = P.n_shots_train

    a_live, a_oracle = [], []             # normalised alignment, per routed user
    irs_win_live, irs_win_oracle, irs_tot = 0, 0, 0
    match_count, match_tot = 0, 0         # per-ELEMENT live-index == oracle-index

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            obs     = env._get_obs()
            demand  = np.full(K, cfg.D_k_bps_hz)
            blocked = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, demand, blocked)
            phi, _, info = actor.forward(s_t, greedy=greedy)
            active_irs = _get_active_irs(phi)
            if active_irs.size == 0:
                _advance(env); continue

            s_phase = _build_phase_state(env.channels, phi, cfg, info['z_t'])
            phase_live, _, _ = phase_net.forward(s_phase, active_irs, greedy=greedy)
            phase_oracle = oracle_phase_idx(env.channels, phi, cfg)

            for m in active_irs:
                q_l, q_r, h_l, d = coherent_gain_quality(
                    env.channels, phi, phase_live, cfg, m)
                q_o, _,   h_o, _ = coherent_gain_quality(
                    env.channels, phi, phase_oracle, cfg, m)
                if q_l.size == 0:
                    continue
                denom = np.maximum(1.0 - q_r, 1e-12)
                a_live.extend(((q_l - q_r) / denom).tolist())
                a_oracle.extend(((q_o - q_r) / denom).tolist())

                # per-element index agreement (elements now legitimately differ)
                match_count += int(np.sum(phase_live[m] == phase_oracle[m]))
                match_tot   += N

                g_dir = np.abs(d) ** 2
                irs_win_live   += int(np.sum(np.abs(h_l) ** 2 > g_dir))
                irs_win_oracle += int(np.sum(np.abs(h_o) ** 2 > g_dir))
                irs_tot        += d.size
            _advance(env)

    a_live, a_oracle = np.asarray(a_live), np.asarray(a_oracle)
    if a_live.size == 0:
        print("No active IRS in any step — agent routed everyone to direct.")
        return
    am_live, am_oracle = float(a_live.mean()), float(a_oracle.mean())
    match_pct = 100.0 * match_count / max(1, match_tot)

    print("=" * 74)
    print(f"  PHASE-ORACLE COMPARISON · ckpt={args.ckpt}")
    print(f"  K={K} M={cfg.M} N={N} · {args.episodes}ep×{args.steps}step · "
          f"{'greedy' if greedy else 'sampled'} live policy · "
          f"{a_live.size} routed-user instances")
    print("=" * 74)
    print(f"  coherent-gain alignment a  (0 = random phases, 1 = perfect co-phasing):")
    print(f"     LIVE   PhaseMLP : mean={am_live*100:.1f}%  "
          f"[p25 {np.percentile(a_live,25)*100:.0f}%  p75 {np.percentile(a_live,75)*100:.0f}%]")
    print(f"     ORACLE (per-el) : mean={am_oracle*100:.1f}%  "
          f"[p25 {np.percentile(a_oracle,25)*100:.0f}%  p75 {np.percentile(a_oracle,75)*100:.0f}%]")
    print(f"     headroom        : Δ = +{(am_oracle - am_live)*100:.1f} pp")
    print(f"  per-ELEMENT index match (live == oracle): {match_pct:.1f}%  "
          f"(chance = {100.0/len(cfg.phase_levels):.0f}%)")
    if irs_tot:
        print(f"  LIVE   IRS beats direct: {100.0*irs_win_live/irs_tot:.1f}%  "
              f"({irs_win_live}/{irs_tot})")
        print(f"  ORACLE IRS beats direct: {100.0*irs_win_oracle/irs_tot:.1f}%  "
              f"({irs_win_oracle}/{irs_tot})")
    print("-" * 74)
    gap = am_oracle - am_live
    if gap < 0.05:
        print(f"  ✅ PhaseMLP ≈ ORACLE ({gap*100:.1f} pp gap) — warmup likely won't help.")
    elif gap < 0.20:
        print(f"  🟡 PhaseMLP within {gap*100:.0f} pp of oracle — modest EXP-3 headroom.")
    else:
        print(f"  ❌ PhaseMLP {gap*100:.0f} pp BELOW oracle — STRONG case for EXP-3 warmup.")
    print("=" * 74)


def _advance(env):
    """Advance mobility one step without re-applying an action (mirror probes)."""
    env.user_pos = env._walk_users(env.user_pos)
    env.channels = env.channel_model.update_user_channels(
        env.user_pos, env.irs_pos, env.channels)


if __name__ == '__main__':
    main()
