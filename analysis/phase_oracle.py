"""
analysis/phase_oracle.py
------------------------
Closed-form ORACLE phase index per IRS for EXP-3 supervised PhaseMLP warmup.

Channel model (CSI/rate.py:effective_channels_all):
    h_irs[m, k] = beta[m] · conj(g_SR[m]) · eff_phi[m] · g_RU[m, k]
    eff_phi[m]  = Σ_{n=1..N} exp(j · φ_{m,n})

With all N elements sharing one phase value θ_m  ⇒  eff_phi[m] = N · exp(j θ_m).
The per-IRS scalar channel model collapses element-wise resolution, so optimal
combining means all elements take the SAME phase index — the oracle search
space reduces from L^N (e.g. 4^24 ≈ 3e14) to L per IRS (e.g. 4).

For the dominant user k assigned to IRS m, the IRS-leg phase that ALIGNS the
reflected signal with the direct link g_SU[k] is:

    θ_m^*  =  arg(g_SU[k])  -  arg(g_RU[m,k])  +  arg(g_SR[m])    (mod 2π)

Quantised to the nearest of the n_levels discrete phases. For multi-user
assignments we pick the user maximising β·|g_SR|·|g_RU| (largest IRS-leg
contributor) and align to it — standard dominant-user heuristic used in
supervised warmup of RIS phase networks.

Usage as a library (call from train.py for --phase-warmup pretraining):

    from analysis.phase_oracle import oracle_phase_idx
    target = oracle_phase_idx(channels, phi, cfg)   # (M, N) int, broadcast per IRS

Usage as a CLI (compare live PhaseMLP alignment vs oracle on a checkpoint):

    python analysis/phase_oracle.py --ckpt results/result_N/checkpoints/ep_00XXX --episodes 15

Reports oracle alignment ceiling + live-vs-oracle gap (= EXP-3 headroom).
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


# ── Core library function ────────────────────────────────────────────────────

def oracle_phase_idx(channels: dict,
                     assignment: np.ndarray,
                     cfg) -> np.ndarray:
    """
    Closed-form per-IRS oracle phase index, broadcast across N elements.

    Parameters
    ----------
    channels   : dict with TRUE (or estimated _hat) channels
                 must contain g_SR (M,), g_RU (M, K), g_SU (K,), beta (M,)
    assignment : (K,) int  IRS assignment per user
                 0  = direct  (user routes through g_SU only)
                 1..M = IRS m  (uses 1-based IRS id, matches env action format)
    cfg        : SystemConfig  (uses cfg.M, cfg.N, cfg.phase_levels)

    Returns
    -------
    phase_idx : (M, N) int in {0, …, n_levels-1}
                Per-IRS the SAME index is broadcast across all N elements
                (per-IRS-scalar channel model → optimal = uniform per IRS).
                Inactive IRS rows (no users assigned) default to 0.
    """
    M, N = cfg.M, cfg.N
    levels = cfg.phase_levels            # (L,) rad in [0, 2π)
    L = len(levels)

    g_SR = channels['g_SR']              # (M,) complex
    g_RU = channels['g_RU']              # (M, K) complex
    g_SU = channels['g_SU']              # (K,) complex
    beta = channels['beta']              # (M,) float

    phase_idx = np.zeros((M, N), dtype=int)

    for m in range(M):
        # users routed to IRS m  (assignment uses 1..M for IRS slots)
        assigned_k = np.where(assignment == (m + 1))[0]
        if assigned_k.size == 0:
            continue                       # inactive IRS — leave index 0

        # Brute search over L = n_levels phase indices, pick the one that
        # maximises Σ_{k∈assigned} |h_total[k]|² = |g_SU[k] + h_irs[m,k]|².
        # All-equal-phase ⇒ eff_phi = N · exp(j·levels[idx]).
        irs_coeff = beta[m] * np.conj(g_SR[m]) * g_RU[m, assigned_k]   # (G_m,)
        best_idx, best_obj = 0, -np.inf
        for idx in range(L):
            eff = N * np.exp(1j * levels[idx])
            h_irs = irs_coeff * eff                          # (G_m,)
            h_tot = g_SU[assigned_k] + h_irs                 # (G_m,)
            obj = float(np.sum(np.abs(h_tot) ** 2))
            if obj > best_obj:
                best_obj, best_idx = obj, idx
        phase_idx[m, :] = best_idx

    return phase_idx


# ── Convenience: closed-form (non-brute) when ONE user dominates ─────────────

def oracle_phase_idx_dominant(channels: dict,
                              assignment: np.ndarray,
                              cfg) -> np.ndarray:
    """
    Closed-form variant — picks the dominant-user-target angle directly without
    brute search. Identical to oracle_phase_idx when each IRS has ≤ 1 assigned
    user; for multi-user it's the single-user surrogate (faster but slightly
    sub-optimal for IRS with K_m > 1). Kept as reference / sanity check.

    θ_m^* = arg(g_SU[k_dom]) − arg(g_RU[m, k_dom]) + arg(g_SR[m])    (mod 2π)
    where k_dom = argmax_k  β[m]·|g_SR[m]|·|g_RU[m,k]| over assigned users.
    """
    M, N = cfg.M, cfg.N
    levels = cfg.phase_levels
    g_SR, g_RU, g_SU, beta = (channels['g_SR'], channels['g_RU'],
                              channels['g_SU'], channels['beta'])
    phase_idx = np.zeros((M, N), dtype=int)
    two_pi = 2 * np.pi

    for m in range(M):
        assigned_k = np.where(assignment == (m + 1))[0]
        if assigned_k.size == 0:
            continue
        strength = beta[m] * np.abs(g_SR[m]) * np.abs(g_RU[m, assigned_k])
        k_dom = assigned_k[int(np.argmax(strength))]
        theta_star = (np.angle(g_SU[k_dom])
                      - np.angle(g_RU[m, k_dom])
                      + np.angle(g_SR[m])) % two_pi
        # nearest discrete level (wrap-aware)
        diffs = np.abs(((levels - theta_star + np.pi) % two_pi) - np.pi)
        phase_idx[m, :] = int(np.argmin(diffs))
    return phase_idx


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

    rand_floor = 1.0 / np.sqrt(N)

    q_live, q_oracle = [], []
    irs_win_live, irs_win_oracle, irs_tot = 0, 0, 0
    match_count, oracle_steps = 0, 0      # fraction live-index == oracle-index

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

            ch = env.channels
            for m in active_irs:
                # coherence under live phase
                ang_l = env.phase_model.index_to_phase(phase_live[m])
                q_l = float(np.abs(np.exp(1j * ang_l).sum()) / N)
                # coherence under oracle phase
                ang_o = env.phase_model.index_to_phase(phase_oracle[m])
                q_o = float(np.abs(np.exp(1j * ang_o).sum()) / N)
                q_live.append(q_l);  q_oracle.append(q_o)

                # index match (per-element, all-equal so first elem is enough)
                match_count += int(phase_live[m, 0] == phase_oracle[m, 0])
                oracle_steps += 1

                routed = np.where(phi == (m + 1))[0]
                if routed.size:
                    g_dir = np.abs(ch['g_SU_hat'][routed]) ** 2
                    # live IRS gain magnitude²
                    c_l = ch['beta'][m] * np.abs(ch['g_SR_hat'][m]) * (q_l * N)
                    g_l = (c_l * np.abs(ch['g_RU_hat'][m, routed])) ** 2
                    irs_win_live += int(np.sum(g_l > g_dir))
                    # oracle IRS gain magnitude²
                    c_o = ch['beta'][m] * np.abs(ch['g_SR_hat'][m]) * (q_o * N)
                    g_o = (c_o * np.abs(ch['g_RU_hat'][m, routed])) ** 2
                    irs_win_oracle += int(np.sum(g_o > g_dir))
                    irs_tot += routed.size
            _advance(env)

    q_live, q_oracle = np.asarray(q_live), np.asarray(q_oracle)
    if q_live.size == 0:
        print("No active IRS in any step — agent routed everyone to direct.")
        return
    a_live   = (q_live   - rand_floor) / (1.0 - rand_floor)
    a_oracle = (q_oracle - rand_floor) / (1.0 - rand_floor)
    am_live, am_oracle = float(a_live.mean()), float(a_oracle.mean())
    match_pct = 100.0 * match_count / max(1, oracle_steps)

    print("=" * 74)
    print(f"  PHASE-ORACLE COMPARISON · ckpt={args.ckpt}")
    print(f"  K={K} M={cfg.M} N={N} · {args.episodes}ep×{args.steps}step · "
          f"{'greedy' if greedy else 'sampled'} live policy · "
          f"{q_live.size} active-IRS instances")
    print("=" * 74)
    print(f"  alignment a (0=rand, 1=opt):")
    print(f"     LIVE   PhaseMLP : mean={am_live*100:.1f}%  "
          f"[p25 {np.percentile(a_live,25)*100:.0f}%  p75 {np.percentile(a_live,75)*100:.0f}%]")
    print(f"     ORACLE (closed) : mean={am_oracle*100:.1f}%  "
          f"[p25 {np.percentile(a_oracle,25)*100:.0f}%  p75 {np.percentile(a_oracle,75)*100:.0f}%]")
    print(f"     headroom        : Δ = +{(am_oracle - am_live)*100:.1f} pp")
    print(f"  per-IRS index match (live == oracle): {match_pct:.1f}%")
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
