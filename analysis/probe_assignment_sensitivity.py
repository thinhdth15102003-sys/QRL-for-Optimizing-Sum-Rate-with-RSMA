"""
probe_assignment_sensitivity.py
-------------------------------
Counterfactual ASSIGNMENT SENSITIVITY (Test 3) + Direct-only cross-check.

Holds POLICY downstream (phase/power/Ck) FIXED — same level as `live_pol` in
probe_assignment_decomp — and re-evaluates QoS + R_tot + reward on the SAME
rollout states under several assignment vectors q (q in {0..M}^K, 0 = direct):

  live        policy's sampled assignment            (sanity: ≈ decomp live_pol)
  oracle      coord-ascent reward-optimal assignment  (≈ decomp orw_pol)
  forced_dir  everyone -> direct (q ≡ 0)              [Direct-only cross-check]
  random      uniform-random q in {0..M}
  break_XX%   oracle with XX% of users randomly re-routed  (partial-correct)

Reads:
  - oracle -> break_XX% QoS slope: steep drop => env HIGHLY SENSITIVE to
    assignment => supports the combinatorial-assignment bottleneck thesis.
  - forced_dir QoS: where the reward-optimal (lambda=1.5) gives up QoS vs the
    pure-QoS all-direct route.

Auto-detects K/M from the ckpt's actor_config.json (cross-case safe).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, argparse, time, json
import params as P
from params import make_config
from CSI.env import ISTNEnv
from probe_assignment_oracle import _load_nets, _eval, _coord_ascent
from probe_critic_ceiling import make_checkpoint_policy


def _perturb(phi, frac, M, rng):
    """Keep `phi` (oracle) but re-route a `frac` fraction of users to a random
    DIFFERENT choice in {0..M}."""
    phi2 = np.array(phi).copy()
    K = len(phi2)
    n = int(round(frac * K))
    if n <= 0:
        return phi2
    idx = rng.choice(K, size=min(n, K), replace=False)
    for i in idx:
        choices = [c for c in range(M + 1) if c != int(phi2[i])]
        phi2[i] = rng.choice(choices)
    return phi2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_6/checkpoints/ep_01200')
    ap.add_argument('--episodes', type=int, default=6)
    ap.add_argument('--steps', type=int, default=15)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--noise_avg', type=int, default=5)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--breaks', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    ap.add_argument('--seed', type=int, default=20260608)
    ap.add_argument('--out', default='results/result_6/assignment_sensitivity.txt')
    args = ap.parse_args()

    # --- auto-detect case dims from ckpt (read config from /agents if present) ---
    raw_ckpt = args.ckpt
    cfgread = raw_ckpt
    if (os.path.isdir(os.path.join(raw_ckpt, 'agents'))
            and not os.path.isfile(os.path.join(raw_ckpt, 'actor_config.json'))):
        cfgread = os.path.join(raw_ckpt, 'agents')
    with open(os.path.join(cfgread, 'actor_config.json')) as f:
        ac = json.load(f)
    K = int(ac.get('K', P.K)); M = int(ac.get('B', P.M))
    if K != P.K or M != P.M:
        P_S = {5: 50.0, 10: 70.0, 20: 100.0}.get(K, P.P_S_dBm)
        print(f"  auto-detect ckpt K={K} M={M} (params has K={P.K} M={P.M}) -> override cfg")
        cfg = make_config(K=K, M=M, P_S_dBm=P_S)
    else:
        cfg = make_config()
    D_k = cfg.D_k_bps_hz; lamD = float(getattr(P, 'lambda_D', 1.5))
    rng = np.random.default_rng(args.seed)

    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps + args.warmup + 2,
                  reward_noise_avg=1)
    policy = make_checkpoint_policy(raw_ckpt, cfg)
    actor, nets = _load_nets(raw_ckpt)

    labels = ['live', 'oracle', 'forced_dir', 'random'] + \
             [f'break_{int(b*100)}%' for b in args.breaks]
    rows = {k: [] for k in labels}

    t0 = time.time(); n_states = 0
    for ep in range(args.episodes):
        env.reset(seed=args.seed * 19 + ep)
        for _ in range(args.warmup):
            env.step(policy(env))
        for _ in range(args.steps):
            obs = env._get_obs()
            blocked = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, np.full(K, D_k), blocked)
            phi_pol, _, info = actor.forward(s_t)
            z_t = info['z_t']
            sigmas = [env.channel_model.sample_noise_sigma2() for _ in range(args.noise_avg)]

            orc = _coord_ascent(env, nets, phi_pol, cfg, z_t, sigmas, D_k, lamD,
                                M, 'reward', args.passes)
            if orc is None:
                continue
            oracle_phi = orc['phi']

            variants = {
                'live':       phi_pol,
                'oracle':     oracle_phi,
                'forced_dir': np.zeros_like(phi_pol),
                'random':     rng.integers(0, M + 1, size=K).astype(phi_pol.dtype),
            }
            for b in args.breaks:
                variants[f'break_{int(b*100)}%'] = _perturb(oracle_phi, b, M, rng)

            tmp = {}; ok = True
            for k, phi in variants.items():
                m = _eval(env, nets, np.asarray(phi), cfg, z_t, sigmas, D_k, lamD)
                if m is None:
                    ok = False; break
                tmp[k] = m
            if not ok:
                continue
            for k, m in tmp.items():
                rows[k].append((m['R_tot'], m['QoS'], m['reward'], m['IRS_share']))
            n_states += 1
        print(f"    {ep+1}/{args.episodes} ep ({n_states} states, {time.time()-t0:.0f}s)")

    lines = []
    lines.append("=" * 72)
    lines.append(f"  ASSIGNMENT SENSITIVITY  ·  K={K} M={M} N={cfg.N} · lambda_D={lamD} D_k={D_k}")
    lines.append(f"  ckpt={args.ckpt} · {n_states} states · downstream=POLICY (held fixed)")
    lines.append("=" * 72)
    lines.append(f"  {'variant':<12} {'R_tot':>7} {'QoS%':>7} {'reward':>8} {'IRS%':>7}")
    qos = {}
    for k in labels:
        a = np.array(rows[k])
        if len(a) == 0:
            continue
        qos[k] = a[:, 1].mean()
        lines.append(f"  {k:<12} {a[:,0].mean():>7.3f} {a[:,1].mean():>7.1f} "
                     f"{a[:,2].mean():>+8.3f} {a[:,3].mean()*100:>7.1f}")
    lines.append("=" * 72)
    if 'oracle' in qos and 'break_20%' in qos:
        drop = qos['oracle'] - qos['break_20%']
        verdict = 'HIGH sensitivity (combinatorial bottleneck supported)' if drop >= 8 \
                  else 'LOW sensitivity'
        lines.append(f"  Δ QoS oracle->break_20% = {drop:+.1f}pt   {verdict}")
    if 'random' in qos and 'oracle' in qos:
        lines.append(f"  Δ QoS oracle->random    = {qos['oracle']-qos['random']:+.1f}pt")
    if 'forced_dir' in qos:
        lines.append(f"  forced_dir QoS = {qos['forced_dir']:.1f}%   "
                     f"(Direct-only cross-check; feasibility probe said ~100%)")
    print("\n".join(lines))
    with open(args.out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  report -> {args.out}")


if __name__ == '__main__':
    main()
