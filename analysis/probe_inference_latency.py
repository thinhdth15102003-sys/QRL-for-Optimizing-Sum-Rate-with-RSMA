"""
probe_inference_latency.py
--------------------------
Wall-clock INFERENCE latency of the assignment actor (VQC-hybrid vs. classical DNN),
to complement the parameter-budget table (tab:param_budget) with a TIME budget.

Times `actor.forward` (state -> assignment) on real states, greedy, after a warm-up.
Sub-actors (phase/power/Ck) are identical in both pipelines, so the assignment-actor
gap = the pure VQC-vs-DNN inference cost. Uses infer.py's own loaders so cfg/env/
n_shots match the deployed path exactly.

Usage:
  python analysis/probe_inference_latency.py \
     --runs C1-VQC:results/result_117 C1-DNN:results/result_107 \
            C2-VQC:results/result_133 C2-DNN:results/result_138 \
     --n-warm 50 --n-time 500 --n-states 200 --seed 0
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from CSI.env import ISTNEnv
from infer import load_training_cfg, load_agents, _compute_irs_favored


def collect_states(env, actor, demand, n_states, seed):
    states, obs = [], env.reset(seed=seed)
    while len(states) < n_states:
        blocked = _compute_irs_favored(env)
        states.append(actor.extract_state(obs, demand, blocked).copy())
        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)
        obs = env._get_obs()
    return states


def time_forward(actor, states, n_warm, n_time):
    for i in range(n_warm):                       # warm-up (JIT/cache)
        actor.forward(states[i % len(states)], greedy=True)
    ts = np.empty(n_time)
    for i in range(n_time):
        s = states[i % len(states)]
        t0 = time.perf_counter()
        actor.forward(s, greedy=True)
        ts[i] = time.perf_counter() - t0
    return ts * 1e3                               # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='label:run_dir ...')
    ap.add_argument('--n-warm',   type=int, default=50)
    ap.add_argument('--n-time',   type=int, default=500)
    ap.add_argument('--n-states', type=int, default=200)
    ap.add_argument('--seed',     type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print(f"  INFERENCE LATENCY — assignment actor.forward (greedy) · "
          f"warm={args.n_warm} time={args.n_time}")
    print("=" * 78)
    print(f"  {'run':<10}{'K,M':<8}{'actor':<11}{'n_shots':<9}"
          f"{'ms/decision mean±std':<24}{'median':>8}")
    print("  " + "-" * 72)
    res = {}
    for spec in args.runs:
        lbl, run_dir = spec.split(':', 1)
        cfg = load_training_cfg(run_dir)
        env = ISTNEnv(cfg, seed=args.seed, n_steps_ep=200)
        demand = np.full(cfg.K, cfg.D_k_bps_hz)
        actor, *_ = load_agents(run_dir, seed=args.seed)
        amode = 'classical' if 'Classical' in type(actor).__name__ else 'quantum'
        nsh = getattr(actor, 'n_shots', '-')
        states = collect_states(env, actor, demand, args.n_states, args.seed)
        ms = time_forward(actor, states, args.n_warm, args.n_time)
        res[lbl] = (cfg.K, cfg.M, amode, ms.mean(), ms.std(), np.median(ms))
        print(f"  {lbl:<10}{f'{cfg.K},{cfg.M}':<8}{amode:<11}{str(nsh):<9}"
              f"{ms.mean():>7.3f} ± {ms.std():<12.3f}{np.median(ms):>8.3f}")
    print("=" * 78)
    # VQC/DNN ratio per case (pairs sharing the same K,M)
    print("  VQC/DNN latency ratio (median):")
    by_case = {}
    for lbl, (K, M, mode, mu, sd, med) in res.items():
        by_case.setdefault((K, M), {})[mode] = med
    for (K, M), d in sorted(by_case.items()):
        if 'quantum' in d and 'classical' in d:
            print(f"    ({K},{M}): {d['quantum'] / d['classical']:.1f}x "
                  f"(VQC {d['quantum']:.3f} ms vs DNN {d['classical']:.3f} ms)")
    print("=" * 78)


if __name__ == '__main__':
    main()
