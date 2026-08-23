"""Does a 2*nq-wide latent starve the ASSIGNMENT head of what it needs?

The AE-loss table already says the reconstruction bottleneck is the decoder, not
the latent: at a fixed [32,16] decoder, widening 16 -> 20 -> 24 moved L_ae
15.23 -> 16.23 -> 14.33, non-monotonically, while swapping the decoder to
[128,64] at the NARROWEST latent dropped it to 10.21. And the run with the best
reconstruction (r264) was the worst on task, by 5.4-6.1%.

But reconstruction weights all 44 input dims equally and the policy does not.
A latent can be wide enough to rebuild the state and still drop the particular
directions the assignment head needs. That is the surviving form of the
hypothesis, and it is what this measures.

Predicts the ORACLE ROUTING target (blocked user -> its IRS, the same signal
--assign-aux-weight teaches) from:
    raw 44   the full actor input, the ceiling
    PCA-k    the best LINEAR k-dim compression, for k around the 2*nq values
    z16      r261's ACTUAL trained encoder output

If PCA-16 already matches raw-44, sixteen dimensions are not the constraint and
more qubits cannot buy assignment accuracy. If accuracy climbs with k through
16 -> 20 -> 24, the hypothesis holds and nq is a real lever.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from infer import load_training_cfg, load_agents, _compute_irs_favored


def fit_predict(Xtr, Ytr, Xte, Yte, K, C, hidden=64, epochs=600, lr=5e-3,
                seed=0):
    """Multiclass per user: K independent (M+1)-way heads on a shared trunk."""
    rng = np.random.default_rng(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    d = Xtr.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, K * C))
    b2 = np.zeros(K * C)
    v = [np.zeros_like(x) for x in (W1, b1, W2, b2)]
    onehot = np.zeros((len(Ytr), K, C))
    for i in range(len(Ytr)):
        onehot[i, np.arange(K), Ytr[i]] = 1.0
    for _ in range(epochs):
        h = np.maximum(0.0, Xtr @ W1 + b1)
        z = (h @ W2 + b2).reshape(-1, K, C)
        z = z - z.max(axis=2, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=2, keepdims=True)
        g = ((p - onehot) / len(Xtr)).reshape(-1, K * C)
        gW2, gb2 = h.T @ g, g.sum(0)
        dh = (g @ W2.T) * (h > 0)
        gW1, gb1 = Xtr.T @ dh, dh.sum(0)
        for arr, gr, i in ((W1, gW1, 0), (b1, gb1, 1), (W2, gW2, 2), (b2, gb2, 3)):
            v[i] = 0.9 * v[i] + gr
            arr -= lr * v[i]
    h = np.maximum(0.0, Xte @ W1 + b1)
    pred = (h @ W2 + b2).reshape(-1, K, C).argmax(axis=2)
    return 100.0 * float((pred == Yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/result_261/checkpoints/ep_10000")
    ap.add_argument("--states", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_training_cfg(a.run)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.states + 5)
    actor, phase_net, power_net, ck_net = load_agents(a.run, seed=a.seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)

    X, Z, Y = [], [], []
    obs = env.reset(seed=a.seed)
    for _ in range(a.states):
        blocked = _compute_irs_favored(env)
        s_t = actor.extract_state(obs, demand, blocked)
        phi, _, ainfo = actor.forward(s_t, greedy=True)
        X.append(np.asarray(s_t, dtype=float))
        Z.append(np.asarray(ainfo["z_t"], dtype=float))
        Y.append(np.asarray(blocked, dtype=int))       # oracle routing target
        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)
        obs = env._get_obs()

    X, Z, Y = np.array(X), np.array(Z), np.array(Y)
    C = int(Y.max()) + 1
    n = len(X)
    ntr = int(0.8 * n)
    print("=" * 74)
    print(f"  LATENT BOTTLENECK · K={cfg.K} M={cfg.M} · {n} states")
    print(f"  raw input d_s={X.shape[1]}   trained latent d={Z.shape[1]}   "
          f"target = oracle routing, {C} classes/user")
    print("=" * 74)

    # variance the best linear k-dim compression keeps
    Xc = X - X.mean(0)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    print(f"  {'input':<14}{'dim':>5}{'var kept':>11}{'routing acc':>14}")
    print("  " + "-" * 44)
    acc_raw = fit_predict(X[:ntr], Y[:ntr], X[ntr:], Y[ntr:], cfg.K, C)
    print(f"  {'raw (ceiling)':<14}{X.shape[1]:>5}{100.0:>10.1f}%{acc_raw:>13.1f}%")

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    for k in (8, 12, 16, 20, 24, 32):
        if k >= X.shape[1]:
            continue
        P = Xc @ Vt[:k].T
        acc = fit_predict(P[:ntr], Y[:ntr], P[ntr:], Y[ntr:], cfg.K, C)
        print(f"  {'PCA-' + str(k):<14}{k:>5}{100*ev[:k].sum():>10.1f}%"
              f"{acc:>13.1f}%   {acc - acc_raw:+.1f} vs ceiling", flush=True)

    acc_z = fit_predict(Z[:ntr], Y[:ntr], Z[ntr:], Y[ntr:], cfg.K, C)
    print(f"  {'z_t (trained)':<14}{Z.shape[1]:>5}{'--':>11}{acc_z:>13.1f}%"
          f"   {acc_z - acc_raw:+.1f} vs ceiling")
    print("=" * 74)
    print("  If PCA-16 already sits at the raw ceiling, sixteen dimensions are")
    print("  not the constraint and more qubits cannot buy routing accuracy.")
    print("=" * 74)


if __name__ == "__main__":
    main()
