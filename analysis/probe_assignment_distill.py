"""
probe_assignment_distill.py
---------------------------
Supervised distillation: can a CLASSICAL classifier recover the oracle assignment from
the same input the VQC sees (z_t)? Separates F2 (credit/training) from F3-A3 (representation).

For N states collect (z_t [VQC input], s_t [44 affinity], full-channel feats, ORACLE
assignment via coordinate-ascent on reward, env snapshot). Train an MLP classifier
X → per-user assignment for X ∈ {z_t, s_t, full}, same arch. On held-out states, RE-EVAL
the predicted assignment's reward on the env (restore snapshot).

Read (reward recovery = (pred − policy)/(oracle − policy)):
  • distill-z_t recovers MOST of oracle gap → z_t IS sufficient + a simple classical head
    expresses the mapping → the LIVE policy's gap is CREDIT/TRAINING dynamics (F2), NOT
    info/representation. "VQC too weak to express it" becomes unlikely (a tiny MLP does it
    from the same input) — would then need supervised-VQC/A1 to fully close, but F3-A3 is out.
  • distill-z_t LOW but distill-full HIGH → z_t (AE-compressed) is lossy → REPRESENTATION
    bottleneck (F3-A3): enrich the assignment head's input.
  • even distill-full LOW → oracle assignment not a learnable function of state (sanity flag).

Usage:
  python analysis/probe_assignment_distill.py --ckpt results/result_11/checkpoints/ep_01200
"""

# ── path bootstrap ──────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import time
import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv
from CSI.env_probe import snapshot_env_state, restore_env_state
from probe_critic_ceiling import make_checkpoint_policy
from probe_assignment_oracle import _load_nets, _eval, _coord_ascent
from train_oracle_critic import _state_vec


# ── minimal numpy multi-head softmax MLP classifier ──────────────────────────
class MLPClf:
    def __init__(self, d_in, K, n_cls, hidden=128, lr=3e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.K, self.nc = K, n_cls
        self.W1 = rng.standard_normal((d_in, hidden)) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, K * n_cls)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(K * n_cls)
        self.lr = lr
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

    def _fwd(self, X):
        self._hpre = X @ self.W1 + self.b1
        self._h = np.maximum(0.0, self._hpre)
        return self._h @ self.W2 + self.b2

    def _softmax(self, logits, N):
        L = logits.reshape(N, self.K, self.nc)
        L = L - L.max(2, keepdims=True)
        e = np.exp(L)
        return e / e.sum(2, keepdims=True)

    def step(self, X, Y):
        N = len(X)
        logits = self._fwd(X)
        p = self._softmax(logits, N)
        oh = np.zeros_like(p)
        for u in range(self.K):
            oh[np.arange(N), u, Y[:, u]] = 1.0
        loss = float(-np.log((p * oh).sum(2) + 1e-9).mean())
        dlog = ((p - oh) / N).reshape(N, -1)
        g = {'W2': self._h.T @ dlog, 'b2': dlog.sum(0)}
        dh = (dlog @ self.W2.T) * (self._hpre > 0)
        g['W1'] = X.T @ dh; g['b1'] = dh.sum(0)
        # Adam
        self._t += 1
        for k, pr in self._params().items():
            self._m[k] = 0.9 * self._m[k] + 0.1 * g[k]
            self._v[k] = 0.999 * self._v[k] + 0.001 * g[k] ** 2
            mh = self._m[k] / (1 - 0.9 ** self._t)
            vh = self._v[k] / (1 - 0.999 ** self._t)
            pr -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        return loss

    def predict(self, X):
        N = len(X)
        return self._softmax(self._fwd(X), N).argmax(2)


def _full_feat(env, cfg):
    obs = env._get_obs()
    g = [obs['g_SR'], obs['g_RU'], obs['g_SU']]
    mag = np.concatenate([np.abs(x).ravel() for x in g])
    ang = np.concatenate([np.angle(x).ravel() for x in g])
    blk = env.channels['su_blocked'].astype(float).ravel()
    return np.concatenate([mag, ang, blk])


def _train_clf(Xtr, Ytr, Xte, d_in, K, n_cls, epochs, seed):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    clf = MLPClf(d_in, K, n_cls, seed=seed)
    n = len(Ztr); rng = np.random.default_rng(seed)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, 128):
            mb = idx[s:s + 128]
            clf.step(Ztr[mb], Ytr[mb])
    return clf.predict(Zte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_11/checkpoints/ep_01200')
    ap.add_argument('--episodes', type=int, default=14)
    ap.add_argument('--steps', type=int, default=24)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--noise_collect', type=int, default=4)
    ap.add_argument('--noise_eval', type=int, default=8)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--val_frac', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=20260603)
    ap.add_argument('--out', default='results/result_11/assignment_distill.txt')
    args = ap.parse_args()

    cfg = make_config(); K, M = cfg.K, cfg.M; D_k = cfg.D_k_bps_hz
    lamD = float(getattr(P, 'lambda_D', 1.5)); n_cls = M + 1
    print("=" * 78)
    print(f"  ASSIGNMENT DISTILLATION  ·  ckpt={args.ckpt}")
    print(f"  K={K} M={M} · collect {args.episodes}×{args.steps} · distill MLP-128 · λ_D={lamD}")
    print("=" * 78)

    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps + args.warmup + 2,
                  reward_noise_avg=1)
    policy = make_checkpoint_policy(args.ckpt, cfg)
    actor, nets = _load_nets(args.ckpt)

    Z, S, F, PHI_pol, PHI_orc, SNAP = [], [], [], [], [], []
    t0 = time.time()
    for ep in range(args.episodes):
        env.reset(seed=args.seed * 23 + ep)
        for _ in range(args.warmup):
            env.step(policy(env))
        for _ in range(args.steps):
            obs = env._get_obs(); blk = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, np.full(K, D_k), blk)
            phi_pol, _, info = actor.forward(s_t); z_t = info['z_t']
            sigmas = [env.channel_model.sample_noise_sigma2() for _ in range(args.noise_collect)]
            phi_greedy, _, _ = actor.forward(s_t, greedy=True)
            m_orc = _coord_ascent(env, nets, phi_greedy, cfg, z_t, sigmas, D_k, lamD, M,
                                  'reward', args.passes)
            if m_orc is None:
                continue
            Z.append(z_t.copy()); S.append(_state_vec(env, cfg)); F.append(_full_feat(env, cfg))
            PHI_pol.append(phi_pol.copy()); PHI_orc.append(m_orc['phi'].copy())
            SNAP.append(snapshot_env_state(env))
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)
        if (ep + 1) % max(1, args.episodes // 5) == 0:
            print(f"  collected {ep+1}/{args.episodes} ep, N={len(Z)} ({time.time()-t0:.0f}s)")

    Z, S, F = np.array(Z), np.array(S), np.array(F)
    PHI_pol, PHI_orc = np.array(PHI_pol), np.array(PHI_orc)
    N = len(Z)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N); n_val = int(N * args.val_frac)
    te, tr = perm[:n_val], perm[n_val:]
    print(f"\n  N={N} (train {len(tr)} / test {len(te)}) · z_t={Z.shape[1]} s_t={S.shape[1]} full={F.shape[1]}\n")

    # train classifiers
    preds = {}
    for nm, X in [('z_t', Z), ('s_t', S), ('full', F)]:
        preds[nm] = _train_clf(X[tr], PHI_orc[tr], X[te], X.shape[1], K, n_cls,
                               args.epochs, args.seed)
        print(f"  trained MLP on {nm} (d_in={X.shape[1]})")

    # re-eval reward of each predicted assignment on held-out (restore snapshot, same noise)
    def reward_of(phi_set):
        rs = []
        for j, i in enumerate(te):
            restore_env_state(env, SNAP[i])
            sig = [env.channel_model.sample_noise_sigma2() for _ in range(args.noise_eval)]
            m = _eval(env, nets, phi_set[j] if phi_set.ndim == 2 else phi_set[i],
                      cfg, Z[i], sig, D_k, lamD)
            if m: rs.append((m['reward'], m['R_tot'], m['QoS'], m['IRS_share']))
        return np.array(rs)

    # use SAME restored state + we want same noise across variants → re-eval per state all variants together
    res = {k: [] for k in ['policy', 'oracle', 'z_t', 's_t', 'full']}
    for j, i in enumerate(te):
        restore_env_state(env, SNAP[i])
        sig = [env.channel_model.sample_noise_sigma2() for _ in range(args.noise_eval)]
        for nm, phi in [('policy', PHI_pol[i]), ('oracle', PHI_orc[i]),
                        ('z_t', preds['z_t'][j]), ('s_t', preds['s_t'][j]), ('full', preds['full'][j])]:
            m = _eval(env, nets, phi, cfg, Z[i], sig, D_k, lamD)
            res[nm].append((m['reward'], m['R_tot'], m['QoS'], m['IRS_share']) if m else (np.nan,)*4)
    res = {k: np.array(v) for k, v in res.items()}

    # ── report ──
    L = []
    L.append("=" * 78)
    L.append(f"  ASSIGNMENT DISTILLATION  ·  ckpt={args.ckpt}  ·  N={N} (test {len(te)})")
    L.append("=" * 78)
    pol_r = np.nanmean(res['policy'][:, 0]); orc_r = np.nanmean(res['oracle'][:, 0])
    gap = orc_r - pol_r
    L.append(f"  reward (re-eval on held-out, same noise):")
    L.append(f"  {'source':10s} {'reward':>8s} {'R_tot':>7s} {'QoS%':>6s} {'IRS%':>6s} {'recovery':>9s}  {'IRS/dir acc':>11s}")
    for nm in ['policy', 'oracle', 'z_t', 's_t', 'full']:
        a = res[nm]; rw = np.nanmean(a[:, 0])
        rec = (rw - pol_r) / gap if gap > 1e-9 and nm not in ('policy', 'oracle') else (
            0.0 if nm == 'policy' else 1.0)
        # binary IRS/direct accuracy vs oracle (only for distilled)
        accs = ""
        if nm in ('z_t', 's_t', 'full'):
            pred = preds[nm]; orc = PHI_orc[te]
            acc = float(((pred > 0) == (orc > 0)).mean())
            accs = f"{100*acc:.1f}%"
        recs = f"{100*rec:.0f}%" if nm not in ('policy', 'oracle') else ('—' if nm == 'policy' else '100%')
        L.append(f"  {nm:10s} {rw:+8.3f} {np.nanmean(a[:,1]):7.3f} {np.nanmean(a[:,2]):6.1f} "
                 f"{100*np.nanmean(a[:,3]):6.1f} {recs:>9s}  {accs:>11s}")
    L.append("-" * 78)
    rec_z = (np.nanmean(res['z_t'][:, 0]) - pol_r) / max(gap, 1e-9)
    rec_f = (np.nanmean(res['full'][:, 0]) - pol_r) / max(gap, 1e-9)
    L.append("  VERDICT:")
    L.append(f"    oracle gap over policy = {gap:+.3f} reward · distill-z_t recovers {100*rec_z:.0f}% · full {100*rec_f:.0f}%")
    if rec_z > 0.6:
        L.append(f"    → z_t SUFFICIENT: a tiny classical MLP recovers {100*rec_z:.0f}% of the oracle gap from")
        L.append(f"      the SAME input the VQC sees → the gap is NOT representation/info (F3-A3 OUT) and a")
        L.append(f"      classical head CAN express the mapping → live gap = CREDIT/TRAINING dynamics (F2).")
        L.append(f"      'VQC too weak to represent it' is unlikely; would need supervised-VQC/A1 to fully rule.")
    elif rec_f - rec_z > 0.25:
        L.append(f"    → z_t LOSSY: distill-z_t {100*rec_z:.0f}% ≪ distill-full {100*rec_f:.0f}% → AE-compressed z_t")
        L.append(f"      misses routing-relevant channel info → REPRESENTATION bottleneck (F3-A3). Enrich head input.")
    else:
        L.append(f"    → even full-channel recovers only {100*rec_f:.0f}% → oracle assignment poorly predictable")
        L.append(f"      from state (noisy oracle / needs interaction terms). Re-examine oracle stability.")
    L.append("=" * 78)
    report = "\n".join(L)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n  report → {args.out}")


if __name__ == '__main__':
    main()
