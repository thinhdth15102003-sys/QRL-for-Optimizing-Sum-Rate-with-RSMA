"""COMMON-STATE representation probe (clean cross-actor).
Roll ONE reference rollout → collect a fixed state set. Encode the SAME states through
BOTH actors' encoders (VQC r110 AE-latent, DNN r101 MLP-latent). Distill each rep
{z_VQC, z_DNN, raw s_t, full-channel} → the SAME oracle assignment. Metric = per-user
assignment-prediction ACCURACY on held-out (exact class + binary IRS/direct). This isolates
"how much assignment info the latent carries" from the state-distribution/gap confound that
made the per-actor probe non-comparable."""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))
import numpy as np
import params as P
from params import make_config
from CSI.env import ISTNEnv
from probe_assignment_oracle import _load_nets, _coord_ascent
from probe_critic_ceiling import make_checkpoint_policy
from probe_assignment_distill import _train_clf, _full_feat
from train_oracle_critic import _state_vec

VQC_CK = "results/result_110/checkpoints/ep_01600"
DNN_CK = "results/result_101/checkpoints/ep_01900"
R_LOS, EPISODES, STEPS, WARMUP, EPOCHS, PASSES = 0.2, 16, 24, 5, 300, 2
SEED = 20260719

cfg = make_config(R_LoS_km=R_LOS)
K, M = cfg.K, cfg.M; D_k = cfg.D_k_bps_hz
lamD = float(getattr(P, 'lambda_D', 1.5)); n_cls = M + 1

env = ISTNEnv(cfg=cfg, seed=SEED, n_steps_ep=STEPS + WARMUP + 2, reward_noise_avg=1)
policy = make_checkpoint_policy(VQC_CK, cfg)          # roll the VQC policy (states shared by both)
vqc, vnets = _load_nets(VQC_CK)                       # VQC actor + downstream (oracle target)
dnn, _ = _load_nets(DNN_CK)                           # DNN actor (encoder only)
print(f"VQC={VQC_CK}  DNN={DNN_CK}  R_LoS={R_LOS}  common states, {n_cls}-class assignment")

Zv, Zd, S, F, PHI = [], [], [], [], []
for ep in range(EPISODES):
    env.reset(seed=SEED * 23 + ep)
    for _ in range(WARMUP):
        env.step(policy(env))
    for _ in range(STEPS):
        obs = env._get_obs(); blk = env.channels['su_blocked'].astype(int)
        s_t = vqc.extract_state(obs, np.full(K, D_k), blk)      # shared 44-d state
        _, _, iv = vqc.forward(s_t); zv = iv['z_t']             # VQC AE-latent
        _, _, idn = dnn.forward(s_t); zd = idn['z_t']           # DNN MLP-latent (SAME s_t)
        phi_g, _, _ = vqc.forward(s_t, greedy=True)
        sig = [env.channel_model.sample_noise_sigma2() for _ in range(4)]
        m = _coord_ascent(env, vnets, phi_g, cfg, zv, sig, D_k, lamD, M, 'reward', PASSES)
        if m is None:
            continue
        Zv.append(zv.copy()); Zd.append(zd.copy())
        S.append(_state_vec(env, cfg)); F.append(_full_feat(env, cfg)); PHI.append(m['phi'].copy())
        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(env.user_pos, env.irs_pos, env.channels)
    if (ep + 1) % 4 == 0:
        print(f"  collected {ep+1}/{EPISODES}  N={len(Zv)}")

Zv, Zd, S, F, PHI = map(np.array, (Zv, Zd, S, F, PHI))
N = len(Zv); rng = np.random.default_rng(SEED)
perm = rng.permutation(N); n_val = int(N * 0.25); te, tr = perm[:n_val], perm[n_val:]
print(f"\nN={N} (train {len(tr)} / test {len(te)}) · z_VQC={Zv.shape[1]} z_DNN={Zd.shape[1]} "
      f"raw={S.shape[1]} full={F.shape[1]}\n")

print(f"  {'representation':16s} {'exact-acc':>10s} {'IRS/dir-acc':>12s}")
for nm, X in [('z_VQC (AE)', Zv), ('z_DNN (MLP)', Zd), ('raw s_t', S), ('full channel', F)]:
    pred = _train_clf(X[tr], PHI[tr], X[te], X.shape[1], K, n_cls, EPOCHS, SEED)
    tru = PHI[te]
    exact = 100 * (pred == tru).mean()
    binary = 100 * ((pred > 0) == (tru > 0)).mean()
    print(f"  {nm:16s} {exact:9.1f}% {binary:11.1f}%")
print("\n(same states, same oracle target, same probe arch → cross-actor CLEAN.)")
