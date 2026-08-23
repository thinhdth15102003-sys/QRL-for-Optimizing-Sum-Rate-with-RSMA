"""Does the new scale block drown the h-block once LayerNorm is applied?

PowerMLP layer-norms its whole input row. The h-block is divided by its own mean
magnitude precisely so it survives that — the fix that took per-user ranking
retention from 27% to 87% and recovered ~0.85 J. Appending a block on a
different scale can undo it: whichever block dominates the row's mean and std
sets the normalisation, and the others get squashed toward zero.

Measures, on real states, the per-block standard deviation AFTER LayerNorm. If
the h-block's share collapses relative to the no-features baseline, the new
input is not free information — it is the old bug returning.
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/Project/IRS-assisted RSMA Quantum-RL")
from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from RL.sub_actors import _layer_norm
from train import _build_power_state, _power_scale_feats
from analysis.oracle_alloc import phi_from_idx

K, M = 10, 2
cfg = make_config(K=K, M=M, P_S_dBm=50.0, R_LoS_km=0.5)
rate = RateComputer(cfg)
env = ISTNEnv(cfg=cfg, seed=5, n_steps_ep=200, reward_noise_avg=1)
env.reset(seed=5)
rng = np.random.default_rng(0)
n_lat = 24

old_sh, new_sh = [], []
for _ in range(200):
    phi = rng.integers(0, M + 1, size=K)
    pidx = rng.integers(0, len(cfg.phase_levels), size=(M, cfg.N))
    h = rate.effective_channels_all(phi, phi_from_idx(pidx, cfg), env.channels)
    rep = rng.normal(0, 1.0, n_lat)          # z_t is ~unit scale

    for tag, s, acc in (("old", _build_power_state(h, rep), old_sh),
                        ("new", _build_power_state(h, rep, cfg), new_sh)):
        z = _layer_norm(s)
        hb = z[:2 * K]
        if tag == "old":
            rb, sb = z[2 * K:], None
        else:
            sb, rb = z[2 * K:4 * K + 1], z[4 * K + 1:]
        acc.append((hb.std(), (sb.std() if sb is not None else np.nan), rb.std()))
    env.user_pos = env._walk_users(env.user_pos)
    env.channels = env.channel_model.update_user_channels(
        env.user_pos, env.irs_pos, env.channels)

o, n = np.array(old_sh), np.array(new_sh)
print(f"  post-LayerNorm per-block std   (K={K}, {len(o)} states)")
print(f"  {'':<10}{'h-block':>10}{'scale-block':>13}{'rep':>10}")
print(f"  {'baseline':<10}{o[:,0].mean():>10.3f}{'--':>13}{o[:,2].mean():>10.3f}")
print(f"  {'+feats':<10}{n[:,0].mean():>10.3f}{n[:,1].mean():>13.3f}{n[:,2].mean():>10.3f}")
d = 100 * (n[:, 0].mean() / o[:, 0].mean() - 1)
print(f"\n  h-block std change: {d:+.1f}%   "
      f"{'OK' if abs(d) < 15 else '<-- DILUTED, the ranking fix is at risk'}")
print("  raw (pre-norm) magnitudes:")
s = _build_power_state(h, rep, cfg)
print(f"    h  |mean|={np.abs(s[:2*K]).mean():.3f}   "
      f"cap |mean|={np.abs(s[2*K:3*K]).mean():.3f}   "
      f"press |mean|={np.abs(s[3*K:4*K]).mean():.4f}   "
      f"logSNR={s[4*K]:.3f}   rep |mean|={np.abs(s[4*K+1:]).mean():.3f}")
