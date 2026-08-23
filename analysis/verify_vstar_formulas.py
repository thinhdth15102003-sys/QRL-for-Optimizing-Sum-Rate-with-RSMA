"""Validate the V*/frontier closed-forms (worksheet §7) against probe_v_star runs.

F1  V*_rate = E[log2(1 + Gamma_max)]  (single-user WTA, all power private, aligned phase)
    vs probe lambda=0 row (post-WTA-patch): C1 10.9345, C2 11.7122 -> expect EXACT match.
F2  Flat-frontier level: K log2(K/(K-1)) (+ common Sum f_c) vs serve-all frontier row.
F3  lambda_cliff = (V*_rate - R_serve) / ((K-1) (Dk/(Dk+eps))^2)
    vs probe transition brackets: C1 in (2, 3), C2 in (1, 1.5).
Probe logs: analysis/data/vstar_c{1,2}_ramp0{5,3}_500states.txt
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import make_config
from CSI.env import ISTNEnv
from analysis.phase_oracle import irs_optimal_gain_mag

CASES = [
    # [07-11 refresh] paper-anchor regimes (probe logs vstar_c1_ramp03 / vstar_c2_ramp05).
    # Old 07-03 anchors: ('C1 ramp0.5', r106/ep2000, 10.9345, 3.0480) · ('C2 ramp0.3', r98/ep2000, 11.7122, 1.6325).
    ('C1 ramp0.3', 'results/result_111/checkpoints/ep_01000', 10.9633, 3.1089, (2.0, 3.0)),
    ('C2 ramp0.5', 'results/result_110/checkpoints/ep_01600', 11.7233, 1.6100, (1.0, 1.5)),
]

for name, ckpt, probe_l0, probe_serveall, cliff_bracket in CASES:
    d = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ckpt))
    hp = None
    for _ in range(4):
        d = os.path.dirname(d)
        p = os.path.join(d, 'hyperparameters.json')
        if os.path.isfile(p):
            hp = json.load(open(p)); break
    s = hp['system']
    keys = ['K','M','N','P_S_dBm','D_k_bps_hz','R_LoS_km','irs_spawn_radius_frac',
            'user_free_radius_frac','kappa','noise_var_dBW','epsilon_qp','beta_blocking',
            'beta_IRS','d_block_km','path_loss_exp','user_speed_mps',
            'balanced_blocked_spawn','h_IRS_km']
    cfg = make_config(**{k: s[k] for k in keys if k in s})
    K, M, N, Dk, eps = cfg.K, cfg.M, cfg.N, cfg.D_k_bps_hz, cfg.epsilon_qp
    env = ISTNEnv(cfg=cfg, seed=42 + 777, n_steps_ep=200, reward_noise_avg=1)

    l0_nom, l0_noisy = [], []
    for ep in range(50):
        env.reset(seed=42 + ep)
        for st in range(200):
            if st % 20 == 0:
                ch = env.channels
                g_dir = np.abs(ch['g_SU_hat']) ** 2                       # (K,)
                g_irs = irs_optimal_gain_mag(ch) ** 2                      # (M,K) aligned
                g_best = np.maximum(g_dir, g_irs.max(axis=0))              # (K,)
                g_max = float(g_best.max())
                l0_nom.append(np.log2(1 + g_max * cfg.P_S / cfg.sigma2))
                draws = [env.channel_model.sample_noise_sigma2() for _ in range(16)]
                l0_noisy.append(np.mean([np.log2(1 + g_max * cfg.P_S / s2) for s2 in draws]))
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(env.user_pos, env.irs_pos, env.channels)

    f1_nom, f1_noisy = float(np.mean(l0_nom)), float(np.mean(l0_noisy))
    f2 = K * np.log2(K / (K - 1))
    pen_unit = (Dk / (Dk + eps)) ** 2
    cliff = (f1_noisy - probe_serveall) / ((K - 1) * pen_unit)
    print(f"{name}: K={K} M={M}")
    print(f"  F1 V*_rate: formula nominal {f1_nom:.3f} · 16-draw {f1_noisy:.3f}  vs probe λ0 {probe_l0:.3f}"
          f"  (probe/formula = {probe_l0/f1_noisy:.3f}; probe ≤ formula expected)")
    print(f"  F2 flat level: K·log2(K/(K−1)) = {f2:.4f}  vs probe serve-all {probe_serveall:.4f}"
          f"  (Δ = common-stream part {probe_serveall - f2:+.3f})")
    print(f"  F3 λ_cliff = {cliff:.3f}  vs probe bracket {cliff_bracket}")
    print()
