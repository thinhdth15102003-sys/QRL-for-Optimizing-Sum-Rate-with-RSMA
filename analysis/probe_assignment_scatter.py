"""
probe_assignment_scatter.py — "assignment law" figure for the paper ablation:
per-step (#blocked, #IRS-assigned) of the trained agents vs the theory rule
(assign exactly the blocked set; over-assignment benign at P50 — admission
marginal ~ 0). Cases 1+2, VQC+DNN, greedy eval on the SAME checkpoints as the
paper's main tables. Saves Research Paper/figures/assignment_law.{pdf,png}.

Usage: python analysis/probe_assignment_scatter.py [--episodes 25]
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer import (load_agents, load_training_cfg, _compute_blocked,
                   _build_phase_state, _build_ck_state,
                   _get_active_irs, _get_active_irs_ids)
from CSI.env import ISTNEnv

CASES = {
    1: [("VQC", "results/result_111/checkpoints/ep_01000"),
        ("DNN", "results/result_104/checkpoints/ep_01000")],
    2: [("VQC", "results/result_110/checkpoints/ep_01600"),
        ("DNN", "results/result_101/checkpoints/ep_01900")],
}
C_COL = {"VQC": "#1f6feb", "DNN": "#e8710a"}


def eval_counts(run_dir, n_eps, n_steps=200, seed=42):
    """Per-step (n_blocked, n_irs_assigned, n_blocked_on_own_irs)."""
    cfg = load_training_cfg(run_dir)
    env = ISTNEnv(cfg, seed=seed, n_steps_ep=n_steps)
    actor, phase_net, power_net, ck_net = load_agents(run_dir, seed=seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    rows = []
    for _ in range(n_eps):
        obs = env.reset()
        for _ in range(n_steps):
            blocked = _compute_blocked(env)
            s_t = actor.extract_state(obs, demand, blocked)
            phi, _, ainfo = actor.forward(s_t, greedy=True)
            z_t = ainfo['z_t']
            rep = ainfo.get('o_hat', z_t) if getattr(actor, 'NO_AE', False) else z_t
            act_irs, act_ids = _get_active_irs(phi), _get_active_irs_ids(phi)
            s_ph = _build_phase_state(env.channels, phi, cfg, rep)
            p_idx, _, _ = phase_net.forward(s_ph, act_irs, greedy=True)
            Phi = env.phase_model.build_phi(env.phase_model.index_to_phase(p_idx))
            h_eff = env.rate_computer.effective_channels_all(phi, Phi, env.channels)
            s_pw = np.concatenate([h_eff.real, h_eff.imag])
            if power_net.d_s > s_pw.shape[0]:
                s_pw = np.concatenate([s_pw, rep])
            w_c, w_p, _, _ = power_net.forward(s_pw, act_ids)
            partial = env.rate_computer.compute_rates_partial(
                phi, Phi, env.channels, w_p, w_c, active_irs_ids=act_ids)
            s_ck = _build_ck_state(demand, partial['R_private'],
                                   partial['R_c_group'], phi, cfg)
            C_k, _, _ = ck_net.forward(s_ck, phi, partial['R_c_group'])
            # ---- record BEFORE stepping (mask/assignment of this step) ----
            nearest = np.argmin(env.channels['d_IRS_U'], axis=0) + 1   # (K,) 1..M
            n_blk  = int(blocked.sum())
            n_irs  = int((phi > 0).sum())
            n_own  = int(((phi > 0) & blocked & (phi == nearest)).sum())
            rows.append((n_blk, n_irs, n_own))
            obs, _, _, _ = env.step({'assignment': phi, 'phase_idx': p_idx,
                                     'w_p': w_p, 'w_c_vec': w_c, 'C_k': C_k})
    return np.array(rows)


def main(n_eps):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    plt.rcParams.update({"font.size": 10})
    for ax, (case, runs) in zip(axes, CASES.items()):
        K = 5 if case == 1 else 10
        # theory line + benign over-assignment region
        xs = np.arange(0, K + 1)
        ax.fill_between(xs, xs, np.minimum(xs + 2, K), color="0.85", alpha=0.6,
                        label="over-assignment region" if case == 1 else None, zorder=0)
        ax.plot(xs, xs, "k--", lw=1.2,
                label="assign exactly the blocked set" if case == 1 else None, zorder=1)
        for off, (name, run_dir) in zip((-0.13, +0.13), runs):
            rows = eval_counts(run_dir, n_eps)
            blk, irs, own = rows[:, 0], rows[:, 1], rows[:, 2]
            uniq, cnt = np.unique(rows[:, :2], axis=0, return_counts=True)
            ax.scatter(uniq[:, 0] + off, uniq[:, 1],
                       s=8 + 110 * cnt / cnt.max(), alpha=0.8,
                       color=C_COL[name], edgecolors="none",
                       label=("HQC-HAC (VQC)" if name == "VQC" else "DNN") if case == 1 else None,
                       zorder=2)
            n = len(rows)
            print(f"C{case} {name}: steps={n}  mean Blk={blk.mean():.2f}  mean Σirs={irs.mean():.2f}"
                  f"  over(Σirs>Blk)={100*np.mean(irs > blk):.1f}%  under={100*np.mean(irs < blk):.1f}%"
                  f"  blocked→own-IRS={100*own.sum()/max(np.minimum(blk, irs).sum(),1):.1f}%"
                  f"  |Σirs−Blk| mean={np.abs(irs-blk).mean():.2f}")
        ax.set_xlabel("blocked users (per step)")
        if case == 1:
            ax.set_ylabel("IRS-assigned users (per step)")
        ax.set_title(f"Case {case} ($K{{=}}{K}$)", fontsize=10)
        ax.set_xticks(xs); ax.set_yticks(xs)
        ax.set_xlim(-0.5, K + 0.5); ax.set_ylim(-0.5, K + 0.5)
        ax.grid(True, alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join("Research Paper", "figures", "assignment_law")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=160, bbox_inches="tight")
    print("saved ->", out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=25)
    args = ap.parse_args()
    main(args.episodes)
