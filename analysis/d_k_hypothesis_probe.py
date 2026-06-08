"""
d_k_hypothesis_probe.py
-----------------------
Verify the D_k=0.15 hypothesis: does the trained RL policy maintain QoS via SMART
IRS-routing + RSMA power allocation, or does it just mimic Direct-only?

Per user-confirmed design:
  (a) Behavioral metrics — corr(IRS,blocked), corr(power,blocked), per-user QoS
  (b) Baselines — Direct-only, All-IRS, RL, Equal-power-mix (isolate routing vs power)
  (c) Decision matrix verdict

Usage:
  python analysis/d_k_hypothesis_probe.py --ckpt results/result_13/checkpoints/ep_00600/agents \
      --episodes 12 --steps 20 --seed 0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv
from RL.quantum_actor import QuantumActor
from RL.sub_actors import PhaseMLP, PowerMLP, CkMLP

# Re-use train.py helpers — same code path the trainer follows.
from train import (
    _build_phase_state,
    _build_ck_state,
    _get_active_irs,
    _get_active_irs_ids,
    _compute_blocked,
)


# ── load all 4 nets from ckpt ─────────────────────────────────────────────────
def _load_nets(ckpt_dir, cfg):
    actor = QuantumActor.from_dir(ckpt_dir, seed=0)
    actor.n_shots = getattr(P, "n_shots_train", 1500)
    phase_net = PhaseMLP.from_dir(ckpt_dir)
    power_net = PowerMLP.from_dir(ckpt_dir)
    ck_net = CkMLP.from_dir(ckpt_dir)
    return actor, phase_net, power_net, ck_net


# ── compute RL action via the EXACT same path as train.py main loop ───────────
def _rl_forward(env, actor, phase_net, power_net, ck_net, D_k):
    cfg = env.cfg
    K = cfg.K
    obs = env._get_obs()
    blocked = _compute_blocked(env)
    demand = np.full(K, D_k)
    s_t = actor.extract_state(obs, demand, blocked)

    # Assignment (greedy for evaluation)
    phi, _, actor_info = actor.forward(s_t, greedy=True)
    z_t = actor_info["z_t"]

    # Phase
    active_irs = _get_active_irs(phi)
    active_irs_ids = _get_active_irs_ids(phi)
    s_phase = _build_phase_state(env.channels, phi, cfg, z_t)
    phase_idx, _, _ = phase_net.forward(s_phase, active_irs)
    phases_rad = env.phase_model.index_to_phase(phase_idx)
    proposed_Phi = env.phase_model.build_phi(phases_rad)

    # Power
    h_eff = env.rate_computer.effective_channels_all(phi, proposed_Phi, env.channels)
    s_power = np.concatenate([h_eff.real, h_eff.imag, z_t])
    w_c_vec, w_p, _, _ = power_net.forward(s_power, active_irs_ids)

    # Ck (need partial first)
    partial = env.rate_computer.compute_rates_partial(
        phi, proposed_Phi, env.channels, w_p, w_c_vec,
        active_irs_ids=active_irs_ids,
    )
    s_ck = _build_ck_state(demand, partial["R_private"], partial["R_c_group"], phi, cfg)
    C_k, _, _ = ck_net.forward(s_ck, phi, partial["R_c_group"])

    return dict(
        assignment=phi, phase_idx=phase_idx,
        w_p=w_p, w_c_vec=w_c_vec, C_k=C_k,
        active_irs_ids=active_irs_ids, proposed_Phi=proposed_Phi,
        z_t=z_t,
    )


# ── evaluate a policy's action on the CURRENT env channel state ───────────────
def _eval_action_on_state(env, action_dict, D_k):
    cfg = env.cfg
    K = cfg.K
    phi = np.asarray(action_dict["assignment"]).astype(int)
    Phi = action_dict.get("proposed_Phi", env.Phi)
    w_p = np.asarray(action_dict["w_p"]).flatten()
    w_c_vec = np.asarray(action_dict["w_c_vec"]).flatten()
    active_irs_ids = action_dict.get(
        "active_irs_ids",
        sorted({int(a) for a in phi if a > 0}),
    )
    C_k = action_dict.get("C_k")

    info = env.rate_computer.compute_sum_rate(
        assignment=phi,
        Phi=Phi,
        channels=env.channels,
        w_p=w_p,
        w_c_vec=w_c_vec,
        active_irs_ids=active_irs_ids,
        C_k=C_k,
        sigma2=env.cfg.sigma2,
    )
    R_user = info["R_private"] + info["C_k"]
    return {
        "R_user": R_user,
        "R_tot": float(R_user.sum()),
        "qos_frac": float((R_user >= D_k).mean()),
        "qos_pass": (R_user >= D_k).astype(int),
        "per_user_power": _per_user_power(phi, w_p, w_c_vec, cfg),
        "irs_chosen": (phi > 0).astype(int),
    }


def _per_user_power(phi, w_p, w_c_vec, cfg):
    """Per-user power proxy = w_p[k] + share of w_c_vec[group(k)]."""
    K = cfg.K
    w_p = np.asarray(w_p).flatten()[:K]
    w_c_vec = np.asarray(w_c_vec).flatten()
    phi = np.asarray(phi).astype(int)
    per_user = w_p.astype(float).copy()
    for k in range(K):
        g = int(phi[k])
        if g < len(w_c_vec):
            n_in_group = max(1, int(np.sum(phi == g)))
            per_user[k] += float(w_c_vec[g]) / n_in_group
    return per_user


# ── baseline action constructors ──────────────────────────────────────────────
def _direct_action(env, cfg):
    K = cfg.K
    P_S_W = 10 ** ((cfg.P_S_dBm - 30) / 10)
    return dict(
        assignment=np.zeros(K, dtype=int),
        w_p=np.full(K, P_S_W / K),
        w_c_vec=np.array([0.0]),
        active_irs_ids=[],
        proposed_Phi=env.Phi,
    )


def _all_irs_action(env, cfg):
    K, M = cfg.K, cfg.M
    P_S_W = 10 ** ((cfg.P_S_dBm - 30) / 10)
    # Default IRS phases (uniform — same as env.Phi)
    return dict(
        assignment=np.ones(K, dtype=int),
        w_p=np.full(K, P_S_W / K),
        w_c_vec=np.array([0.0, 0.0]),
        active_irs_ids=[1],
        proposed_Phi=env.Phi,
    )


def _equal_power_mix_action(env, cfg, rl_action):
    K = cfg.K
    P_S_W = 10 ** ((cfg.P_S_dBm - 30) / 10)
    phi = np.asarray(rl_action["assignment"]).astype(int)
    # n_groups: direct (0) + each active IRS
    active_groups = sorted(set(int(g) for g in phi))
    n_groups_vec = max(active_groups) + 1
    return dict(
        assignment=phi,
        w_p=np.full(K, P_S_W / K),
        w_c_vec=np.zeros(n_groups_vec + 1),
        active_irs_ids=rl_action["active_irs_ids"],
        proposed_Phi=rl_action["proposed_Phi"],
    )


# ── main probe driver ─────────────────────────────────────────────────────────
def run_probe(ckpt_dir: str, n_ep: int, n_steps: int, seed: int):
    cfg = make_config(D_k_bps_hz=0.15)
    D_k = cfg.D_k_bps_hz
    R_LoS = cfg.R_LoS_km
    env = ISTNEnv(cfg=cfg, seed=seed, n_steps_ep=n_steps, reward_noise_avg=1)
    actor, phase_net, power_net, ck_net = _load_nets(ckpt_dir, cfg)

    pol_tags = ["RL", "Direct-only", "All-IRS", "Equal-power-mix"]
    agg = {t: dict(R_tot=[], qos_frac=[]) for t in pol_tags}
    rec_blocked, rec_irs_chosen, rec_power, rec_R, rec_dist = [], [], [], [], []

    K = cfg.K
    for ep in range(n_ep):
        env.reset(seed=seed * 19 + ep)
        for _ in range(n_steps):
            blocked = env.channels["su_blocked"].astype(int)
            dist = np.linalg.norm(env.user_pos, axis=1)  # (K,) km from origin

            # RL action (full pipeline)
            rl = _rl_forward(env, actor, phase_net, power_net, ck_net, D_k)
            ev_rl = _eval_action_on_state(env, rl, D_k)
            agg["RL"]["R_tot"].append(ev_rl["R_tot"])
            agg["RL"]["qos_frac"].append(ev_rl["qos_frac"])

            # Direct-only
            ev_d = _eval_action_on_state(env, _direct_action(env, cfg), D_k)
            agg["Direct-only"]["R_tot"].append(ev_d["R_tot"])
            agg["Direct-only"]["qos_frac"].append(ev_d["qos_frac"])

            # All-IRS
            ev_i = _eval_action_on_state(env, _all_irs_action(env, cfg), D_k)
            agg["All-IRS"]["R_tot"].append(ev_i["R_tot"])
            agg["All-IRS"]["qos_frac"].append(ev_i["qos_frac"])

            # Equal-power-mix (RL routing, equal power)
            ev_m = _eval_action_on_state(env, _equal_power_mix_action(env, cfg, rl), D_k)
            agg["Equal-power-mix"]["R_tot"].append(ev_m["R_tot"])
            agg["Equal-power-mix"]["qos_frac"].append(ev_m["qos_frac"])

            # per-user records
            rec_blocked.append(blocked)
            rec_irs_chosen.append(ev_rl["irs_chosen"])
            rec_power.append(ev_rl["per_user_power"])
            rec_R.append(ev_rl["R_user"])
            rec_dist.append(dist)

            # advance env using RL action
            env.step({
                "assignment": rl["assignment"], "phase_idx": rl["phase_idx"],
                "w_p": rl["w_p"], "w_c_vec": rl["w_c_vec"], "C_k": rl["C_k"],
            })

    # ── analysis
    n_states = len(rec_blocked)
    blocked_arr = np.array(rec_blocked).flatten()
    irs_arr = np.array(rec_irs_chosen).flatten()
    power_arr = np.array(rec_power).flatten()
    R_arr = np.array(rec_R).flatten()
    dist_arr = np.array(rec_dist).flatten()

    print()
    print("=" * 76)
    print(f"  D_k HYPOTHESIS PROBE  ·  ckpt={ckpt_dir}")
    print(f"  D_k={D_k}  R_LoS={R_LoS}km  N_states={n_states}  N_records={len(blocked_arr)}")
    print("=" * 76)

    # Cross-policy aggregate
    print()
    print("  POLICY-LEVEL AGGREGATE (mean over states)")
    print(f"    {'policy':<22s}  {'QoS%':>7s}  {'R_tot':>8s}")
    for t in pol_tags:
        q = 100 * np.mean(agg[t]["qos_frac"])
        r = np.mean(agg[t]["R_tot"])
        print(f"    {t:<22s}  {q:7.1f}   {r:8.3f}")

    # Behavioral correlations
    print()
    print("  PER-USER BEHAVIORAL CORRELATIONS (RL policy)")
    corr_routing = float(np.corrcoef(irs_arr, blocked_arr)[0, 1]) if irs_arr.std() > 0 else 0.0
    corr_power_block = float(np.corrcoef(power_arr, blocked_arr)[0, 1]) if power_arr.std() > 0 else 0.0
    corr_power_dist = float(np.corrcoef(power_arr, dist_arr)[0, 1]) if power_arr.std() > 0 else 0.0
    def _tag(v, thr_strong, thr_partial):
        if v >= thr_strong: return "✅ STRONG"
        if v >= thr_partial: return "🟡 PARTIAL"
        return "❌ no/weak"
    print(f"    corr(chose_IRS, blocked)        = {corr_routing:+.3f}   {_tag(corr_routing, 0.5, 0.3)}")
    print(f"    corr(power_allocated, blocked)  = {corr_power_block:+.3f}   {_tag(corr_power_block, 0.3, 0.15)}")
    print(f"    corr(power_allocated, distance) = {corr_power_dist:+.3f}   (edge=far → ≥ 0.2 boosts edge users)")

    # Per user-type breakdown
    print()
    print("  PER-USER-TYPE BREAKDOWN (RL policy)")
    print(f"    {'group':<18s}  {'count':>6s}  {'P(IRS)':>8s}  {'mean_pwr':>10s}  {'mean_R':>8s}  {'QoS%':>6s}")
    blocked_mask = blocked_arr.astype(bool)
    for tag, m in [("unblocked", ~blocked_mask), ("blocked", blocked_mask)]:
        n = int(m.sum())
        if n == 0:
            continue
        p_irs = float(irs_arr[m].mean())
        mp = float(power_arr[m].mean())
        mr = float(R_arr[m].mean())
        qos = 100 * float((R_arr[m] >= D_k).mean())
        print(f"    {tag:<18s}  {n:>6d}  {p_irs:>8.3f}  {mp:>10.3e}  {mr:>8.3f}  {qos:>6.1f}")

    # Decision matrix verdict
    print()
    print("=" * 76)
    print("  DECISION MATRIX VERDICT")
    print("=" * 76)
    rl_qos = 100 * np.mean(agg["RL"]["qos_frac"])
    direct_qos = 100 * np.mean(agg["Direct-only"]["qos_frac"])
    epm_qos = 100 * np.mean(agg["Equal-power-mix"]["qos_frac"])
    smart_route = corr_routing >= 0.5
    smart_route_partial = corr_routing >= 0.3
    smart_pwr = corr_power_block >= 0.3
    smart_pwr_partial = corr_power_block >= 0.15

    if rl_qos < 80:
        v = ("❌ POLICY FAIL", f"RL QoS={rl_qos:.0f}% < 80% — fails D_k=0.15 regime. Hypothesis SỤP.")
    elif not (smart_route or smart_route_partial) and not (smart_pwr or smart_pwr_partial):
        v = ("❌ NO LEARNING", f"RL QoS={rl_qos:.0f}% but no smart routing+power → lucky generalization.")
    elif smart_route and smart_pwr:
        v = ("✅ STRONG", f"RL QoS={rl_qos:.0f}% WITH smart routing AND power → RSMA+IRS truly needed. PIVOT D_k=0.15 DEFENSIBLE.")
    elif smart_route or smart_pwr:
        v = ("🟡 PARTIAL", f"RL QoS={rl_qos:.0f}% with SOME smart behavior. Could push D_k higher or frame as Pareto trade.")
    elif smart_route_partial or smart_pwr_partial:
        v = ("🟡 WEAK PARTIAL", f"RL QoS={rl_qos:.0f}% with weak smart behavior. Marginal story.")
    else:
        v = ("⚠ UNCLEAR", f"RL QoS={rl_qos:.0f}% — borderline. Consider longer training.")
    print(f"  {v[0]}")
    print(f"    {v[1]}")
    # Comparison check: RL vs Equal-power-mix
    print(f"    RL QoS {rl_qos:.1f} vs Equal-power-mix QoS {epm_qos:.1f}  "
          f"→ Δ={rl_qos - epm_qos:+.1f}pt  "
          f"({'power allocation CONTRIBUTES' if rl_qos - epm_qos >= 5 else 'power allocation ≈ neutral'})")
    print(f"    Direct-only baseline QoS {direct_qos:.1f}")
    print("=" * 76)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .../agents/ dir.")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_probe(args.ckpt, args.episodes, args.steps, args.seed)


if __name__ == "__main__":
    main()
