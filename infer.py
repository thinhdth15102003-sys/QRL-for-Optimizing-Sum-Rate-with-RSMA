"""
infer.py
--------
Inference / evaluation script for a trained HQC-HAC agent.

Loads a trained agent from results/result_N/agents/ and benchmarks it
against four classical baselines over a fixed number of episodes.

Usage
-----
    python infer.py results/result_5
    python infer.py results/result_5 --episodes 100
    python infer.py results/result_5 --seed 42
    python infer.py results/result_5 --stochastic   # sample-based, not greedy
"""

import os
import sys
import json
import argparse
import numpy as np

# Ensure the console accepts UTF-8 box-drawing characters on Windows.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import params as P
from params     import make_config
from CSI.env    import ISTNEnv
from CSI.baselines import RandomPolicy, GreedyPolicy, DirectOnlyPolicy, AllIRSPolicy
from RL         import QuantumActor, ClassicalActor, PhaseMLP, PowerMLP, CkMLP


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline helpers  (mirror of train.py — kept local to avoid circular imports)
# ══════════════════════════════════════════════════════════════════════════════

def _build_phase_state(channels: dict, phi: np.ndarray, cfg,
                       z_t: np.ndarray) -> np.ndarray:
    """Build (M, N, 4K + n_latent + K) PER-ELEMENT state for PhaseMLP.
    ⚠ EXACT MIRROR of train.py::_build_phase_state — any divergence silently
    changes the policy at eval time. Keep the two in sync.

    Row (m,n) = [Re(c^SRU_{m,n})(K), Im(c^SRU_{m,n})(K),
                 Re(ĝ_SU)(K), Im(ĝ_SU)(K), z_t(n_latent), phi_mask_m(K)]
    with c_{m,n,k} = conj(ĝ_SR[m])·ĝ_RU[m,n,k] — element n's OWN cascade channel.
    ĝ_SU is the direct-link reference the elements must align against
    (θ_n = arg g_SU − arg c_n); without it the oracle phase is not a function of
    the state — see the rationale in train.py."""
    M, N, K  = cfg.M, cfg.N, cfg.K
    n_latent = len(z_t)
    off_su, off_z, off_mask = 2 * K, 4 * K, 4 * K + n_latent
    s        = np.zeros((M, N, 5 * K + n_latent))
    g_su     = channels['g_SU_hat']
    EPS      = 1e-30
    for k in range(K):
        m = int(phi[k]) - 1
        if m >= 0:
            c_mnk = channels['g_SR_hat'][m].conj() * channels['g_RU_hat'][m, :, k]  # (N,)
            a_c = np.abs(c_mnk) + EPS                     # UNIT PHASOR — see train.py
            s[m, :, k]              += c_mnk.real / a_c
            s[m, :, K + k]          += c_mnk.imag / a_c
            a_g = abs(g_su[k]) + EPS
            s[m, :, off_su + k]     += g_su[k].real / a_g
            s[m, :, off_su + K + k] += g_su[k].imag / a_g
            s[m, :, off_mask + k]    = 1.0   # phi_mask: user k belongs to IRS m
    s[:, :, off_z : off_z + n_latent] = z_t[None, None, :]   # broadcast z_t
    return s


def _get_active_irs(phi: np.ndarray) -> np.ndarray:
    """Sorted 0-based IRS indices with ≥1 assigned user (for PhaseMLP)."""
    return np.array(
        sorted({int(phi[k]) - 1 for k in range(len(phi)) if phi[k] > 0}),
        dtype=int,
    )


def _get_active_irs_ids(phi: np.ndarray) -> list:
    """Sorted 1-based physical IRS gids with ≥1 assigned user (for PowerMLP/rate)."""
    return sorted(set(int(phi[k]) for k in range(len(phi)) if phi[k] > 0))


def _build_ck_state(demand: np.ndarray, R_private: np.ndarray,
                    R_c_group: dict, phi: np.ndarray, cfg) -> np.ndarray:
    """Build (5·K,) [D_k, R_p_k, R_c_g_k, shortfall_k, phi_float_k] for CkMLP (mirror of train.py)."""
    K     = cfg.K
    R_c_g = np.zeros(K)
    for k in range(K):
        R_c_g[k] = float(R_c_group.get(int(phi[k]), 0.0))
    shortfall = np.maximum(0.0, demand - R_private)
    phi_float = phi[:K].astype(float)
    return np.concatenate([demand, R_private, R_c_g, shortfall, phi_float])


def _compute_irs_favored(env: ISTNEnv) -> np.ndarray:
    """Per-user indicator: True when the best IRS path beats the direct link, on
    ESTIMATED CSI. NOTE: this is *IRS-favoured*, not the physical blockage flag
    (env.channels['su_blocked']); it also depends on the CURRENT env.Phi. It is
    currently passed to actor.extract_state() but IGNORED there -- the affinity
    features encode blockage implicitly (blocked link -> a_{k,m} ~ 0)."""
    ch       = env.channels
    h_direct = np.abs(ch['g_SU_hat'])
    h_irs    = np.array([
        ch['beta'][m] * np.abs(np.sum(
            ch['g_SR_hat'][m].conj() * np.diag(env.Phi[m])[:, np.newaxis]
            * ch['g_RU_hat'][m],
            axis=0))
        for m in range(env.cfg.M)
    ])
    return h_irs.max(axis=0) > h_direct


# ══════════════════════════════════════════════════════════════════════════════
# Agent loading
# ══════════════════════════════════════════════════════════════════════════════

def load_training_cfg(run_dir: str, kappa=None, noise_var_dBW=None,
                      r_los=None, irs_spawn_frac=None, user_free_frac=None,
                      n_elements=None, p_s_dbm=None, user_speed=None,
                      natural_spawn=False):
    """
    Build a SystemConfig that exactly matches the training-time topology.

    Reads agents/training_config.json (written by train.py) to recover K, M,
    N, quantization_bits, P_S_dBm, and D_k_bps_hz, AND the run's
    hyperparameters.json to recover the curriculum ENV geometry — R_LoS_km +
    irs_spawn_radius_frac + user_free_radius_frac — which training_config.json
    does NOT store.  Without this the eval env silently reverts to params.py's
    R_LoS/radius (ramp0.2 confined) → a ramp0.3/0.5 or full-unlock policy would
    be evaluated on the WRONG env (tell-tale: Greedy baseline identical across
    ramps).  Remaining fields fall back to params.py defaults.

    kappa / noise_var_dBW / r_los / irs_spawn_frac / user_free_frac: optional
    TEST-TIME overrides for zero-shot env sweeps — evaluate a FIXED trained
    policy under a different noise / ramp / spawn geometry WITHOUT retraining.
    """
    topo_path = os.path.join(run_dir, 'agents', 'training_config.json')
    if not os.path.isfile(topo_path):
        raise FileNotFoundError(
            f"training_config.json not found in {run_dir}/agents/. "
            "Re-train with the current train.py to generate it.")
    with open(topo_path) as f:
        t = json.load(f)
    ov = dict(
        K=t['K'],
        M=t['M'],
        N=t['N'],
        quantization_bits=t['quantization_bits'],
        P_S_dBm=t['P_S_dBm'],
        D_k_bps_hz=t['D_k_bps_hz'],
    )
    # Recover the training-time ramp + spawn geometry from hyperparameters.json
    # (walk up from run_dir; training_config.json does NOT store these).
    # Check run_dir ITSELF first, then walk up (for a checkpoints/ep_XXXXX subdir).
    # The previous version called dirname() *before* the check, so a run root was
    # skipped entirely and R_LoS silently fell back to params.py — a ramp-0.5 policy
    # was then evaluated at the params.py ramp (0.2). Order matters here.
    _d = os.path.abspath(run_dir)
    for _ in range(5):
        _hp = os.path.join(_d, 'hyperparameters.json')
        if os.path.isfile(_hp):
            try:
                _sys = json.load(open(_hp)).get('system', {})
                for _k in ('R_LoS_km', 'irs_spawn_radius_frac',
                           'user_free_radius_frac'):
                    if _k in _sys:
                        ov[_k] = _sys[_k]
            except Exception:
                pass
            break
        _d = os.path.dirname(_d)
    # Test-time overrides (zero-shot env sweep) — applied AFTER the training env.
    if kappa is not None:          ov['kappa']          = float(kappa)
    if noise_var_dBW is not None:  ov['noise_var_dBW']  = float(noise_var_dBW)
    if r_los is not None:          ov['R_LoS_km']              = float(r_los)
    if irs_spawn_frac is not None: ov['irs_spawn_radius_frac'] = float(irs_spawn_frac)
    if user_free_frac is not None: ov['user_free_radius_frac'] = float(user_free_frac)
    if n_elements is not None:     ov['N']                     = int(n_elements)
    if p_s_dbm is not None:        ov['P_S_dBm']               = float(p_s_dbm)
    if user_speed is not None:     ov['user_speed_mps']        = float(user_speed)
    if natural_spawn:              ov['balanced_blocked_spawn'] = False
    return make_config(**ov)


def load_agents(run_dir: str, seed: int = None):
    """
    Load all four trained actor networks from run_dir/agents/.

    Each network class reconstructs its own architecture from the saved
    *_config.json — no cfg argument needed.  QuantumActor.from_dir() uses
    saved B (=M) and K so it is immune to params.py changes.
    """
    agents_dir = os.path.join(run_dir, 'agents')
    if not os.path.isdir(agents_dir):
        raise FileNotFoundError(
            f"No agents/ directory found in {run_dir}. "
            f"Run training first with train.py.")
    # Flat-PPO baseline (train_flat.py) — ONE network holds the whole policy, so
    # there are no phase/power/Ck nets to load. Signalled by flat_config.json;
    # the three Nones tell run_hqchac_episode to take the flat path.
    if os.path.isfile(os.path.join(agents_dir, 'flat_config.json')):
        from RL.flat_actor import FlatActor
        return FlatActor.from_dir(agents_dir, seed=seed), None, None, None
    # Detect actor type from saved config: the classical baseline
    # (--actor-mode classical) saves {"mode": "classical"} and must load via
    # ClassicalActor — QuantumActor.from_dir would KeyError on 'n_qubits'.
    _ac_path = os.path.join(agents_dir, 'actor_config.json')
    _amode = 'quantum'
    if os.path.isfile(_ac_path):
        try:
            _amode = json.load(open(_ac_path)).get('mode', 'quantum')
        except Exception:
            _amode = 'quantum'
    if _amode == 'classical':
        actor = ClassicalActor.from_dir(agents_dir, seed=seed)
    else:
        actor = QuantumActor.from_dir(agents_dir, seed=seed)
    phase_net = PhaseMLP.from_dir(agents_dir, seed=seed)
    power_net = PowerMLP.from_dir(agents_dir, seed=seed)
    ck_net    = CkMLP.from_dir(agents_dir, seed=seed)
    # The ③ power-fairness knobs are part of the trained policy's action mapping
    # (they blend the APPLIED power) but are NOT saved in power_config.json.
    # Restore all three from the run's hyperparameters.json (walk up from agents/);
    # missing any of them silently changes the executed policy at eval time:
    #   power_fairness       → split + common axes
    #   power_fairness_priv  → private axis (defaults to power_fairness)
    #   power_priv_frac      → target private split f (defaults to 0.8)
    _af, _ap, _pf, _d = 0.0, None, 0.8, os.path.abspath(agents_dir)
    for _ in range(5):
        _d = os.path.dirname(_d)
        _hp = os.path.join(_d, 'hyperparameters.json')
        if os.path.isfile(_hp):
            try:
                _pn = json.load(open(_hp)).get('power_net', {})
                power_net.scale_feats = bool(_pn.get('power_scale_feats', False))
                # The tanh divisor is part of the input definition, not a
                # cosmetic knob: evaluating an s=4 policy with s=1 feeds it a
                # feature it never saw. Runs predating the flag have no field,
                # and 1.0 is exactly what they trained with.
                power_net.feat_scale = float(_pn.get('power_feat_scale', 1.0))
                power_net.mag_feats = _pn.get('power_mag_feats', None)
                _af = float(_pn.get('power_fairness', 0.0))
                _ap = _pn.get('power_fairness_priv', None)
                _ap = _af if _ap is None else float(_ap)
                _pf = float(_pn.get('power_priv_frac', 0.8))
            except Exception:
                _af, _ap, _pf = 0.0, 0.0, 0.8
            break
    power_net.power_fairness      = _af
    power_net.power_fairness_priv = _af if _ap is None else _ap
    power_net.power_priv_frac     = _pf
    if _af > 0 or (_ap or 0) > 0:
        print(f"  ⚙ power_fairness α_split/common={_af:.2f} α_private="
              f"{power_net.power_fairness_priv:.2f} f={_pf:.2f} "
              f"restored from hyperparameters.json")
    return actor, phase_net, power_net, ck_net


# ══════════════════════════════════════════════════════════════════════════════
# Episode runners
# ══════════════════════════════════════════════════════════════════════════════

def run_hqchac_episode(env: ISTNEnv,
                       actor, phase_net, power_net, ck_net,
                       cfg, demand: np.ndarray,
                       n_steps: int,
                       greedy: bool = True) -> dict:
    """Run one episode with the HQC-HAC agent. Returns per-episode metrics."""
    # PhaseMLP's weights are N-agnostic (shared across reflecting elements,
    # d_out = n_levels), but the loaded instance keeps the N it trained at and its
    # forward loops `range(self.N)`. Evaluating on an array of a different size then
    # either index-errors (N_eval < N_train) or, worse, SILENTLY leaves the extra
    # elements at their previous phase (N_eval > N_train) — plausible-looking numbers
    # for a policy that only steered part of the surface. Retarget at the eval array;
    # no weight is touched, so this is a no-op whenever N matches.
    if phase_net is not None and getattr(phase_net, 'N', None) != cfg.N:
        phase_net.N = cfg.N
    obs = env.reset()

    ep_reward   = 0.0
    ep_sum_rate = []
    ep_feasible = []
    ep_qos_ok   = []
    ep_irs_frac = []

    is_flat = phase_net is None                    # flat-PPO baseline: one net
    flat_prev = None                               # (r_k, r_0) rate feedback

    for _ in range(n_steps):
        blocked = _compute_irs_favored(env)
        if is_flat:
            # The whole action comes from ONE forward on the raw state; C_k is
            # drawn from the SAME logits once the group common rates are known.
            # State must be built EXACTLY as train_flat.rollout builds it —
            # combined channel + rate feedback included, and update_norm=False so
            # evaluation cannot drift the normalisation trained under.
            h_now = env.rate_computer.effective_channels_all(
                env.assignment, env.Phi, env.channels)
            s_t = actor.extract_state(obs, demand, blocked, h_eff=h_now,
                                      prev_rates=flat_prev, update_norm=False)
            act, _, fi = actor.forward(s_t, greedy=greedy)
            phi, phase_idx = act['phi'], act['phase_idx']
            w_p, w_c_vec = fi['w_p'], fi['w_c_vec']
            proposed_Phi = env.phase_model.build_phi(
                env.phase_model.index_to_phase(phase_idx))
            partial = env.rate_computer.compute_rates_partial(
                phi, proposed_Phi, env.channels, w_p, w_c_vec,
                active_irs_ids=act['active_ids'])
            C_k, _, _, _ = actor.sample_ck(fi['logits'], phi,
                                           partial['R_c_group'], greedy=greedy)
            obs, reward, _, info = env.step({'assignment': phi,
                                             'phase_idx':  phase_idx,
                                             'w_p':        w_p,
                                             'w_c_vec':    w_c_vec,
                                             'C_k':        C_k})
            flat_prev = (np.asarray(info['R_private']), float(np.mean(C_k)))
            ep_reward += reward
            ep_sum_rate.append(float(info['sum_rate']))
            ep_feasible.append(bool(info['feasible']))
            ep_qos_ok.append(int(np.sum(info['R_tot'] >= cfg.D_k_bps_hz)))
            ep_irs_frac.append(float(np.mean(phi > 0)))
            continue
        s_t     = actor.extract_state(obs, demand, blocked)
        phi, _, actor_info = actor.forward(s_t, greedy=greedy)
        z_t = actor_info['z_t']                       # arch-2 spatial latent (shared by phase/power)
        # [--no-ae] sub-actors consume the quantum readout o_hat instead of the AE
        # latent z_t (mirror of train.py); classical/AE actors keep z_t.
        rep = actor_info.get('o_hat', z_t) if getattr(actor, 'NO_AE', False) else z_t

        active_irs     = _get_active_irs(phi)
        active_irs_ids = _get_active_irs_ids(phi)

        s_phase   = _build_phase_state(env.channels, phi, cfg, rep)
        phase_idx, _, _ = phase_net.forward(s_phase, active_irs, greedy=greedy)
        phases_rad   = env.phase_model.index_to_phase(phase_idx)
        proposed_Phi = env.phase_model.build_phi(phases_rad)

        h_eff   = env.rate_computer.effective_channels_all(phi, proposed_Phi, env.channels)
        # channel block rescaled to survive LayerNorm — see train._build_power_state.
        # [--power-clean-input] auto-detect from the loaded PowerMLP's expected input
        # dim: a clean PowerMLP reads ONLY h_eff (2K); a normal one also reads the rep.
        _hs = float(np.mean(np.abs(h_eff))) + 1e-9
        s_power = np.concatenate([h_eff.real / _hs, h_eff.imag / _hs])
        # [--power-scale-feats] absolute-scale block, appended BEFORE the rep so
        # the layout matches train._build_power_state exactly. Driven by the flag
        # recorded in hyperparameters.json, not inferred from d_s — with the rep
        # also optional, d_s alone cannot tell the two blocks apart.
        if getattr(power_net, 'scale_feats', False):
            from train import _power_scale_feats
            cfg.power_feat_scale = getattr(power_net, 'feat_scale', 1.0)
            s_power = np.concatenate([s_power, _power_scale_feats(h_eff, cfg)])
        # [--power-mag-feats] appended AFTER the scale block and BEFORE the rep,
        # matching train._build_power_state's order exactly — the head indexes by
        # position, so a swapped order feeds it a permuted state it never saw.
        if getattr(power_net, 'mag_feats', None):
            from train import _power_mag_feats
            s_power = np.concatenate(
                [s_power, _power_mag_feats(h_eff, power_net.mag_feats)])
        if power_net.d_s > s_power.shape[0]:
            s_power = np.concatenate([s_power, rep])
        w_c_vec, w_p, _, _ = power_net.forward(s_power, active_irs_ids)

        partial = env.rate_computer.compute_rates_partial(
            phi, proposed_Phi, env.channels, w_p, w_c_vec,
            active_irs_ids=active_irs_ids)
        s_ck = _build_ck_state(demand, partial['R_private'], partial['R_c_group'], phi, cfg)
        C_k, _, _ = ck_net.forward(s_ck, phi, partial['R_c_group'])

        action = {
            'assignment': phi,
            'phase_idx':  phase_idx,
            'w_p':        w_p,
            'w_c_vec':    w_c_vec,
            'C_k':        C_k,
        }
        obs, reward, _, info = env.step(action)

        ep_reward   += reward
        ep_sum_rate.append(float(info['sum_rate']))
        ep_feasible.append(bool(info['feasible']))
        ep_qos_ok.append(int(np.sum(info['R_tot'] >= cfg.D_k_bps_hz)))
        ep_irs_frac.append(float(np.mean(phi > 0)))

    return {
        'reward':      ep_reward,
        'sum_rate':    float(np.mean(ep_sum_rate)),
        'feasibility': float(np.mean(ep_feasible)),
        'qos_frac':    float(np.mean(ep_qos_ok)) / cfg.K,
        'irs_frac':    float(np.mean(ep_irs_frac)),
    }


def run_baseline_episode(env: ISTNEnv, policy, policy_name: str,
                         cfg, n_steps: int) -> dict:
    """Run one episode with a classical baseline policy."""
    obs = env.reset()

    ep_reward   = 0.0
    ep_sum_rate = []
    ep_feasible = []
    ep_qos_ok   = []

    for _ in range(n_steps):
        if policy_name == 'greedy':
            action = policy.act(obs, env)
        else:
            action = policy.act(obs)

        obs, reward, _, info = env.step(action)
        ep_reward   += reward
        ep_sum_rate.append(float(info['sum_rate']))
        ep_feasible.append(bool(info['feasible']))
        ep_qos_ok.append(int(np.sum(info['R_tot'] >= cfg.D_k_bps_hz)))

    return {
        'reward':      ep_reward,
        'sum_rate':    float(np.mean(ep_sum_rate)),
        'feasibility': float(np.mean(ep_feasible)),
        'qos_frac':    float(np.mean(ep_qos_ok)) / cfg.K,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(run_dir: str, n_episodes: int, seed: int,
             greedy: bool, n_steps: int,
             kappa=None, noise_var_dBW=None, shots=None,
             r_los=None, irs_spawn_frac=None, user_free_frac=None,
             n_elements=None, p_s_dbm=None, user_speed=None,
             natural_spawn=False) -> dict:

    # Rebuild cfg from saved topology — guarantees K/M/N/bits AND the curriculum
    # ramp/radius geometry match the env the policy was TRAINED on.
    # kappa/noise_var/r_los/spawn/free/N/P_S/speed/spawn-mode = optional
    # test-time zero-shot overrides.
    cfg    = load_training_cfg(run_dir, kappa=kappa, noise_var_dBW=noise_var_dBW,
                               r_los=r_los, irs_spawn_frac=irs_spawn_frac,
                               user_free_frac=user_free_frac,
                               n_elements=n_elements, p_s_dbm=p_s_dbm,
                               user_speed=user_speed, natural_spawn=natural_spawn)
    print(f"  ⚙ eval env: R_LoS={cfg.R_LoS_km} · radius irs/user="
          f"{cfg.irs_spawn_radius_frac}/{cfg.user_free_radius_frac}")
    if kappa is not None or noise_var_dBW is not None:
        print(f"  ⚙ R4 noise override: kappa={cfg.kappa} noise_var_dBW={cfg.noise_var_dBW}")
    if (n_elements is not None or p_s_dbm is not None or user_speed is not None
            or natural_spawn):
        print(f"  ⚙ zero-shot env override: N={cfg.N} P_S={cfg.P_S_dBm}dBm "
              f"speed={cfg.user_speed_mps}m/s balanced_spawn={cfg.balanced_blocked_spawn}")
    env    = ISTNEnv(cfg, seed=seed, n_steps_ep=n_steps)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)

    actor, phase_net, power_net, ck_net = load_agents(run_dir, seed=seed)
    if shots is not None and hasattr(actor, 'n_shots'):
        print(f"  ⚙ deploy-shots override: n_shots {actor.n_shots} → {shots}")
        actor.n_shots = shots

    rng = np.random.default_rng(seed)
    policies = {
        'HQC-HAC':    None,
        'Greedy':     GreedyPolicy(cfg),
        'AllIRS':     AllIRSPolicy(cfg),
        'DirectOnly': DirectOnlyPolicy(cfg),
        'Random':     RandomPolicy(cfg, rng=rng),
    }

    results = {name: {'reward': [], 'sum_rate': [], 'feasibility': [], 'qos_frac': []}
               for name in policies}

    W = 100
    print(f"\n{'═'*W}")
    print(f"  HQC-HAC Evaluation  ·  {run_dir}")
    print(f"  {'─'*W}")
    print(f"  Episodes : {n_episodes}  |  Steps/ep : {n_steps}  |  "
          f"Mode : {'greedy' if greedy else 'stochastic'}  |  Seed : {seed}")
    print(f"{'═'*W}")
    print(f"  {'Episode':>8}  {'HQC-HAC Reward':>16}  {'HQC-HAC Rate':>14}", flush=True)
    print(f"  {'─'*8}  {'─'*16}  {'─'*14}")

    for ep in range(n_episodes):
        r = run_hqchac_episode(env, actor, phase_net, power_net, ck_net,
                               cfg, demand, n_steps, greedy=greedy)
        for k, v in r.items():
            if k in results['HQC-HAC']:
                results['HQC-HAC'][k].append(v)

        for name, policy in policies.items():
            if name == 'HQC-HAC':
                continue
            r_bl = run_baseline_episode(env, policy, name.lower(), cfg, n_steps)
            for k, v in r_bl.items():
                results[name][k].append(v)

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            hqc_r  = results['HQC-HAC']['reward']
            hqc_sr = results['HQC-HAC']['sum_rate']
            print(f"  {ep+1:>8}  {np.mean(hqc_r):>16.3f}  {np.mean(hqc_sr):>14.4f}",
                  flush=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  {'Policy':<14}  {'Reward':>10}  {'ΣRate(bps/Hz)':>14}  "
          f"{'Feasibility':>12}  {'QoS frac':>10}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*14}  {'─'*12}  {'─'*10}")

    order = ['HQC-HAC', 'Greedy', 'AllIRS', 'DirectOnly', 'Random']
    for name in order:
        r = results[name]
        rew  = float(np.mean(r['reward']))
        rate = float(np.mean(r['sum_rate']))
        feas = float(np.mean(r['feasibility'])) * 100
        qos  = float(np.mean(r['qos_frac']))  * 100
        print(f"  {name:<14}  {rew:>10.3f}  {rate:>14.4f}  {feas:>11.1f}%  {qos:>9.1f}%")

    print(f"{'═'*W}\n")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate a trained HQC-HAC agent vs baselines.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('run_dir', type=str,
                        help='Path to results/result_N/ directory')
    parser.add_argument('--episodes', type=int, default=P.n_eval_episodes,
                        help='Number of evaluation episodes')
    parser.add_argument('--steps',    type=int, default=P.n_steps_per_ep,
                        help='Environment steps per episode')
    parser.add_argument('--seed',     type=int, default=P.seed_eval,
                        help='Evaluation seed')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic (sampled) policy instead of greedy')
    parser.add_argument('--kappa', type=float, default=None,
                        help='R4 test-time override: CSI error coefficient κ (default = training '
                             'value 0.05). Sweep up (0.1/0.2/...) for noise-robustness eval (no retrain).')
    parser.add_argument('--noise-var', dest='noise_var', type=float, default=None,
                        help='R4 test-time override: receiver noise variance noise_var_dBW (default '
                             '= training 10.0). Sweep up for SNR-robustness eval (no retrain).')
    parser.add_argument('--shots', type=int, default=None,
                        help='Deploy-time override: VQC measurement shots (default = trained '
                             'n_shots, usu. 1500). Raise (6000/10000) to cut o_hat shot-noise → '
                             'more consistent assignment (no retrain). Ignored for classical actor.')
    parser.add_argument('--R-LoS', dest='r_los', type=float, default=None,
                        help='Test-time override: eval R_LoS_km ramp (default = training value from '
                             'hyperparameters.json). Set to eval a policy on a DIFFERENT ramp (zero-shot).')
    parser.add_argument('--irs-spawn-frac', dest='irs_spawn_frac', type=float, default=None,
                        help='Test-time override: irs_spawn_radius_frac (default = training value). '
                             '=1.0 → full-unlock eval geometry.')
    parser.add_argument('--user-free-frac', dest='user_free_frac', type=float, default=None,
                        help='Test-time override: user_free_radius_frac (default = training value). '
                             '=1.0 → full-unlock eval geometry.')
    parser.add_argument('--n-elements', dest='n_elements', type=int, default=None,
                        help='Zero-shot override: IRS elements per panel N (default = training '
                             'value). d_aff is N-independent so the policy transfers as-is.')
    parser.add_argument('--p-s-dbm', dest='p_s_dbm', type=float, default=None,
                        help='Zero-shot override: satellite power P_S in dBm (default = training value).')
    parser.add_argument('--user-speed', dest='user_speed', type=float, default=None,
                        help='Zero-shot override: user mobility speed in m/s (default = training 1.5).')
    parser.add_argument('--natural-spawn', dest='natural_spawn', action='store_true',
                        help='Zero-shot override: disable balanced_blocked_spawn — blocked users '
                             'spawn naturally (binomial across buildings) instead of evenly split.')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    evaluate(
        run_dir    = args.run_dir,
        n_episodes = args.episodes,
        seed       = args.seed,
        greedy     = not args.stochastic,
        n_steps    = args.steps,
        kappa         = args.kappa,
        noise_var_dBW = args.noise_var,
        shots         = args.shots,
        r_los          = args.r_los,
        irs_spawn_frac = args.irs_spawn_frac,
        user_free_frac = args.user_free_frac,
        n_elements     = args.n_elements,
        p_s_dbm        = args.p_s_dbm,
        user_speed     = args.user_speed,
        natural_spawn  = args.natural_spawn,
    )
