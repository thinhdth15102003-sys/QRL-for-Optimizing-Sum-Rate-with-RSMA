"""Main-results table: every policy scored on the SAME states, with seed spread.

Produces the rows of the Case-1 / Case-2 comparison tables in the paper:
HQC-HAC (VQC), DNN, PPO-flat, AO, Greedy, Random.

Two kinds of "seed" are reported and they are not interchangeable:

  · a LEARNED policy (VQC / DNN / flat) has one checkpoint per TRAINING seed;
    the spread across those checkpoints is what the paper's ± must report,
    because re-running inference on one checkpoint only re-samples the channel,
    not the optimisation. Pass one checkpoint per training seed.
  · a SOLVER (AO / Greedy / Random) has no training seed, so its ± comes from
    the environment seeds alone.

Every policy sees the identical pool of states (same env seeds, same walks), so
row-to-row differences are the policy, not the draw.

Scoring convention — the same one env.step uses, and the one the whole project
compares on:
    decide on estimated CSI  (ĝ)      — what every policy observes
    score  on true CSI       (g)      — averaged over reward_noise_avg σ² draws
    J = ΣR − λ_D·Σ(shortfall/(D_k+ε))²
Comparing R_tot across rows at different QoS is meaningless; J is the objective
all of them optimise.

Reported per row: J, R_tot, QoS%, and mean shortfall as BOTH a percentage of
D_k and an absolute rate. The percentage matters because binary QoS hides how
close a miss was — a user at 0.9·D_k and one at 0.1·D_k both count as "failed".
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import params as P
from params import make_config
from CSI.env import ISTNEnv
from CSI.baselines import RandomPolicy, GreedyPolicy
from RL import QuantumActor, ClassicalActor, PhaseMLP, PowerMLP, CkMLP
from train import (_build_phase_state, _build_ck_state, _get_active_irs,
                   _get_active_irs_ids, _compute_irs_favored, _build_power_state)
from analysis.oracle_alloc import phi_from_idx
from analysis.probe_ao_rate import AORatePolicy


# ── metric accumulation ──────────────────────────────────────────────────────
class Acc:
    """Per-step metrics, aggregated the way the paper table reports them."""

    def __init__(self, Dk, eps):
        self.Dk, self.eps = Dk, eps
        self.J, self.R, self.qos, self.sh_abs, self.sh_pct = [], [], [], [], []

    def add(self, info):
        R_tot = np.asarray(info['R_tot'], dtype=float)
        short = np.maximum(0.0, self.Dk - R_tot)
        self.J.append(float(info['sum_rate'])
                      - float(info.get('qp_penalty', 0.0)))
        self.R.append(float(info['sum_rate']))
        self.qos.append(float(np.mean(R_tot >= self.Dk - 1e-12)))
        # Averaged over ALL users, not only the failing ones: a row where few
        # users miss but miss badly must not look the same as one where many
        # miss narrowly.
        self.sh_abs.append(float(np.mean(short)))
        self.sh_pct.append(float(np.mean(short / self.Dk)))

    def summary(self):
        return dict(J=np.mean(self.J), R=np.mean(self.R),
                    qos=np.mean(self.qos),
                    sh_abs=np.mean(self.sh_abs), sh_pct=np.mean(self.sh_pct))


def build_pool(cfg, env_seed, n_ep, n_steps):
    """A reproducible list of (env_state_setter) episodes: seeds only.
    Each policy re-creates the env from the same seed, so all see one pool."""
    return [(env_seed * 1000 + i) for i in range(n_ep)]


# ── learned-policy rollout (hierarchical: VQC / DNN) ─────────────────────────
def run_hier(run_dir, cfg, seeds, n_steps, greedy=True, alpha=None):
    d = os.path.join(run_dir, 'agents')
    ac = json.load(open(os.path.join(d, 'actor_config.json')))
    actor = (ClassicalActor if ac.get('mode') == 'classical'
             else QuantumActor).from_dir(d, seed=0)
    if ac.get('mode') != 'classical':
        actor.n_shots = P.n_shots_eval
    ph = PhaseMLP.from_dir(d, seed=0)
    # PhaseMLP's WEIGHTS are N-agnostic (shared per reflecting element, d_out=n_levels),
    # but the loaded instance carries the N it trained at and its forward loops
    # `range(self.N)`. Evaluating at a different array size therefore truncates
    # (N_eval > N_train: the extra elements silently keep their previous phase) or
    # index-errors (N_eval < N_train). Retarget it at the eval array — no weights change.
    ph.N = cfg.N
    pw = PowerMLP.from_dir(d, seed=0)
    ck = CkMLP.from_dir(d, seed=0)
    # power-fairness is part of the trained action mapping but lives in
    # hyperparameters.json, not power_config.json — omitting it silently changes
    # the executed policy at eval time.
    hp_dir = os.path.abspath(d)
    for _ in range(5):
        hp_dir = os.path.dirname(hp_dir)
        hp = os.path.join(hp_dir, 'hyperparameters.json')
        if os.path.isfile(hp):
            pn = json.load(open(hp)).get('power_net', {})
            pw.power_fairness = float(pn.get('power_fairness', 0.0))
            pw.power_fairness_priv = float(pn.get('power_fairness_priv',
                                                  pn.get('power_fairness', 0.0)))
            pw.power_priv_frac = float(pn.get('power_priv_frac', 0.8))
            break
    if alpha is not None:
        # Override the fairness prior at EVALUATION time, leaving the trained
        # weights untouched. The prior is applied in distribution space when the
        # action is drawn, not learned, so sweeping it here asks what the same
        # policy would do under a different prior — which is exactly the question
        # when the failure is suspected to be a fixed constant that does not
        # adapt to the power regime.
        pw.power_fairness = pw.power_fairness_priv = float(alpha)

    acc = Acc(cfg.D_k_bps_hz, cfg.epsilon_qp)
    dem = np.full(cfg.K, cfg.D_k_bps_hz)
    for s in seeds:
        env = ISTNEnv(cfg=cfg, seed=s, n_steps_ep=n_steps,
                      reward_noise_avg=getattr(P, 'reward_noise_avg', 1))
        obs = env.reset(seed=s)
        for _ in range(n_steps):
            s_t = actor.extract_state(obs, dem, _compute_irs_favored(env))
            phi, _, inf = actor.forward(s_t, greedy=greedy)
            rep = inf['z_t']
            act_irs, act_ids = _get_active_irs(phi), _get_active_irs_ids(phi)
            pidx, _, _ = ph.forward(
                _build_phase_state(env.channels, phi, cfg, rep), act_irs,
                greedy=greedy)
            Phi = env.phase_model.build_phi(env.phase_model.index_to_phase(pidx))
            h = env.rate_computer.effective_channels_all(phi, Phi, env.channels)
            wc, wp, _, _ = pw.forward(_build_power_state(h, rep), act_ids)
            part = env.rate_computer.compute_rates_partial(
                phi, Phi, env.channels, wp, wc, active_irs_ids=act_ids)
            C_k, _, _ = ck.forward(
                _build_ck_state(dem, part['R_private'], part['R_c_group'], phi, cfg),
                phi, part['R_c_group'])
            obs, _, _, info = env.step({'assignment': phi, 'phase_idx': pidx,
                                        'w_p': wp, 'w_c_vec': wc, 'C_k': C_k})
            acc.add(info)
    return acc.summary()


# ── flat-PPO rollout ─────────────────────────────────────────────────────────
def run_flat(run_dir, cfg, seeds, n_steps, greedy=True):
    from RL.flat_actor import FlatActor
    fa = FlatActor.from_dir(os.path.join(run_dir, 'agents'), seed=0)
    acc = Acc(cfg.D_k_bps_hz, cfg.epsilon_qp)
    dem = np.full(cfg.K, cfg.D_k_bps_hz)
    for s in seeds:
        env = ISTNEnv(cfg=cfg, seed=s, n_steps_ep=n_steps,
                      reward_noise_avg=getattr(P, 'reward_noise_avg', 1))
        obs = env.reset(seed=s)
        prev = None
        for _ in range(n_steps):
            h_now = env.rate_computer.effective_channels_all(
                env.assignment, env.Phi, env.channels)
            st = fa.extract_state(obs, dem, _compute_irs_favored(env),
                                  h_eff=h_now, prev_rates=prev, update_norm=False)
            act, _, fi = fa.forward(st, greedy=greedy)
            Phi = env.phase_model.build_phi(
                env.phase_model.index_to_phase(act['phase_idx']))
            part = env.rate_computer.compute_rates_partial(
                act['phi'], Phi, env.channels, fi['w_p'], fi['w_c_vec'],
                active_irs_ids=act['active_ids'])
            C_k, _, _, _ = fa.sample_ck(fi['logits'], act['phi'],
                                        part['R_c_group'], greedy=greedy)
            obs, _, _, info = env.step({'assignment': act['phi'],
                                        'phase_idx': act['phase_idx'],
                                        'w_p': fi['w_p'], 'w_c_vec': fi['w_c_vec'],
                                        'C_k': C_k})
            prev = (np.asarray(info['R_private']), float(np.mean(C_k)))
            acc.add(info)
    return acc.summary()


# ── solver / blind baselines ─────────────────────────────────────────────────
def run_solver(kind, cfg, seeds, n_steps):
    acc = Acc(cfg.D_k_bps_hz, cfg.epsilon_qp)
    for s in seeds:
        env = ISTNEnv(cfg=cfg, seed=s, n_steps_ep=n_steps,
                      reward_noise_avg=getattr(P, 'reward_noise_avg', 1))
        obs = env.reset(seed=s)
        if kind == 'ao':
            pol = AORatePolicy(cfg, env.rate_computer, rounds=2, n_split=11,
                               route_ascent=True, phase_sweeps=1)
        elif kind == 'greedy':
            pol = GreedyPolicy(cfg)
        else:
            pol = RandomPolicy(cfg, rng=np.random.default_rng(s))
        for _ in range(n_steps):
            a = pol.act(obs, env) if kind in ('ao', 'greedy') else pol.act(obs)
            obs, _, _, info = env.step(a)
            acc.add(info)
    return acc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--M', type=int, required=True)
    ap.add_argument('--hier', nargs='*', default=[],
                    help='label=run_dir (one per TRAINING seed), e.g. DNN=results/result_234/checkpoints/ep_20000')
    ap.add_argument('--flat', nargs='*', default=[])
    ap.add_argument('--env-seeds', default='42,43,44')
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--P', dest='ps', type=float, default=50.0)
    ap.add_argument('--R-LoS', dest='rlos', type=float, default=0.5)
    ap.add_argument('--skip-ao', action='store_true')
    # Zero-shot distribution shift: the POLICY is unchanged (loaded from a
    # checkpoint trained at the base operating point) and only the evaluation
    # environment moves. Passed as k=v pairs so one invocation covers one row of
    # the generalization table, e.g. --shift P_S_dBm=40 or
    # --shift P_S_dBm=40 kappa=0.20 user_speed_mps=6
    ap.add_argument('--shift', nargs='*', default=[],
                    help='cfg overrides applied at EVAL time only, as key=value')
    ap.add_argument('--alpha', type=float, default=None,
                    help='override power-fairness at eval time, both axes')
    a = ap.parse_args()

    over = dict(K=a.K, M=a.M, P_S_dBm=a.ps, R_LoS_km=a.rlos)
    for kv in a.shift:
        k, v = kv.split('=', 1)
        # Integer-valued knobs (N, quantization_bits, ...) must stay int: they end up
        # as array shapes, and numpy rejects a float size. Only fall back to float
        # when the literal actually has a decimal/exponent.
        if v.lstrip('-').isdigit():
            over[k] = int(v)
        elif any(c in v for c in '.eE'):
            over[k] = float(v)
        else:
            over[k] = v
    cfg = make_config(**over)
    seeds = [int(x) for x in a.env_seeds.split(',')]
    Dk = cfg.D_k_bps_hz

    print("=" * 96)
    print(f"  MAIN TABLE  K={cfg.K} M={cfg.M} P_S={a.ps}dBm R_LoS={a.rlos} "
          f"D_k={Dk} lam={cfg.lambda_D}")
    print(f"  env seeds {seeds} x {a.steps} steps = {len(seeds)*a.steps} states/row")
    print("=" * 96)
    # Reward is the per-EPISODE return the paper tables report; J is the same
    # quantity per step (env.step returns sum_rate - qp_penalty each step), so
    # Reward = J * steps exactly. Both are printed so the table can use either
    # without anyone having to multiply by hand.
    hdr = (f"  {'Policy':<16}{'Reward':>20}{'J (per step)':>18}"
           f"{'R_tot':>18}{'QoS':>16}{'mean shortfall':>22}")
    print(hdr)
    print("  " + "-" * 110)

    def emit(label, runs):
        """runs: list of summary dicts, one per TRAINING seed (or one for solvers)."""
        g = lambda k: np.array([r[k] for r in runs])
        n = len(runs)
        pm = lambda v: (f"{v.mean():.4f}" if n == 1
                        else f"{v.mean():.4f}±{v.std(ddof=1):.4f}")
        pmq = lambda v: (f"{v.mean()*100:.1f}%" if n == 1
                         else f"{v.mean()*100:.1f}±{v.std(ddof=1)*100:.1f}%")
        sh = g('sh_pct').mean() * 100
        sa = g('sh_abs').mean()
        rew = g('J') * a.steps
        print(f"  {label:<16}{pm(rew):>20}{pm(g('J')):>18}{pm(g('R')):>18}"
              f"{pmq(g('qos')):>16}{f'{sh:.2f}%/{sa:.5f}':>22}")

    for spec in a.hier:
        lbl, dirs = spec.split('=', 1)
        runs = [run_hier(d, cfg, seeds, a.steps, alpha=a.alpha)
                for d in dirs.split(',')]
        emit(f"{lbl} ({len(runs)}s)", runs)
    for spec in a.flat:
        lbl, dirs = spec.split('=', 1)
        runs = [run_flat(d, cfg, seeds, a.steps) for d in dirs.split(',')]
        emit(f"{lbl} ({len(runs)}s)", runs)
    if not a.skip_ao:
        emit('AO', [run_solver('ao', cfg, seeds, a.steps)])
    emit('Greedy', [run_solver('greedy', cfg, seeds, a.steps)])
    emit('Random', [run_solver('random', cfg, seeds, a.steps)])
    print("=" * 96)
    print(f"  J = SR - lam*sum((shortfall/(D_k+eps))^2); shortfall = max(0, D_k - R_k),")
    print(f"  averaged over ALL K users and all steps. '%' is of D_k={Dk}.")
    print(f"  +/- on learned rows = spread across TRAINING seeds; solver rows have none.")


if __name__ == '__main__':
    sys.exit(main())
