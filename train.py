"""
train.py
--------
Training script: Quantum Actor full pipeline for IRS-assisted RSMA.

Actor pipeline (4 stages per step)
------------------------------------
  1. Quantum IRS selection  : state s_t  →  φ ∈ {0,..,M}^K
  2. PhaseMLP               : per-IRS state (M, 2K)  →  phase_idx ∈ {0..L-1}^{G×N}  (G active IRS only)
  3. PowerMLP               : h_eff state (2K) [Re‖Im]  →  [w_c_vec (G+1), w_p (K)] × P_S
  4. CkMLP                  : [D_k, R_p_k, R_c_g_k] state (3·K) → C_k per user

All hyperparameters live in params.py — edit there, not here.
Results are saved to results/result_<N>/ after training completes.

Usage
-----
    python train.py                           # defaults from params.py
    python train.py --episodes 50 --steps 50  # quick test run
    python train.py --seed 42
    python train.py --no-plots                # skip matplotlib
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

import params as P
from params  import make_config
from CSI.env import ISTNEnv
from CSI.baselines import DirectOnlyPolicy, AllIRSPolicy
from analysis.phase_oracle import oracle_phase_idx   # --phase-aux-weight target
from RL        import QuantumActor, ClassicalActor, ClassicalCritic, PhaseMLP, PowerMLP, CkMLP
from RL.critic_diag import (compute_critic_diag, format_critic_diag_lines,
                            write_critic_diag_jsonl)
from RL.quantum_circuit import GPU_BACKEND


# ══════════════════════════════════════════════════════════════════════════════
# Logging helpers
# ══════════════════════════════════════════════════════════════════════════════

class _Tee:
    """Mirrors all writes to both the terminal and a log file."""
    def __init__(self, fh):
        self._fh     = fh
        self._stdout = sys.stdout
    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._fh.write(data)
    def log_only(self, data: str) -> None:
        """Write to the log file ONLY (not the terminal)."""
        self._fh.write(data)
    def flush(self) -> None:
        self._stdout.flush()
        self._fh.flush()
    def isatty(self) -> bool:
        return False


def _flog(msg: str = "") -> None:
    """Write a line ONLY to the training_log file (skips the terminal).
    Used for verbose diagnostics we want recorded but not cluttering the screen."""
    out = sys.stdout
    if hasattr(out, 'log_only'):
        out.log_only(msg + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline state helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_phase_state(channels: dict, phi: np.ndarray, cfg,
                       z_t: np.ndarray, d_s: int = None) -> np.ndarray:
    """
    Build per-IRS cascade channel states for PhaseMLP.

    c^SRU_{m,k} = conj(g_SR_hat[m]) · g_RU_hat[m,k]  if phi[k] == m+1, else 0.
    Uses ESTIMATED channels (g_hat) — same CSI the system uses for precoding.

    Appends z_t (system spatial latent) and a per-IRS binary user-mask to each row.
      phi_mask_m[k] = 1  iff user k is assigned to IRS m  — tells PhaseMLP which
      users are its "clients" without having to unmix the global z_t.

    Returns (M, N, 4K + n_latent + K) float — one row PER REFLECTING ELEMENT:
      row (m,n) = [Re(c^SRU_{m,n}) (K), Im(c^SRU_{m,n}) (K),
                   Re(g_SU) (K), Im(g_SU) (K),          ← added 2026-07-21
                   z_t (n_latent), phi_mask_m (K)]
      with c^SRU_{m,n,k} = conj(g_SR_hat[m]) · g_RU_hat[m,n,k] — element n's OWN
      cascade channel. PhaseMLP is applied per element with shared weights, so this
      is what makes phi_{m,n} a real decision (a per-IRS row cannot distinguish
      elements).
      Inactive IRS rows are all-zero in channel/g_SU/mask cols; z_t is broadcast.

    ⭐ WHY g_SU IS IN THE STATE (added 2026-07-21 — it was the phase bottleneck).
    The effective channel is h_k = g_SU[k] + Σ_n e^{jθ_n} c_{n,k}, so the oracle
    phase is θ_n = arg(g_SU[k]) − arg(c_{n,k}): it needs the DIRECT-LINK ANGLE to
    know what to align against. Without g_SU that angle is a per-state constant
    absent from the input, so the supervised target was NOT A FUNCTION OF THE
    STATE and the best possible fit was uniform. Measured on result_151
    (K10/M2): phase warm-up converged to CE 1.376 vs ln(4)=1.386 (0.010 nats
    below a uniform predictor) and per-element match 29.2% vs 25% chance; the
    live head then scored coherent-gain alignment 0.1% (q=0.204 vs random
    0.203) — i.e. indistinguishable from random phases, leaving the whole
    per-element phase lever (|h|² ×11 at oracle phase) untouched.

    ⭐ WHY UNIT PHASORS, NOT RAW Re/Im (2026-07-21 — this was THE phase bottleneck).
    PhaseMLP layer-norms each element row as a whole, and the row mixes blocks
    that differ by three orders of magnitude: c^SRU ~1e-3, g_SU ~5e-3, z_t ~4e-1,
    mask 1.0. The norm's mean/std are set by z_t and mask, so the tiny channel
    entries all collapse to nearly the same value and their ANGLE — the only
    thing the phase target depends on — is destroyed before the MLP sees it.
    Measured on one row: arg(c) = [2.633, -0.341, 0.402, 1.266] became
    [-2.356, -2.356, -2.331, -2.358] after the norm (max Δ 4.99 rad).

    Feeding c/|c| and g_SU/|g_SU| (i.e. cos∠, sin∠) keeps the angle exactly and
    is already scale-free, so the norm cannot corrupt it. Offline supervised fit
    on the oracle target, identical architecture/seed, only the encoding changed:
        raw Re/Im            CE 1.3317  match 35.2%
        unit phasor          CE 0.4571  match 84.2%      ← this
        unit phasor + log|·| CE 0.9552  match 61.3%   (log|·| ~ -7 re-breaks the norm)
    Magnitude is deliberately dropped: the target is angle-determined, and the
    third row is direct evidence that adding raw-scale magnitude back HURTS.
    This is a re-encoding of the SAME observed quantities — no new information,
    no oracle leakage; the head still has to learn to combine ∠g_SU with ∠c.

    LEGACY d_s: pass d_s=3K+n_latent to emit the PRE-g_SU layout, for probing
    checkpoints trained before 2026-07-21 (their PhaseMLP has the narrow input
    layer and would otherwise fail on a width mismatch). Probes should pass
    `phase_net.d_s`; training always uses the new layout.
    """
    M, N, K  = cfg.M, cfg.N, cfg.K
    n_latent = len(z_t)
    legacy   = (d_s is not None and int(d_s) == 3 * K + n_latent)
    # width MUST include the trailing phi_mask block (K) — legacy 2K+nl+K = 3K+nl,
    # new 4K+nl+K = 5K+nl. It must equal the PhaseMLP d_s exactly.
    if legacy:
        off_z, off_mask = 2 * K, 2 * K + n_latent
        width = 3 * K + n_latent
    else:
        off_z, off_mask = 4 * K, 4 * K + n_latent
        width = 5 * K + n_latent
    off_su   = 2 * K                       # Re/Im g_SU block (new layout only)
    s        = np.zeros((M, N, width))
    g_su     = channels['g_SU_hat']        # (K,) ESTIMATED — same CSI as c^SRU
    EPS      = 1e-30
    for k in range(K):
        m = int(phi[k]) - 1          # 0-based IRS index; -1 for direct users
        if m >= 0:
            c_mnk = channels['g_SR_hat'][m].conj() * channels['g_RU_hat'][m, :, k]  # (N,)
            if legacy:
                s[m, :, k]     += c_mnk.real
                s[m, :, K + k] += c_mnk.imag
            else:
                # UNIT PHASOR (cos∠, sin∠) rather than raw Re/Im — see the
                # normalisation note in the docstring.
                a_c = np.abs(c_mnk) + EPS
                s[m, :, k]     += c_mnk.real / a_c
                s[m, :, K + k] += c_mnk.imag / a_c
                # direct-link phasor of THIS user, broadcast across the elements —
                # constant in n, but it is the reference the elements align to.
                a_g = abs(g_su[k]) + EPS
                s[m, :, off_su + k]     += g_su[k].real / a_g
                s[m, :, off_su + K + k] += g_su[k].imag / a_g
            s[m, :, off_mask + k]      = 1.0   # phi_mask: user k belongs to IRS m
    # Broadcast z_t into every element row (same system spatial state everywhere)
    s[:, :, off_z : off_z + n_latent] = z_t[None, None, :]
    return s


def _beta_pwr_priv(args) -> float:
    """
    Extra entropy bonus on the PRIVATE power axis (on top of the global β).

    ⚠ params.py default is 0.003, i.e. 4× the global β=0.001 on that one axis.
    That was tuned when PowerMLP could not learn from reward at all (the
    phantom-action bug), so an entropy crutch was the only thing spreading w_p.
    With the Dirichlet policy the axis DOES learn, and the extra pressure now
    fights it: measured on result_150 (K5/M1, ep1200) the private max-share sat
    frozen at 29% against a 20% flat floor while split/common moved freely
    (43→81%). Pass --beta-entropy-pwr-private 0.0 to remove the crutch.
    """
    v = getattr(args, 'beta_entropy_pwr_private', None)
    return float(P.beta_entropy_pwr_private if v is None else v)


def _closed_form_ck(partial: dict, cfg) -> np.ndarray:
    """
    Closed-form common-rate split: fill the CHEAPEST unmet demand first inside
    each group, then spend the leftover budget evenly (leftover is QoS-neutral
    but pure sum-rate, so never leave it unspent).

    Used by --closed-form-ck. Unlike the phase oracle (which only maximises
    coherent gain, a HEURISTIC for J), this is provably optimal for the
    sub-problem it solves: given R_private and the group budget R_c_group,
    cheapest-demand-first maximises the number of users clearing D_k.

    Measured motivation (result_159 ep_00200, K10/M2): the learned CkMLP hit
    QoS 77.4% where this allocator hits 89.7% on the SAME routing/phase/power —
    it over-allocated 0.514/state above need against only 0.089 below, i.e. it
    spread C_k over users already met by their private rate instead of filling
    the ones short of D_k.
    """
    from analysis.oracle_alloc import oracle_ck_met
    Dk = cfg.D_k_bps_hz
    _, C_k = oracle_ck_met(partial['R_private'], partial['R_c_group'],
                           partial['groups'], Dk)
    for gid, members in partial['groups'].items():
        mem = np.asarray(list(members), dtype=int)
        if mem.size == 0:
            continue
        left = float(partial['R_c_group'].get(int(gid), 0.0)) - float(C_k[mem].sum())
        if left > 1e-12:
            C_k[mem] += left / mem.size
    return C_k


def _po_est(ch: dict) -> dict:
    """Env channels → the plain keys oracle_phase_idx expects, on ESTIMATED CSI."""
    return {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
            'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}


def _ck_aux_target(partial: dict, cfg) -> dict:
    """
    Per-group demand-fill SHARE vector (sums to 1) used as the teacher signal for
    --ck-aux-weight. Groups whose common budget is ~0 map to None and are skipped.
    """
    C_k = _closed_form_ck(partial, cfg)
    out = {}
    for gid, members in partial['groups'].items():
        mem = np.asarray(list(members), dtype=int)
        if mem.size == 0:
            out[gid] = None
            continue
        v = np.asarray(C_k[mem], dtype=float)
        s = float(v.sum())
        out[gid] = (v / s) if s > 1e-12 else None
    return out


# Grid for the power teacher, chosen by measuring what each candidate leaves on the
# table against the full 7×13 sweep the ceiling is computed on (2026-07-24):
#   (0,1,4,16)×(0.4,0.6,0.8)        Case 1 0.2%   Case 2 4.1%   ← kept, 12 pts
#   (0,1,4,8,16)×(0.35,.5,.6,.7)    Case 1 14.4%  Case 2 3.1%
#   (0,2,8,16)×(0.35,.55,.75,.95)   Case 1 8.1%   Case 2 6.6%
#   (0,1,2,4,8,16)×(0.4,0.6,0.8)    Case 1 0.2%   Case 2 3.8%   (+50% cost)
# f* differs sharply by case (Case 1 wants ~0.8, Case 2 ~0.6), so dropping either
# endpoint is expensive; extra β rungs buy almost nothing. The residual 4% is not a
# hard bound — the head still receives the full PPO gradient and can pass the target.
_PW_AUX_BETAS = (0.0, 1.0, 4.0, 16.0)     # private-power concentration exponent
_PW_AUX_FRACS = (0.4, 0.6, 0.8)           # fraction of P_S given to the private stream

# [--power-aux-beta-cap] 2026-08-01. Swept beta on 100 states of r232, CHOOSING on
# the estimated channel and SCORING on the true one, and the achieved J is:
#     beta   0.00    0.50    1.00    2.00    4.00   16.00
#     J    1.5312  1.5538  1.5765  1.4697  1.3752  1.3920
#     QoS   99.6%   99.3%   98.8%   96.3%   93.8%   93.3%
# J peaks at beta=1 (a 33x dynamic range) and collapses beyond it — heavy
# concentration is WORSE than equal power once the allocation is paid on the real
# channel. The grid above is selected per state by maximising on g-hat, where
# extreme concentration looks good precisely because it exploits estimation
# error, so it hands the head a target that loses on g. That is a concrete
# mechanism for the 21 power-aux runs that all came out net-harmful.
_PW_AUX_BETAS_CAPPED = (0.0, 0.25, 0.5, 1.0)


def _power_aux_target(rate_computer, phi, Phi, channels, active_irs_ids,
                      h_eff, cfg) -> dict:
    """
    Teacher allocation for --power-aux-weight: the (β*, f*) on a small grid that
    maximises the DESIGN-view objective for this state, expressed as the three
    simplex vectors PowerMLP samples.

        private ∝ |h_eff|^{2β}      β=0 equal power … β large winner-take-all
        split   = [1-f, f]          (common budget, private budget)
        common  = uniform over the active slots

    ⚠ 2026-07-24 KNOWN-LIMITED. A noise-aware variant (J averaged over σ²
    quantiles) was tried and REVERTED — it did not change the picked β (0.826 →
    0.816 max-private, still 88% WTA), because on the agent's imperfect routing
    the design objective genuinely rates high β well. Measured separately: clean
    β-power + closed-form C_k on the agent's routing gives achieved J 1.46 (β=0)
    → 1.60 (β=4) → 1.59 (β=16), i.e. β is a MINOR lever and even WTA is fine. The
    live power-aux collapse (r208 J 0.70) comes from the learned power × neural
    C_k interaction, NOT from over-concentration. Power-aux with neural C_k was
    net-harmful vs fairness-only; prefer --closed-form-ck when using this. Do not
    reintroduce noise-averaging here without re-measuring — it costs 4× for ~0.

    12 grid points, each one partial + one sum-rate evaluation on the estimated
    channel — the same design-view convention the phase and C_k teachers use.
    """
    g = np.abs(h_eff) ** 2 + 1e-30
    G = len(active_irs_ids)
    Dk  = float(cfg.D_k_bps_hz)
    lam = float(cfg.lambda_D)
    eps = float(getattr(cfg, 'epsilon_qp', 1e-3))
    best_J, best_w, best_f = None, None, _PW_AUX_FRACS[1]
    betas = (_PW_AUX_BETAS_CAPPED if getattr(cfg, 'power_aux_beta_cap', False)
             else _PW_AUX_BETAS)
    for b in betas:
        w = g ** b
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0.0:
            continue
        w = w / s
        for f in _PW_AUX_FRACS:
            w_p = w * (cfg.P_S * f)
            w_c = np.full(G + 1, cfg.P_S * (1.0 - f) / (G + 1))
            part = rate_computer.compute_rates_partial(
                phi, Phi, channels, w_p, w_c, active_irs_ids=active_irs_ids)
            C_k = _closed_form_ck(part, cfg)
            out = rate_computer.compute_sum_rate(
                phi, Phi, channels, w_p, w_c, C_k=C_k,
                active_irs_ids=active_irs_ids, sigma2=cfg.sigma2)
            R_tot = np.asarray(out['R_private']) + np.asarray(out['C_k'])
            sh    = np.maximum(0.0, Dk - R_tot) / (Dk + eps)
            J     = float(out['sum_rate']) - lam * float((sh ** 2).sum())
            if best_J is None or J > best_J:
                best_J, best_w, best_f = J, w, f
    if best_w is None:
        return None
    return {'split':   np.array([1.0 - best_f, best_f], dtype=float),
            'common':  np.ones(cfg.M + 1, dtype=float),   # masked+renormalised later
            'private': np.asarray(best_w, dtype=float)}


def _build_power_state(h_eff, rep=None, cfg=None, mag=None):
    """
    Power-head input, with the channel block RESCALED to survive LayerNorm.

    ⚠ 2026-07-24 RANKING FIX. Raw |h_eff| (~0.02) is ~44× smaller than the latent
    rep (~1.0), so PowerMLP's input LayerNorm was dominated by rep and collapsed
    the h-block — the per-user |h|² ranking survived only 27% of the time, and the
    head therefore concentrated private power on the WRONG user 85% of the time
    (top-1 agreement with |h|² was 15% vs the teacher's 95%), costing ~0.85 J at
    D_k=0.05. Dividing h_eff by its own per-state magnitude makes the block
    scale-free (like the PhaseMLP unit-phasor), so the joint LayerNorm preserves
    the ranking (measured 27% → 87%). Absolute |h| is dropped, but it cancels in
    the interference-limited SINR anyway; the rep still carries state context.
    """
    hs  = float(np.mean(np.abs(h_eff))) + 1e-9
    blk = np.concatenate([h_eff.real / hs, h_eff.imag / hs])
    if cfg is not None:
        blk = np.concatenate([blk, _power_scale_feats(h_eff, cfg)])
    if mag:
        blk = np.concatenate([blk, _power_mag_feats(h_eff, mag)])
    return blk if rep is None else np.concatenate([blk, rep])


def _power_mag_feats(h_eff, mode: str) -> np.ndarray:
    """[--power-mag-feats] The per-user MAGNITUDE ORDERING, handed over directly.

    The teacher allocation this head keeps failing to reach is a power law,
    private ∝ |h_eff|^{2β}, with β up to 16 — a 32nd power. From [Re h/hs, Im h/hs]
    the head would have to square its own inputs and then raise that to β, which
    a ReLU MLP approximates badly. Measured on 600 states of result_261, fitting
    the teacher's log-shares by least squares:

        current input   R² 0.245
        + |h|²          R² 0.601      ← the quantity the target is a power OF
        + log|h|        R² 0.559

    So over half the linearly available signal is simply absent from what the
    head is shown. This is NOT what --power-scale-feats supplies: that block is
    tanh(log2(cap/D_k)), a saturated FEASIBILITY flag, not the magnitude ordering.

    'sq'  : |h|² / mean|h|²           closest to the target's own form
    'log' : log|h|² - mean log|h|²    scale-free; a power law becomes LINEAR here,
                                      which is what the Dirichlet concentration
                                      needs, and it cannot blow up the row the way
                                      a raw ratio can

    ⚠ PowerMLP layer-norms its whole input row, so a block that dominates the row
    statistics squashes the h-block and destroys the per-user ranking — the
    2026-07-24 bug. 'sq' is a ratio and can reach 100× on a spread-out state;
    'log' cannot. Check either against scratchpad/ln_dilution.py before training.
    """
    a2 = np.abs(np.asarray(h_eff)) ** 2 + 1e-30
    if mode == 'sq':
        v = a2 / float(np.mean(a2))
    elif mode == 'log':
        v = np.log(a2)
    else:
        raise ValueError(
            f"--power-mag-feats: expected 'sq' or 'log', got {mode!r}")
    # Standardised PER STATE, and this is not cosmetic. Measured on 300 states
    # (K=10), appending the raw blocks cut the h-block's post-LayerNorm std by
    # 19% (sq) and 49% (log) — over the 15% line that flags the 2026-07-24
    # ranking bug. Standardised, both land at +1.5% and per-user ranking
    # retention RISES (top-1 95.7% → 96.7%, Spearman 0.795 → 0.818), because the
    # block now carries the ordering redundantly instead of crowding it out.
    # Both modes contribute mean 0 / std 1 to the row, so LayerNorm sees them
    # identically; only their internal shape differs.
    return (v - float(v.mean())) / (float(v.std()) + 1e-9)


def _power_scale_feats(h_eff, cfg) -> np.ndarray:
    """The ABSOLUTE scale the rescaled h-block deliberately throws away.

    Dividing h_eff by its own mean (above) is what makes the per-user RANKING
    survive LayerNorm, but it also removes every absolute quantity: after it the
    head knows who is strongest and nothing about whether ANYONE can reach D_k.
    Two states with identical rankings but a 10 dB difference in link budget look
    the same to it — which is why the policy cannot react to a P_S shift, and why
    a fixed power-fairness prior is the only thing carrying QoS.

    Restores what is missing, per user and once globally:
      tanh(log2(cap_k / D_k))   feasibility margin. Sign says whether user k can
                                reach the demand at all; magnitude says by how
                                much, in octaves.
      tanh(log10(mean Gamma))   which regime the link is in (noise- vs
                                interference-limited).

    ⚠ BOTH ARE BOUNDED TO [-1,1] ON PURPOSE. PowerMLP layer-norms the whole input
    row, so a block on a larger scale takes over the row statistics and squashes
    the others — the exact mechanism of the 2026-07-24 ranking bug, where the rep
    was 44x the h-block and cost ~0.85 J. A first version of this function
    returned raw log2(1+Gamma) and D_k/cap; measured, they came out 3.3x the
    h-block and cut its post-LayerNorm std by 55%, reintroducing that bug. Any
    feature added here must be checked against scratchpad/ln_dilution.py before
    it is trained on.

    Shortfall is deliberately NOT here: it is a function of the rate, which is a
    function of the power this head is about to choose, so it is not observable
    at decision time.
    """
    g = np.abs(np.asarray(h_eff)) ** 2 * cfg.P_S / cfg.sigma2      # per-user SNR
    cap = np.log2(1.0 + g)                                          # (K,)
    # [--power-feat-scale] log2(cap/D_k) was measured to have sd ~5.8-6.0 and a
    # p5..p95 span of -9.5..+7.0, so tanh() of it pins 70% of entries past 0.99:
    # the feature degenerates into a sign bit, "can user k reach D_k", and the
    # by-how-much that the docstring promises is thrown away. Dividing first
    # restores the gradation; s=4 measured 2.2-2.5% saturation while keeping ~90%
    # of the variance. Default 1.0 keeps the trained runs (r270/r273) reproducible.
    s = float(getattr(cfg, 'power_feat_scale', 1.0) or 1.0)
    margin = np.tanh(np.log2(np.maximum(cap, 1e-9) / cfg.D_k_bps_hz) / s)
    regime = np.tanh(np.log10(float(np.mean(g)) + 1e-12))
    return np.concatenate([margin, [regime]])


def _pf_priv(args) -> float:
    """α for the PRIVATE power axis: --power-fairness-priv, falling back to the
    global --power-fairness when unset (so the default behaviour is unchanged)."""
    v = getattr(args, 'power_fairness_priv', None)
    if v is None:
        v = getattr(args, 'power_fairness', 0.0) or 0.0
    return max(0.0, min(1.0, float(v)))


def _get_active_irs(phi: np.ndarray) -> np.ndarray:
    """
    Sorted 0-based indices of IRS panels with ≥1 assigned user.
    phi: (K,) int — 0=direct, 1..M=IRS (1-based).
    Used by PhaseMLP which expects 0-based indices.
    """
    return np.array(
        sorted({int(phi[k]) - 1 for k in range(len(phi)) if phi[k] > 0}),
        dtype=int,
    )


def _get_active_irs_ids(phi: np.ndarray) -> list:
    """
    Sorted 1-based physical IRS gids with ≥1 assigned user.
    Used by PowerMLP and RateComputer (which use 1-based gid convention).
    """
    return sorted(set(int(phi[k]) for k in range(len(phi)) if phi[k] > 0))


def _build_ck_state(demand: np.ndarray, R_private: np.ndarray,
                    R_c_group: dict, phi: np.ndarray, cfg) -> np.ndarray:
    """
    Build the [D_k, R_p_k, R_c_g_k, shortfall_k, phi_float_k] state for CkMLP.

    shortfall_k = max(0, D_k - R_p_k)  — minimum C_k still needed to hit QoS.
    phi_float_k = phi[k].astype(float) — group-ID context (0=direct, 1..M=IRS).

    Returns (5·K,) float.
    """
    K     = cfg.K
    R_c_g = np.zeros(K)
    for k in range(K):
        gid      = int(phi[k])
        R_c_g[k] = float(R_c_group.get(gid, 0.0))
    shortfall = np.maximum(0.0, demand - R_private)
    phi_float = phi[:K].astype(float)
    return np.concatenate([demand, R_private, R_c_g, shortfall, phi_float])


# ══════════════════════════════════════════════════════════════════════════════
# Generic helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_action_vec(phi: np.ndarray, phase_idx: np.ndarray,
                      w_c_vec: np.ndarray, active_irs_ids: list,
                      w_p: np.ndarray, C_k: np.ndarray,
                      cfg, n_levels: int) -> np.ndarray:
    """
    Fixed-size (d_action,) float encoding of the joint action for the
    action-conditioned critic Q(s, a).

    Encoding (each sub-vector normalised to ~[0, 1]):
      phi_f      : (K,)     assignment index / M
      phase_f    : (M*N,)   phase level     / (n_levels - 1)
      w_c_padded : (M+1,)   common power per group padded to full size / P_S
      w_p_n      : (K,)     private power / P_S
      C_k        : (K,)     common-rate share (bps/Hz)

    d_action = K + M*N + (M+1) + K + K
    """
    phi_f   = phi.astype(float) / max(cfg.M, 1)
    phase_f = phase_idx.flatten().astype(float) / max(n_levels - 1, 1)

    # Expand (G+1,) w_c_vec back to fixed (M+1,) — slot 0 = direct, slot gid = IRS gid
    w_c_pad = np.zeros(cfg.M + 1)
    w_c_pad[0] = float(w_c_vec[0])
    for j, gid in enumerate(active_irs_ids):
        if 1 <= gid <= cfg.M:
            w_c_pad[gid] = float(w_c_vec[j + 1])
    w_c_pad /= cfg.P_S

    w_p_n = w_p / cfg.P_S
    return np.concatenate([phi_f, phase_f, w_c_pad, w_p_n, C_k])


def _compute_irs_favored(env: ISTNEnv) -> np.ndarray:
    """
    Per-user indicator: True when the best IRS path (via estimated CSI)
    gives a stronger effective channel than the direct satellite link.

    NOTE: this is *IRS-favoured*, NOT the physical blockage flag
    (env.channels['su_blocked']), and it depends on the CURRENT env.Phi.
    It is passed to actor.extract_state() but IGNORED there — the affinity
    features already encode blockage implicitly (blocked link → a_{k,m} ≈ 0).
    """
    ch       = env.channels
    h_direct = np.abs(ch['g_SU_hat'])
    h_irs    = np.array([
        ch['beta'][m] * np.abs(np.sum(
            ch['g_SR_hat'][m].conj() * np.diag(env.Phi[m])[:, np.newaxis] * ch['g_RU_hat'][m],
            axis=0))
        for m in range(env.cfg.M)
    ])
    return h_irs.max(axis=0) > h_direct


def _irs_favored_target(env: ISTNEnv):
    """Routing-shaping target (Option 1): per user, the IRS id (1-based) whose
    estimated-CSI path is strongest, plus a mask = (best IRS path > direct).
    Reuses the _compute_irs_favored channel math → target is the IRS that makes the
    user 'IRS-favored'. Returns (target (K,) int 1-based, mask (K,) float)."""
    ch       = env.channels
    h_direct = np.abs(ch['g_SU_hat'])
    h_irs    = np.array([
        ch['beta'][m] * np.abs(np.sum(
            ch['g_SR_hat'][m].conj() * np.diag(env.Phi[m])[:, np.newaxis] * ch['g_RU_hat'][m],
            axis=0))
        for m in range(env.cfg.M)
    ])                                                       # (M, K)
    target = (h_irs.argmax(axis=0) + 1).astype(int)          # (K,) 1-based IRS id
    mask   = (h_irs.max(axis=0) > h_direct).astype(np.float64)
    return target, mask


def _cf_assign_rewards(env, phase_net, power_net, ck_net, phi, z_t, cfg, demand,
                       lamD: float) -> np.ndarray:
    """Counterfactual-assignment rewards for the q-head credit fix (COMA-style).
    For each user k and each link choice c ∈ {0=direct, 1..M=IRS}, re-run the
    (fixed) downstream policies (phase/power/Ck) under φ' = φ with φ_k=c and
    recompute the SAME reward (sum_rate − λ·Σ(shortfall/D_k)²) at nominal σ².
    Returns R_cf (K, M+1). Per-user marginal value → de-confounds assignment credit
    + discourages crowding (adding k to a busy IRS → its shared phase serves k
    poorly → low R_cf). Cost = K·(M+1) downstream forwards/step (flag-gated)."""
    K, M = cfg.K, cfg.M
    nc = M + 1
    D_k = demand
    eps = cfg.epsilon_qp
    sigma2 = cfg.sigma2                       # nominal (deterministic → clean signal)
    R_cf = np.zeros((K, nc))
    for k in range(K):
        for c in range(nc):
            phi2 = phi.copy(); phi2[k] = int(c)
            ai  = _get_active_irs(phi2)
            aid = _get_active_irs_ids(phi2)
            s_ph = _build_phase_state(env.channels, phi2, cfg, z_t)
            pidx, _, _ = phase_net.forward(s_ph, ai)
            Phi2 = env.phase_model.build_phi(env.phase_model.index_to_phase(pidx))
            heff = env.rate_computer.effective_channels_all(phi2, Phi2, env.channels)
            s_pw = np.concatenate([heff.real, heff.imag, z_t])
            wcv, wp, _, _ = power_net.forward(s_pw, aid)
            part = env.rate_computer.compute_rates_partial(
                phi2, Phi2, env.channels, wp, wcv, active_irs_ids=aid)
            s_ck = _build_ck_state(D_k, part['R_private'], part['R_c_group'], phi2, cfg)
            Ck, _, _ = ck_net.forward(s_ck, phi2, part['R_c_group'])
            # Design (phase/power/Ck above) used ĝ; the counterfactual reward is
            # scored on the true channel g, matching the real env.step reward.
            res = env.rate_computer.compute_sum_rate(
                phi2, Phi2, env.channels, wp, wcv, C_k=Ck,
                active_irs_ids=aid, sigma2=sigma2, use_true=True)
            Rt = res['R_private'] + res['C_k']
            sf = np.maximum(0.0, D_k - Rt)
            qp = lamD * float(np.sum((sf / (D_k + eps)) ** 2))
            R_cf[k, c] = float(res['sum_rate']) - qp
    return R_cf


def _save_agents(run_dir: str, actor, phase_net, power_net, ck_net, cfg,
                 critic=None) -> str:
    """Save all four actor networks (+ critic) + topology snapshot to run_dir/agents/."""
    agents_dir = os.path.join(run_dir, 'agents')
    os.makedirs(agents_dir, exist_ok=True)
    actor.save(agents_dir)
    phase_net.save(agents_dir)
    power_net.save(agents_dir)
    ck_net.save(agents_dir)
    # Persist the critic too so a --resume run can warm-start V(s) instead of
    # cold-starting it (a blind critic feeds noisy advantages to a good actor).
    if critic is not None:
        critic.save(agents_dir)
    # Topology snapshot so infer.py can rebuild a consistent cfg regardless of
    # what the user has set in params.py at inference time.
    topo = {
        'K':                cfg.K,
        'M':                cfg.M,
        'N':                cfg.N,
        'quantization_bits': cfg.quantization_bits,
        'P_S_dBm':          cfg.P_S_dBm,
        'D_k_bps_hz':       cfg.D_k_bps_hz,
    }
    with open(os.path.join(agents_dir, 'training_config.json'), 'w') as f:
        json.dump(topo, f, indent=2)
    return agents_dir


def _accum_grads(acc: dict, new: dict) -> dict:
    """In-place add a grad-dict into the accumulator (or initialise it).
    Handles empty dicts gracefully (e.g. when no IRS is active)."""
    if not new:                   # new is None or {} — nothing to add
        return acc
    if not acc:                   # acc is None or {} — start fresh
        return {k: v.copy() for k, v in new.items()}
    for k, v in new.items():
        acc[k] += v
    return acc


def _scale_grads(g: dict, factor: float) -> None:
    """In-place scale every entry in a grad-dict."""
    if g is None:
        return
    for k in g:
        g[k] *= factor


def _ppo_eff_adv(log_curr: float, log_old: float,
                 advantage: float, epsilon: float) -> tuple:
    """
    PPO-clip effective advantage for one sample.

    r_t = π_current(a|s) / π_old(a|s) = exp(log_curr − log_old)

    Returns
    -------
    eff_adv   : float  pass as `advantage` to compute_grads
                       = r_t × Â  (gradient flows) or 0 (clipped)
    ratio     : float  r_t (for monitoring)
    ppo_loss  : float  −min(r_t·Â, clip(r_t)·Â)  (for logging)
    """
    ratio = float(np.exp(np.clip(log_curr - log_old, -10.0, 10.0)))
    surr1 = ratio * advantage
    surr2 = float(np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)) * advantage
    if surr1 <= surr2:          # surr1 is the min → gradient: r_t × ∇log π
        return ratio * advantage, ratio, float(-surr1)
    else:                       # surr2 is min (clip active) → gradient zeroed
        return 0.0, ratio, float(-surr2)


def _ppo_eff_adv_batch(lp_new_b: np.ndarray, lp_old_b: np.ndarray,
                       adv_b: np.ndarray, epsilon: float) -> tuple:
    """
    Vectorised PPO-clip effective advantage for a mini-batch.

    Mirrors the scalar _ppo_eff_adv logic element-wise:
      eff_adv[b] = ratio[b] × adv[b]  when surr1 ≤ surr2  (not clipped)
                 = 0                   otherwise            (clipped)

    Returns
    -------
    eff_adv_b  : (B,) float  pass to compute_grads_batch as effective advantage
    ppo_loss_b : (B,) float  −min(surr1, surr2) per sample (for logging)
    clip_frac  : float       fraction of samples whose gradient was zeroed (clipped)
    """
    ratio_b = np.exp(np.clip(lp_new_b - lp_old_b, -10.0, 10.0))
    surr1_b = ratio_b * adv_b
    surr2_b = np.clip(ratio_b, 1.0 - epsilon, 1.0 + epsilon) * adv_b
    eff_adv_b  = np.where(surr1_b <= surr2_b, surr1_b, 0.0)
    ppo_loss_b = -np.minimum(surr1_b, surr2_b)
    clip_frac  = float(np.mean(surr1_b > surr2_b))
    return eff_adv_b, ppo_loss_b, clip_frac


def _clip_grad_norm(g: dict, max_norm: float) -> float:
    """In-place global gradient norm clipping. Returns the PRE-clip global norm
    (a key divergence diagnostic — a spike here precedes blow-ups)."""
    if not g:
        return 0.0
    norm = np.sqrt(sum(float(np.sum(v ** 2)) for v in g.values()))
    if not np.isfinite(norm):          # NaN/Inf grad → drop the update (don't poison weights)
        for k in g:
            g[k] = np.zeros_like(g[k])
        return norm
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        for k in g:
            g[k] *= scale
    return float(norm)


def _explained_variance(returns: np.ndarray, values: np.ndarray) -> float:
    """1 − Var(returns − V) / Var(returns).  ≈1 critic predicts well, ≤0 = useless."""
    var_y = float(np.var(returns))
    if var_y < 1e-12:
        return float('nan')
    return 1.0 - float(np.var(returns - values)) / var_y


def _warmup_critic(env, actor, critic, phase_net, power_net, ck_net,
                   cfg, args, demand: np.ndarray,
                   n_episodes: int, n_epochs: int) -> dict:
    """
    Calibrate V(s) to the (resumed) policy BEFORE PPO begins.

    A freshly-built critic is blind (explVar≈0) for the first hundreds of
    episodes; with a small GAE-λ its noisy 1-step advantages then erode a good
    warm-started actor. Here we roll the *current* policy for `n_episodes`,
    compute Monte-Carlo discounted returns (no bootstrap → unbiased value
    scale, matched to the current env/reward params), and fit the critic by
    supervised regression. The actors are NOT updated. Finally the target net
    is synced to the fitted weights so TD targets are calibrated from step one.
    """
    buf_s: list  = []
    buf_ret: list = []
    rng = np.random.default_rng(args.seed + 31337)

    for _ep in range(n_episodes):
        obs  = env.reset()
        ep_s: list = []
        ep_r: list = []
        for _step in range(args.steps):
            blocked = _compute_irs_favored(env)
            s_t     = actor.extract_state(obs, demand, blocked)
            phi, _, actor_info = actor.forward(s_t)
            z_t = actor_info['z_t']
            # [--no-ae] feed sub-actors the quantum readout o_hat instead of the AE latent
            rep = actor_info['o_hat'] if getattr(args, 'no_ae', False) else z_t
            active_irs     = _get_active_irs(phi)
            active_irs_ids = _get_active_irs_ids(phi)
            s_phase  = _build_phase_state(env.channels, phi, cfg, rep)
            phase_idx, _, _ = phase_net.forward(s_phase, active_irs)
            phases_rad   = env.phase_model.index_to_phase(phase_idx)
            proposed_Phi = env.phase_model.build_phi(phases_rad)
            h_eff   = env.rate_computer.effective_channels_all(
                          phi, proposed_Phi, env.channels)
            s_power = _build_power_state(
                h_eff, None if getattr(args, 'power_clean_input', False) else rep,
                cfg if getattr(args, 'power_scale_feats', False) else None,
                getattr(args, 'power_mag_feats', None))
            w_c_vec, w_p, _, _ = power_net.forward(s_power, active_irs_ids)
            partial = env.rate_computer.compute_rates_partial(
                phi, proposed_Phi, env.channels, w_p, w_c_vec,
                active_irs_ids=active_irs_ids)
            s_ck = _build_ck_state(
                demand, partial['R_private'], partial['R_c_group'], phi, cfg)
            C_k, _, _ = ck_net.forward(s_ck, phi, partial['R_c_group'])
            action = {'assignment': phi, 'phase_idx': phase_idx,
                      'w_p': w_p, 'w_c_vec': w_c_vec, 'C_k': C_k}
            obs, reward, _, _ = env.step(action)
            ep_s.append(s_t.copy())
            ep_r.append(float(reward))
        # Truncated Monte-Carlo return (γ^steps ≈ 0 → ≈ infinite-horizon).
        G = 0.0
        ep_ret = [0.0] * len(ep_r)
        for t in reversed(range(len(ep_r))):
            G = ep_r[t] + args.gamma * G
            ep_ret[t] = G
        buf_s.extend(ep_s)
        buf_ret.extend(ep_ret)

    rets = np.asarray(buf_ret, dtype=float)
    n    = len(buf_s)
    bs   = P.batch_size
    idx  = np.arange(n)
    # PopArt: initialise (μ,σ) from the warm-up returns so the normalised head
    # starts calibrated (no-op when popart disabled).
    critic.update_popart_stats(rets)
    for _epoch in range(n_epochs):
        rng.shuffle(idx)
        for start in range(0, n, bs):
            mb = idx[start:start + bs]
            _, g = critic.compute_grads_batch(
                [buf_s[i] for i in mb], [None] * len(mb), rets[mb])
            critic.apply_grads(g)

    V_pred = np.array([critic.forward(s) for s in buf_s])
    return {
        'n_samples': n,
        'ret_mean':  float(rets.mean()),
        'ret_std':   float(rets.std()),
        'v_mean':    float(V_pred.mean()),
        'expl_var':  _explained_variance(rets, V_pred),
    }


def _oracle_assign_env(env) -> np.ndarray:
    """Oracle assignment target (Common-Knowledge [N]): a blocked user → its building's
    (nearest) IRS, a non-blocked user → direct (0). Closed-form, no policy."""
    ch  = env.channels
    blk = np.asarray(ch['su_blocked'], dtype=bool)
    d   = np.linalg.norm(env.user_pos[:, None, :] - env.irs_pos[None, :, :], axis=2)  # (K,M)
    nearest = d.argmin(axis=1) + 1                              # 1-based IRS id
    return np.where(blk, nearest, 0).astype(int)


def _warmup_assignment(env, actor, cfg, args, demand: np.ndarray,
                       n_episodes: int, n_epochs: int) -> dict:
    """⭐ Supervised actor (Q-head) warm-up on the ORACLE routing (improvement ①a, 2026-06-14).

    probe_assignment_quality showed the agent never learns blocked→IRS at scale (live
    18-32% correct). The oracle target is closed-form (blocked→its IRS), so warm-start the
    VQC Q-head on it — mirrors _warmup_phase but on the actor: roll the frozen actor to
    collect (s_t, oracle_assign), then CE-fit via the actor's own grad path with eff_adv=+1
    (pass lp_old = current log π so ratio=1 → loss = -log π(oracle|s)). ae_weight=0 so only
    the encoding→QC→head routing path moves, not the AE reconstruction. Run BEFORE the phase
    warm-up so phase targets are built for GOOD routing (kills the result_32 post-warmup yank)."""
    buf = []
    rng = np.random.default_rng(args.seed + 71717)
    for _ep in range(n_episodes):
        obs = env.reset()
        for _step in range(args.steps):
            blocked = _compute_irs_favored(env)
            s_t = actor.extract_state(obs, demand, blocked)
            buf.append({'s_t': s_t.copy(), 'phi': _oracle_assign_env(env)})
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)
            obs = env._get_obs()
    n_buf = len(buf)
    if n_buf == 0:
        return {'n_samples': 0, 'match_initial': 0.0, 'match_final': 0.0}

    def _match(sample=300):
        idx = np.unique(np.linspace(0, n_buf - 1, min(sample, n_buf)).astype(int))
        n_ok = n = 0
        for i in idx:
            phi_pred, _, _ = actor.forward(buf[i]['s_t'], greedy=True)
            n_ok += int(np.sum(np.asarray(phi_pred) == buf[i]['phi']))
            n    += len(buf[i]['phi'])
        return 100.0 * n_ok / max(1, n)

    m_init = _match()
    bs = P.batch_size
    idx_arr = np.arange(n_buf)
    # Per-epoch match curve + early-stop at plateau (only ever stops EARLY → caps
    # wasted epochs; see 2026-06-15 warmup-epoch study). Patience = 3 checks, δ=0.5pp.
    #
    # ⚠ COST: this trains the VQC ACTOR — every epoch is a full PQC forward/SPSA
    # pass over the buffer, ~3 min/epoch at K10/M2 (NOT a cheap classical CE fit).
    # So each wasted check costs real wall-clock; hence the saturation exit below.
    log_every = max(1, min(n_epochs // 10, 10))   # cap @10: check/early-stop mỗi ≤10 ep
    # Saturation exit: assignment is a classification task, so once the match is
    # essentially perfect there is nothing left to learn and waiting out `patience`
    # non-improving checks is pure cost (r151: hit 100% @ep30, still ran to ep50
    # ≈ 1 extra hour). Stop immediately instead.
    sat_thr = 99.5
    best_m, no_imp, patience = m_init, 0, 3
    m_final = m_init
    for _epoch in range(n_epochs):
        rng.shuffle(idx_arr)
        for start in range(0, n_buf, bs):
            mb = idx_arr[start:start + bs]
            s_list   = [buf[i]['s_t'] for i in mb]
            phi_list = [buf[i]['phi'] for i in mb]
            lp_old = actor.compute_logprobs_batch(s_list, phi_list)
            adv    = np.ones(len(mb), dtype=float)
            out = actor.compute_logprobs_grads_batch(
                s_list, phi_list, lp_old, adv, P.ppo_epsilon, 0.0, 0.0)
            actor.apply_grads(out[3], out[4], out[5])      # g_ae, g_qc, g_xi
        if (_epoch + 1) % log_every == 0 or _epoch == n_epochs - 1:
            m_final = _match()
            print(f"      ① assign-warmup ep {_epoch+1:>4}/{n_epochs}: match {m_final:5.1f}%")
            if m_final >= sat_thr:
                print(f"      ① assign-warmup SATURATED @ep {_epoch+1} "
                      f"(match {m_final:.1f}% ≥ {sat_thr}% — nothing left to fit)")
                break
            if m_final > best_m + 0.5:
                best_m, no_imp = m_final, 0
            else:
                no_imp += 1
            if no_imp >= patience:
                print(f"      ① assign-warmup EARLY-STOP @ep {_epoch+1} "
                      f"(match plateaued ~{m_final:.1f}% for {patience} checks)")
                break
    return {'n_samples': n_buf, 'match_initial': float(m_init), 'match_final': float(m_final)}


def _warmup_phase(env, actor, phase_net, cfg, args, demand: np.ndarray,
                  n_episodes: int, n_epochs: int,
                  use_oracle_assign: bool = False) -> dict:
    """
    EXP-3 supervised PhaseMLP pretrain on closed-form ORACLE phase targets.

    Channel model (CSI/rate.py), PER-ELEMENT:
        h_irs[m,k] = Σ_n e^{jθ_{m,n}} · c_{m,n,k},  c = β·conj(g_SR[m])·g_RU[m,n,k].
    Each element sees its own Rayleigh draw, so each needs its OWN phase to co-phase
    its contribution. The oracle is closed-form per element for a single routed user
    (θ_n = arg g_SU[k] − arg c_{n,k}, quantised) plus coordinate ascent when an IRS
    carries several; see analysis/phase_oracle.oracle_phase_idx.

    ⚠ The "live 41% vs oracle 100%, index match 26%" headroom recorded @result_14
    predates the per-element refactor (g_RU was (M,K), one scalar per IRS, so all
    elements shared one optimal index). Those numbers do NOT transfer — re-measure
    with `python analysis/phase_oracle.py --ckpt <run>` before quoting them.

    Procedure: roll the (frozen) current actor for n_episodes, collect (s_phase,
    oracle_idx) tuples for each active IRS, then fit PhaseMLP via cross-entropy
    for n_epochs over the buffer (re-uses compute_grads_batch with eff_adv=+1 →
    equivalent to -log π(oracle|s) loss). Actor/Power/Ck are NOT touched.

    Run BEFORE critic warmup (so the critic fits returns under a warmer phase).
    """
    from analysis.phase_oracle import oracle_phase_idx

    buf: list = []
    rng = np.random.default_rng(args.seed + 91919)

    # ── Stage 1: rollout to collect supervised (s_phase, oracle_idx) tuples ──
    for _ep in range(n_episodes):
        obs = env.reset()
        for _step in range(args.steps):
            blocked = _compute_irs_favored(env)
            s_t = actor.extract_state(obs, demand, blocked)
            phi, _, info = actor.forward(s_t)               # frozen-actor sample
            z_t = info['z_t']
            rep = info['o_hat'] if getattr(args, 'no_ae', False) else z_t   # [--no-ae]
            # ① dual oracle warm-up: build phase targets for the ORACLE routing (blocked→IRS)
            #    instead of the random-actor routing → phase learns to align for the users that
            #    SHOULD be on each IRS (removes the post-warmup KL_ph yank seen in result_32).
            assign = _oracle_assign_env(env) if use_oracle_assign else phi
            active_irs = _get_active_irs(assign)
            s_phase = _build_phase_state(env.channels, assign, cfg, rep)
            target_idx = oracle_phase_idx(env.channels, assign, cfg)   # (M, N) int

            if active_irs.size > 0:
                buf.append({
                    's_phase':   s_phase.copy(),
                    'phi':       assign.copy(),
                    'phase_idx': target_idx.copy(),
                })
            # Advance mobility without applying an env.step (cheap; we don't
            # need rewards). Mirrors analysis/probe_phase_quality._advance.
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)
            obs = env._get_obs()

    n_buf = len(buf)
    if n_buf == 0:
        return {'n_samples': 0, 'ce_initial': float('nan'),
                'ce_final': float('nan'), 'match_initial': 0.0, 'match_final': 0.0}

    # ── Metric: PER-ELEMENT index match (live argmax == oracle) ──────────────
    # ⚠ 2026-07-21: this used to read `idx_live[0] == tgt[0]` — element 0 only.
    # Correct while all N elements shared one index (scalar g_RU); under
    # per-element g_RU each element is an independent choice, so element 0
    # sampled ~1/N of the signal and made the metric extremely noisy (and made
    # --phase-warmup-target gate on a coin flip). Now averaged over all N.
    # Random floor is unchanged at 100/L %. The TRAINING loss was always
    # over all elements — only this reported metric was wrong.
    def _eval_match(active_buf):
        n_pair = 0; n_match = 0.0; ce_sum = 0.0
        for tr in active_buf:
            active_irs = _get_active_irs(tr['phi'])
            for m in active_irs:
                # forward single IRS → (N,) indices, (N, n_levels) probs
                idx_live, _, probs = phase_net._forward_irs(tr['s_phase'][m], greedy=True)
                tgt = np.asarray(tr['phase_idx'][m])        # (N,) per-element
                n_match += float(np.mean(idx_live == tgt))
                n_pair  += 1
                ce_sum  -= float(np.mean(
                    np.log(probs[np.arange(tgt.shape[0]), tgt] + 1e-10)))
        return ce_sum / max(1, n_pair), 100.0 * n_match / max(1, n_pair)

    ce_init, match_init = _eval_match(buf)

    # ── Stage 2: supervised CE epochs (reuse compute_grads_batch w/ eff_adv=+1) ──
    bs = P.batch_size
    idx_arr = np.arange(n_buf)
    eff_one = np.ones(bs, dtype=float)
    # Per-epoch CE/match curve + early-stop at plateau (caps wasted epochs; only
    # ever stops EARLY). Sampled subset for the periodic check, full buf at the end.
    sub = buf[:300]
    log_every = max(1, min(n_epochs // 10, 10))   # cap @10: check/early-stop mỗi ≤10 ep
                                                  # (saturated assign-warmup dừng nhanh; phase dừng khi plateau)
    # Early-stop must tell a genuine HIGH-match plateau (converged) apart from the
    # SLOW-START plateau at the random floor (100/L %). r54 stopped @ep40 with match
    # 27.3% ≈ random (L=4 → 25%): the noisy near-floor match looked "flat" for 3
    # checks, but phase hadn't started learning (CE 1.387→1.384, still creeping). So
    # gate the plateau-stop behind ESCAPING the floor; n_epochs is the backstop if
    # phase genuinely can't escape (e.g. lossy-z_t VQC ceiling). This decouples the
    # stop criterion from the check cadence: log_every=10 no longer stops too early
    # (r54), and we never need the coarse log_every=60 that ran too long (r51).
    #
    # ⭐ 2026-07-21: the stop criterion now reads CE, NOT match. Match is a coarse
    # argmax statistic — it quantises progress and is noisy, so late in the fit it
    # sits inside the 0.5pp band while the model is still clearly improving.
    # Observed in result_153: EARLY-STOP fired @ep480 on "match plateaued ~74.5%
    # for 3 checks" while CE was descending monotonically 0.657→0.653→0.649→0.645.
    # A still-improving supervised fit was killed ~120 epochs short of the cap.
    # CE is the actual training objective, smooth, and monotone here.
    floor_ce = float(np.log(phase_net.n_levels))   # CE of a uniform predictor
    esc_ce   = 0.97 * floor_ce                     # must be meaningfully below uniform
    ce_rtol  = 0.005                               # 0.5% RELATIVE CE gain per check counts
    floor_m  = 100.0 / phase_net.n_levels          # kept for the log line only
    esc_thr  = floor_m + 12.0                      # kept for the pre-escape tag only
    best_ce, no_imp, patience = ce_init, 0, 3
    n_chk, stall_chk = 0, 20          # failsafe: abort if never escapes the floor
    ce_final, match_final = ce_init, match_init
    for _epoch in range(n_epochs):
        rng.shuffle(idx_arr)
        for start in range(0, n_buf, bs):
            mb = idx_arr[start:start + bs]
            batch = [buf[i] for i in mb]
            eff_b = eff_one if len(mb) == bs else np.ones(len(mb), dtype=float)
            _, _, grads = phase_net.compute_grads_batch(batch, eff_b, beta_entropy=0.0)
            if grads:
                phase_net.apply_grads(grads)
        if (_epoch + 1) % log_every == 0 or _epoch == n_epochs - 1:
            ce_chk, match_chk = _eval_match(sub)
            tag = "" if ce_chk <= esc_ce else "  (pre-escape: no early-stop)"
            print(f"      🌡 phase-warmup ep {_epoch+1:>4}/{n_epochs}: "
                  f"CE {ce_chk:.3f}  match {match_chk:5.1f}%{tag}")
            # plateau tracked on CE (smooth) with a RELATIVE tolerance
            if ce_chk < best_ce * (1.0 - ce_rtol):
                best_ce, no_imp = ce_chk, 0
            else:
                no_imp += 1
            # Stop policy: (a) target → stop only when match ≥ target (else run to cap);
            # (b) --full-phase-warmup → never early-stop; (c) default → CE escaped the
            #     uniform floor AND stopped improving for `patience` consecutive checks.
            _target = getattr(args, 'phase_warmup_target', None)
            if _target is not None:
                if match_chk >= _target:
                    print(f"      🌡 phase-warmup TARGET-HIT @ep {_epoch+1} "
                          f"(match {match_chk:.1f}% ≥ target {_target:.0f}%)")
                    break
                # chasing a target → no plateau early-stop (run to cap if unreached)
            elif not getattr(args, 'full_phase_warmup', False):
                if ce_chk <= esc_ce and no_imp >= patience:
                    print(f"      🌡 phase-warmup EARLY-STOP @ep {_epoch+1} "
                          f"(CE plateaued ~{ce_chk:.3f} — <{ce_rtol*100:.1f}% relative gain "
                          f"for {patience} checks)")
                    break
                # FAILSAFE: still glued to the uniform predictor after a long run
                # ⇒ the target is not learnable from this state (r151: CE 1.376 vs
                # ln4 1.386 for 1000 epochs). Escaping takes <150 epochs when the
                # encoding is right, so this is a genuine failure, not slow start.
                n_chk += 1
                if n_chk >= stall_chk and ce_chk > esc_ce:
                    print(f"      🌡 phase-warmup ABORT @ep {_epoch+1} — NOT LEARNING "
                          f"(CE {ce_chk:.3f} still ≈ uniform {floor_ce:.3f} after "
                          f"{n_chk} checks). Target likely not a function of the "
                          f"phase state — do NOT burn the remaining cap.")
                    break

    ce_final, match_final = _eval_match(buf)
    return {
        'n_samples':     n_buf,
        'ce_initial':    float(ce_init),
        'ce_final':      float(ce_final),
        'match_initial': float(match_init),
        'match_final':   float(match_final),
    }


def _divergence_signals(d: dict, gnorm_ema: dict) -> list:
    """Triggered divergence signals. Gradients are clipped before apply, so a
    large pre-clip norm is normal — we flag a sudden SPIKE vs its EMA, not the
    absolute value. KL and NaN/Inf are scale-independent red flags."""
    sig = []
    for k, v in d.get('scalars', {}).items():            # numerical blow-up
        if not np.isfinite(v):
            sig.append(f"{k}=NaN/Inf")
    for net, gn in d.get('gnorm', {}).items():           # sudden grad-norm spike
        ema = gnorm_ema.get(net)
        if ema is not None and ema > 1e-6 and np.isfinite(gn) and gn > 8.0 * ema and gn > 1.0:
            sig.append(f"‖∇{net}‖={gn:.1f} (8×↑ vs {ema:.1f})")
    for act, kl in d.get('kl', {}).items():              # policy moving too fast
        if np.isfinite(kl) and kl > 0.5:
            sig.append(f"KL_{act}={kl:.2f}>0.5")
    return sig


def _print_diag_panel(ep: int, d: dict, signals: list) -> None:
    """Compact multi-line health panel, printed each PPO update."""
    gn, kl, ent = d['gnorm'], d['kl'], d['ent']
    print(f"  ┄ diag[{ep+1}]  ‖∇‖ ae={gn['ae']:.2f} qc={gn['qc']:.2f} xi={gn['xi']:.2f} "
          f"ph={gn['phase']:.2f} pw={gn['power']:.2f} ck={gn['ck']:.2f} V={gn['critic']:.2f}")
    print(f"            KL q={kl['q']:.3f} ph={kl['ph']:.3f} pw={kl['pw']:.3f} ck={kl['ck']:.3f}"
          f"  │ ent q={ent['q']:.3f} ph={ent['ph']:.3f} pw={ent['pw']:.3f} ck={ent['ck']:.3f}")
    print(f"            V̄={d['v_mean']:+.2f} σV={d['v_std']:.2f} explVar={d['expl_var']:+.2f}"
          f"  │ λ|max| y={d['lam_y']:.2f} z={d['lam_z']:.2f}"
          f"  │ R̄={d['sum_rate']:.3f} − qp {d['qp']:.3f}")
    if signals:
        print(f"  ⚠ DIVERGENCE SIGNAL[{ep+1}]: " + " ; ".join(signals))


def _env_feasibility_report(cfg, seed: int, n_ep: int = 15, n_steps: int = 12) -> dict:
    """
    Probe the CURRENT environment with fixed reference policies to separate
    'geometry difficulty' from 'agent skill'. Reports the fraction of users that
    meet D_k under Direct-only / All-IRS / per-user best-of-both.

    ⚠ 'servable_frac' is a BASELINE FLOOR, not a ceiling (relabelled 07-21). It
    is the per-user max over two FIXED naive policies at hardcoded equal power
    and the current Phi. A real policy mixes routing per user and tunes
    phase/split/C_k, and beats it by a wide margin (K10/M2 @0.5: 59.8% here vs
    99.4% under oracle routing+phase+equal power). Read it as:
      agent-QoS  <  this  → agent is worse than a naive baseline → real problem
      agent-QoS  >  this  → normal and expected, NOT a bug
    """
    probe = ISTNEnv(cfg=cfg, seed=seed + 12345, n_steps_ep=n_steps,
                    reward_noise_avg=1)
    pol_direct, pol_irs = DirectOnlyPolicy(cfg), AllIRSPolicy(cfg)
    qos_direct, qos_irs, qos_best = [], [], []
    rt_direct,  rt_irs  = [], []
    blocked = []
    Dk = cfg.D_k_bps_hz
    for _ in range(n_ep):
        obs = probe.reset()
        for _ in range(n_steps):
            # per-user ACHIEVED rate under each reference link choice (true g, so
            # the servable fraction is apples-to-apples with the agent's true-g QoS)
            info_d = probe.rate_computer.compute_sum_rate(**_baseline_kwargs(pol_direct.act(obs), probe), use_true=True)
            info_i = probe.rate_computer.compute_sum_rate(**_baseline_kwargs(pol_irs.act(obs), probe), use_true=True)
            Rd = info_d['R_private'] + info_d['C_k']
            Ri = info_i['R_private'] + info_i['C_k']
            Rbest = np.maximum(Rd, Ri)
            qos_direct.append(np.mean(Rd >= Dk)); qos_irs.append(np.mean(Ri >= Dk))
            qos_best.append(np.mean(Rbest >= Dk))
            rt_direct.append(float(np.sum(Rd))); rt_irs.append(float(np.sum(Ri)))
            blocked.append(float(np.mean(probe.channels['su_blocked'])))
            obs, _, _, _ = probe.step(pol_irs.act(obs))
    rep = {
        'servable_frac':  float(np.mean(qos_best)),
        'qos_direct':     float(np.mean(qos_direct)),
        'qos_irs':        float(np.mean(qos_irs)),
        'rtot_direct':    float(np.mean(rt_direct)),
        'rtot_irs':       float(np.mean(rt_irs)),
        'blocked_frac':   float(np.mean(blocked)),
    }
    print(f"\n  Environment feasibility probe  (R_LoS={cfg.R_LoS_km} km, D_k={Dk}, λ_D={cfg.lambda_D})")
    print(f"  ────────────────────────────────────────────────────────────────")
    print(f"    Best-of-2-baselines (per-user max) : {rep['servable_frac']*100:5.1f}%   "
          f"← BASELINE FLOOR, *not* a ceiling")
    print(f"    QoS under Direct-only              : {rep['qos_direct']*100:5.1f}%   "
          f"(Σ R_tot {rep['rtot_direct']:.2f})")
    print(f"    QoS under All-IRS                  : {rep['qos_irs']*100:5.1f}%   "
          f"(Σ R_tot {rep['rtot_irs']:.2f})")
    print(f"    Direct link blocked                : {rep['blocked_frac']*100:5.1f}% of users")
    # ⚠ 2026-07-21 RELABELLED. This is the per-user max over TWO FIXED naive
    # baselines (all-direct, all-IRS), each at its own hardcoded equal power
    # split and the CURRENT Phi. It is NOT "the QoS ceiling for any policy":
    # a real policy mixes routing per user, optimises phase, the private/common
    # split and C_k, and routinely BEATS it — measured K10/M2 @R_LoS=0.5,
    # oracle routing+phase+equal power reaches QoS 99.4% against this number's
    # 59.8%. Treat it as a sanity FLOOR the agent should clear, nothing more.
    print(f"    ↑ per-user max over the two naive baselines at fixed power — a policy that")
    print(f"      mixes routing + tunes phase/power/C_k should EXCEED this comfortably.")
    return rep


def _baseline_kwargs(action: dict, env) -> dict:
    """Map a baseline action dict to compute_sum_rate kwargs (uses current Phi)."""
    assignment = action['assignment']
    active = sorted(set(int(a) for a in assignment if a > 0))
    return dict(assignment=assignment, Phi=env.Phi, channels=env.channels,
                w_p=action['w_p'], w_c_vec=action['w_c_vec'],
                active_irs_ids=active, sigma2=env.cfg.sigma2)


def _analysis_summary(hist: dict, n_diverge: int, args) -> None:
    """End-of-run analysis + heuristic recommendations for the next run."""
    W = 110
    rew = hist['episode_reward']; n = len(rew)
    if n < 10:
        return
    w = max(10, n // 20)                       # tail window ≈ last 5%
    tail   = lambda k: float(np.mean(hist[k][-w:])) if hist.get(k) else float('nan')
    best   = float(np.max(rew))
    best_ep = int(np.argmax(rew)) + 1
    # plateau check: did best improve in the last 20% of episodes?
    cut = max(1, int(0.8 * n))
    plateaued = (np.max(rew[cut:]) <= np.max(rew[:cut]) + 1e-6) if cut < n else False
    ev   = tail('mean_expl_var')   if 'mean_expl_var'   in hist else float('nan')
    clipq= tail('mean_clip_q')     if 'mean_clip_q'     in hist else float('nan')
    qos  = tail('mean_qos_rate')
    qp   = tail('mean_qp_penalty') if 'mean_qp_penalty' in hist else float('nan')
    sr   = tail('mean_sum_rate')

    print(f"\n  {'═'*W}")
    print(f"  ANALYSIS SUMMARY  (tail window = last {w} ep)")
    print(f"  {'─'*W}")
    print(f"    reward  tail={tail('episode_reward'):8.1f}   best={best:8.1f} @ep {best_ep}"
          f"   {'⚠ PLATEAUED (no new best in last 20%)' if plateaued else '↗ still improving'}")
    print(f"    rates   ΣR_tot={sr:6.3f}   QoS={qos*100:4.1f}%   qp_penalty={qp:6.3f}")
    print(f"    health  explVar={ev:+.2f}   clip_q={clipq:.2f}   divergence-signals={n_diverge}")
    print(f"  {'─'*W}")
    print(f"  Recommendations for next run:")
    recs = []
    if not np.isnan(qos) and qos < 0.6 and not np.isnan(qp) and qp > 1.0:
        recs.append(f"QoS low ({qos*100:.0f}%) + penalty high → compare against the "
                    f"baseline FLOOR in the feasibility probe (not a ceiling): below it = "
                    f"worse than a naive policy; above it, look at routing/phase/C_k, not λ_D")
    if not np.isnan(ev) and ev < 0.2:
        recs.append(f"explVar low ({ev:+.2f}) → critic struggling: ↑lr_critic or ↓reward variance "
                    f"(↑reward_noise_avg)")
    if not np.isnan(clipq) and clipq > 0.3:
        recs.append(f"clip_q high ({clipq:.2f}) → policy moving fast: ↓ppo_epochs or ↑reward_noise_avg")
    if n_diverge > n // 50:
        recs.append(f"{n_diverge} divergence signals → consider ↓lr or tighter grad-clip")
    if plateaued:
        recs.append("plateaued → raise difficulty (R_LoS) via --resume, or stop")
    if not recs:
        recs.append("no red flags — safe to raise difficulty (R_LoS↑ / case↑) via --resume")
    for r in recs:
        print(f"    • {r}")
    print(f"  {'═'*W}\n")


def _compute_beta(ep: int, n_episodes: int) -> float:
    """Linear entropy annealing: β₀ → β_min by beta_entropy_anneal_end of training."""
    end_ep = max(1, int(P.beta_entropy_anneal_end * n_episodes))
    prog   = min(1.0, ep / end_ep)
    return max(P.beta_entropy_min, P.beta_entropy * (1.0 - prog))


def _aux_w(args, name: str, ep: int, n_episodes: int) -> float:
    """
    Aux-teacher weight at episode `ep`, with optional linear anneal to zero.

    `--<head>-aux-weight w` alone is CONSTANT for the whole run (the original and
    still the default): the teacher must not decay the way the one-shot phase
    warm-up did, since PPO eroded that (alignment 15.6% → 7.0% over 1300 ep).

    `--phase-aux-anneal-end f` makes the phase teacher decay linearly w → 0 by
    f·n_episodes. Motivated by cost: the teacher runs `oracle_phase_idx` every
    rollout step and adds a term to every PPO epoch, and PPO updates are 58-65%
    of wall-clock. Once the phase head tracks the oracle there is little left to
    teach — the measured phase gap is only 5-7% of the remaining J gap.

    ⚠ The ASSIGNMENT teacher can never anneal, and that is enforced HERE rather
    than by simply not exposing a flag: it is what fixed the routing-identity
    collapse (7/7 runs peaked at ep~2500 then fell to 40-66% QoS without it), so a
    stray attribute must not be able to decay it back into that failure.
    """
    w0 = float(getattr(args, f'{name}_aux_weight', 0.0) or 0.0)
    end_frac = None if name == 'assign' else getattr(args, f'{name}_aux_anneal_end', None)
    if w0 <= 0.0 or not end_frac:
        return w0
    end_ep = max(1, int(float(end_frac) * n_episodes))
    return w0 * max(0.0, 1.0 - min(1.0, ep / end_ep))


def _compute_lr_frac(ep: int, n_episodes: int) -> float:
    """Linear LR decay after lr_decay_start warm-up: 1.0 → lr_min_frac."""
    start_ep = int(P.lr_decay_start * n_episodes)
    if ep < start_ep:
        return 1.0
    prog = (ep - start_ep) / max(1, n_episodes - start_ep)
    return max(P.lr_min_frac, 1.0 - prog)


def _next_run_dir(base: str = "results") -> tuple:
    os.makedirs(base, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(base, f"result_{i}")):
        i += 1
    path = os.path.join(base, f"result_{i}")
    os.makedirs(path)
    return path, i


def _moving_avg(arr: list, w: int) -> np.ndarray:
    a = np.array(arr, dtype=float)
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode='valid')


# ══════════════════════════════════════════════════════════════════════════════
# Results persistence
# ══════════════════════════════════════════════════════════════════════════════

def _save_hyperparameters(run_dir: str, cfg, actor: QuantumActor,
                           critic: ClassicalCritic,
                           phase_net: PhaseMLP, power_net: PowerMLP,
                           ck_net: CkMLP,
                           args, run_id: int) -> None:
    hp = {
        "run_id":    run_id,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        # Training seed — recorded so a reported number can be traced to its run and
        # so multi-seed tables (mean±std over TRAINING seeds, not just eval seeds) can
        # be assembled after the fact. Runs before 2026-07-25 did not store it; they
        # all used params.seed_default (0) unless --seed was passed explicitly.
        "seed":      int(getattr(args, 'seed', P.seed_default)),
        "system": {
            "K":              cfg.K,
            "M":              cfg.M,
            "N":              cfg.N,
            "P_S_dBm":        cfg.P_S_dBm,
            "f_GHz":          cfg.f_GHz,
            "h_SR_km":        cfg.h_SR_km,
            "G_S_dBi":        cfg.G_S_dBi,
            "G_U_dBi":        cfg.G_U_dBi,
            "noise_mean_dBW": cfg.noise_mean_dBW,
            "noise_var_dBW":  cfg.noise_var_dBW,
            "kappa":          cfg.kappa,
            "R_LoS_km":       cfg.R_LoS_km,
            "irs_spawn_radius_frac": getattr(cfg, 'irs_spawn_radius_frac', 1.0),
            "user_free_radius_frac": getattr(cfg, 'user_free_radius_frac', 1.0),
            "balanced_blocked_spawn": getattr(cfg, 'balanced_blocked_spawn', False),
            "h_IRS_km":       cfg.h_IRS_km,
            "beta_IRS":       cfg.beta_IRS,
            "beta_blocking":  cfg.beta_blocking,
            "d_block_km":     cfg.d_block_km,
            "path_loss_exp":  cfg.path_loss_exp,
            "user_speed_mps": cfg.user_speed_mps,
            "D_k_bps_hz":     cfg.D_k_bps_hz,
            "lambda_D":       cfg.lambda_D,
            "epsilon_qp":     cfg.epsilon_qp,
            "lagrangian":            getattr(args, 'lagrangian', False),
            "lagrangian_qos_target": getattr(args, 'qos_target', None),
            "lagrangian_lambda_lr":  getattr(args, 'lambda_lr', None),
            "lagrangian_lambda_clip": [getattr(args, 'lambda_min', None),
                                       getattr(args, 'lambda_max', None)],
            "counterfactual_assign": getattr(args, 'counterfactual_assign', False),
            "cf_coef":               getattr(args, 'cf_coef', None),
            "cf_anneal_end":         getattr(args, 'cf_anneal_end', None),
        },
        "training": {
            "n_episodes":      args.episodes,
            "n_steps_per_ep":  args.steps,
            "gamma":           args.gamma,
            "ae_weight":                P.ae_weight,
            "beta_entropy":             P.beta_entropy,
            "beta_entropy_min":         P.beta_entropy_min,
            "beta_entropy_anneal_end":  P.beta_entropy_anneal_end,
            "lr_decay_start":           P.lr_decay_start,
            "lr_min_frac":              P.lr_min_frac,
            "batch_size":               P.batch_size,
            "ppo_epsilon":   P.ppo_epsilon,
            "ppo_epochs":    P.ppo_epochs,
            "lr_actor_ae":   P.lr_actor_ae,
            "lr_actor_qc":     P.lr_actor_qc,
            "lr_actor_xi":     P.lr_actor_xi,
            "lr_critic":       P.lr_critic,
            "n_shots_train":      P.n_shots_train,
            "n_shots_eval":       P.n_shots_eval,
            "eval_interval":      P.eval_interval,
            "gae_lambda":         P.gae_lambda,
            "reward_noise_avg":   getattr(P, 'reward_noise_avg', 1),
            "ae_pretrain_epochs": P.ae_pretrain_epochs,
            "ae_pretrain_lr":     P.ae_pretrain_lr,
            "seed":               args.seed,
        },
        "actor": {
            "actor_mode":    getattr(args, 'actor_mode', 'quantum'),
            "no_ae":         bool(getattr(args, 'no_ae', False)),
            "d_s":           actor.d_s,
            "n_latent":      actor.N_LATENT,
            "n_qubits":      actor.N_QUBITS,
            "n_var_layers":  actor.N_VAR_LAYERS,
            "n_shots":       actor.n_shots,
            "hidden_ae":     actor.N_HIDDEN_AE,
            "hidden_post":   actor.N_HIDDEN_POST,
            "n_choices":        actor.n_choices,
            "data_reuploading": actor.DATA_REUPLOADING,
            "architecture":  (f"{actor.d_s} → LN → "
                              f"{'→'.join(str(h) for h in actor.N_HIDDEN_AE)} → "
                              f"{actor.N_LATENT}(z) | QC({actor.N_QUBITS}q,L="
                              f"{actor.N_VAR_LAYERS}) → {actor.N_QUANTUM}(o) → "
                              f"{actor.N_LATENT + actor.N_QUANTUM} → "
                              f"{'→'.join(str(h) for h in actor.N_HIDDEN_POST)} → "
                              f"{cfg.K}×{actor.n_choices}"),
        },
        "critic": {
            "d_state":      critic.d_state,
            "d_action":     critic.d_action,
            "d_in":         critic.d_in,
            "hidden":       critic.hidden,
            "gamma":        critic.gamma,
            "architecture": critic.architecture_str,
            "lr":           P.lr_critic,
            "grad_clip":    P.grad_clip_critic,
            "popart":       getattr(critic, 'popart', False),
            "popart_beta":  getattr(critic, 'pa_beta', None),
        },
        "phase_net": {
            "d_s":      phase_net.d_s,
            "M":        cfg.M,
            "N":        cfg.N,
            "n_levels": phase_net.n_levels,
            "hidden":   P.n_hidden_phase,
            "lr":       P.lr_phase,
        },
        "power_net": {
            "d_s":    cfg.K,
            "K":      cfg.K,
            "M":      cfg.M,
            "P_S":    cfg.P_S,
            "hidden": P.n_hidden_power,
            "lr":     P.lr_power,
            "power_fairness": max(0.0, min(1.0, float(getattr(args, 'power_fairness', 0.0) or 0.0))),
            "power_priv_frac": float(getattr(args, 'power_priv_frac', 0.8)),
            # infer.py must rebuild the SAME input or the head is fed a state it
            # never saw; it reads this flag rather than guessing from d_s.
            "power_scale_feats": bool(getattr(args, 'power_scale_feats', False)),
            "no_entangle": bool(getattr(args, 'no_entangle', False)),
            "power_feat_scale": float(getattr(args, 'power_feat_scale', 1.0) or 1.0),
            "power_mag_feats": (getattr(args, 'power_mag_feats', None) or None),
            "power_fairness_priv": _pf_priv(args),
        },
        "ck_net": {
            "d_s":    3 * cfg.K,
            "K":      cfg.K,
            "hidden": P.n_hidden_ck,
            "lr":     P.lr_ck,
        },
        # ⚡ levers that change the EXECUTED policy or the loss but live only on the
        # CLI — record them or a finished run cannot be reproduced/identified.
        # (r169-r178 predate this and have to be told apart by probing.)
        "levers": {
            "assign_aux_weight": float(getattr(args, 'assign_aux_weight', 0.0) or 0.0),
            "phase_aux_weight":  float(getattr(args, 'phase_aux_weight', 0.0) or 0.0),
            "phase_aux_anneal_end": getattr(args, 'phase_aux_anneal_end', None),
            "phase_aux_sweeps":  int(getattr(args, 'phase_aux_sweeps', 0)),
            "phase_aux_hspread": float(getattr(args, 'phase_aux_hspread', 0.0) or 0.0),
            "power_aux_anneal_end": getattr(args, 'power_aux_anneal_end', None),
            "ck_aux_weight":     float(getattr(args, 'ck_aux_weight', 0.0) or 0.0),
            "power_aux_weight":  float(getattr(args, 'power_aux_weight', 0.0) or 0.0),
            "closed_form_ck":    bool(getattr(args, 'closed_form_ck', False)),
            "freeze_phase":      bool(getattr(args, 'freeze_phase', False)),
            "beta_entropy_pwr_private": _beta_pwr_priv(args),
        },
    }
    with open(os.path.join(run_dir, "hyperparameters.json"), 'w') as f:
        json.dump(hp, f, indent=2)


def _save_plots(run_dir: str, hist: dict, args, window: int = 10) -> list:
    if not HAVE_MPL or args.no_plots:
        return []

    saved = []
    eps   = np.arange(1, len(hist['episode_reward']) + 1)
    w     = min(window, max(1, len(eps) // 5))

    C = dict(reward='steelblue', lpg='tomato', lae='darkorange',
             td='mediumseagreen', feas='mediumpurple',
             ohm='royalblue', znm='darkcyan')

    def _plot_single(ax, key, label, color, pct=False):
        vals = [v * 100 for v in hist[key]] if pct else hist[key]
        ax.plot(eps, vals, alpha=0.30, color=color, lw=1)
        if len(eps) >= w:
            ma = _moving_avg(vals, w)
            ax.plot(np.arange(w, len(eps) + 1), ma,
                    color=color, lw=2.0, label=f'{w}-ep avg')
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.legend(fontsize=8)

    # 1. Reward curve
    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_single(ax, 'episode_reward', 'Total episode reward', C['reward'])
    ax.set_title('Episode Reward', fontsize=11)
    fig.tight_layout()
    p = os.path.join(run_dir, 'reward_curve.png')
    fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)

    # 2. Losses (2 panels: combined actor + critic TD)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_single(axes[0], 'mean_L_actor', 'L_actor  (combined actor loss)', C['lpg'])
    _plot_single(axes[1], 'mean_td_loss', 'TD loss  (critic)',               C['td'])
    for ax in axes:
        ax.set_title(ax.get_ylabel(), fontsize=10)
    fig.suptitle('Training Losses', fontsize=12, y=1.01)
    fig.tight_layout()
    p = os.path.join(run_dir, 'losses.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); saved.append(p)

    # 3. Feasibility
    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_single(ax, 'feasibility_rate', 'Feasible steps (%)', C['feas'], pct=True)
    ax.set_ylim(-2, 105)
    ax.axhline(100, color='gray', ls='--', lw=0.8, label='100 %')
    ax.set_title('Feasibility Rate', fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(run_dir, 'feasibility.png')
    fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)

    # 4. Quantum features (2 panels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_single(axes[0], 'mean_o_hat_norm', '‖o_hat‖  (quantum output)', C['ohm'])
    _plot_single(axes[1], 'mean_z_norm',     '‖z_t‖  (latent vector)',    C['znm'])
    for ax in axes:
        ax.set_title(ax.get_ylabel(), fontsize=10)
    fig.suptitle('Quantum Feature Statistics', fontsize=12, y=1.01)
    fig.tight_layout()
    p = os.path.join(run_dir, 'quantum_stats.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); saved.append(p)

    # 5. Sub-actor losses (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_single(axes[0], 'mean_L_phase', 'L_phase (phase-shift MLP)',   'saddlebrown')
    _plot_single(axes[1], 'mean_L_power', 'L_power (power alloc. MLP)',  'steelblue')
    _plot_single(axes[2], 'mean_L_ck',    'L_ck    (C_k split MLP)',     'darkgreen')
    for ax in axes:
        ax.set_title(ax.get_ylabel(), fontsize=10)
    fig.suptitle('Sub-actor Losses', fontsize=12, y=1.01)
    fig.tight_layout()
    p = os.path.join(run_dir, 'sub_actor_losses.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); saved.append(p)

    # 6. Rates (sum-rate & QoS fraction)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_single(axes[0], 'mean_sum_rate', 'Σ R_tot  (bps/Hz)',            'teal')
    _plot_single(axes[1], 'mean_qos_rate', 'QoS fraction  (users / K)',   'darkorchid')
    for ax in axes:
        ax.set_title(ax.get_ylabel(), fontsize=10)
    axes[1].set_ylim(-0.05, 1.05)
    fig.suptitle('Rate & QoS Statistics', fontsize=12, y=1.01)
    fig.tight_layout()
    p = os.path.join(run_dir, 'rates.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); saved.append(p)

    # 7. Summary dashboard (2×2)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    _plot_single(axes[0, 0], 'episode_reward',   'Total reward',            C['reward'])
    _plot_single(axes[0, 1], 'feasibility_rate', 'Feasible steps (%)',      C['feas'], pct=True)
    _plot_single(axes[1, 0], 'mean_L_actor',     'L_actor  (combined)',     C['lpg'])
    _plot_single(axes[1, 1], 'mean_td_loss',     'TD loss  (critic)',        C['td'])
    fig.suptitle(f'Training Summary  —  result_{run_dir.split("_")[-1]}',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    p = os.path.join(run_dir, 'summary.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); saved.append(p)

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def _pick_best_agents(run_dir):
    """Run analysis/pick_resume_ckpt.py on a run-dir → best (sweet-spot) ckpt's agents/ path,
    or None. Lets --resume take a results/result_N dir and auto-select the best agents."""
    import subprocess, re as _re
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'analysis', 'pick_resume_ckpt.py')
    try:
        out = subprocess.run([sys.executable, script, '--run', run_dir],
                             capture_output=True, text=True, timeout=90).stdout
    except Exception:
        return None
    m = _re.search(r'(?:PICK|BEST-AVAILABLE):\s*ep_(\d+)', out)
    if not m:
        return None
    p = os.path.join(run_dir, 'checkpoints', f'ep_{int(m.group(1)):05d}', 'agents')
    return p if os.path.isdir(p) else None


def _build_components(args):
    """
    Build all environment and agent objects without printing anything.
    Called before the log file is opened so that hyperparameters.json
    can be written first.
    """
    # Entropy-schedule overrides (mutate P.* so _compute_beta + the json dump + the
    # banner all reflect them). Default None → params.py value unchanged.
    if getattr(args, 'beta_entropy', None) is not None:
        P.beta_entropy = float(args.beta_entropy)
    if getattr(args, 'beta_entropy_min', None) is not None:
        P.beta_entropy_min = float(args.beta_entropy_min)
    if getattr(args, 'beta_entropy_anneal_end', None) is not None:
        P.beta_entropy_anneal_end = float(args.beta_entropy_anneal_end)
    cfg_overrides = {}
    if getattr(args, 'D_k', None) is not None:
        cfg_overrides['D_k_bps_hz'] = float(args.D_k)
    if getattr(args, 'R_LoS_km', None) is not None:
        cfg_overrides['R_LoS_km'] = float(args.R_LoS_km)
    if getattr(args, 'lambda_D_fixed', None) is not None:
        cfg_overrides['lambda_D'] = float(args.lambda_D_fixed)
    if getattr(args, 'P_S_dBm', None) is not None:
        cfg_overrides['P_S_dBm'] = float(args.P_S_dBm)   # R3 scale-K: fix P_S across K sweep
    if getattr(args, 'irs_spawn_frac', None) is not None:
        cfg_overrides['irs_spawn_radius_frac'] = float(args.irs_spawn_frac)
    if getattr(args, 'user_free_frac', None) is not None:
        cfg_overrides['user_free_radius_frac'] = float(args.user_free_frac)
    # ── RESUME run-dir → AUTO-PICK BEST ckpt (chọn agents tốt nhất) ───────────
    # Nếu --resume trỏ vào RUN-DIR (results/result_N, có checkpoints/ nhưng không phải
    # agents/ckpt) → tự chọn sweet-spot ckpt qua pick_resume_ckpt. Fallback: ep mới nhất.
    _rs = getattr(args, 'resume', None)
    if (_rs and os.path.isdir(os.path.join(_rs, 'checkpoints'))
            and not os.path.isfile(os.path.join(_rs, 'actor_config.json'))):
        _best = _pick_best_agents(_rs)
        if _best:
            print(f"  ⭐ RESUME AUTO-PICK best ckpt: {_best}")
            args.resume = _best
        else:
            _eps = sorted(d for d in os.listdir(os.path.join(_rs, 'checkpoints'))
                          if d.startswith('ep_'))
            if _eps:
                args.resume = os.path.join(_rs, 'checkpoints', _eps[-1], 'agents')
                print(f"  ⚠ pick_resume_ckpt no pick → fallback latest: {args.resume}")
    # ── RESUME case auto-detect ──────────────────────────────────────────────
    # On --resume the env MUST match the checkpoint's Case (K,M); otherwise the
    # loaded actor (sized for the ckpt) gets a wrong-sized state from an env built
    # off params.py's CURRENT ACTIVE CASE → matmul shape crash in _dual_encode.
    # Read K,M from the ckpt's actor_config.json (B field = M) and override cfg so
    # resume is robust to whatever Case params.py is set to. make_config re-derives
    # all per-case hypers (P_S, N, …) from the overridden K.
    if getattr(args, 'resume', None):
        _rd = args.resume
        if (not os.path.isfile(os.path.join(_rd, 'actor_config.json'))
                and os.path.isfile(os.path.join(_rd, 'agents', 'actor_config.json'))):
            _rd = os.path.join(_rd, 'agents')
        _acfg = os.path.join(_rd, 'actor_config.json')
        if os.path.isfile(_acfg):
            with open(_acfg) as _f:
                _ac = json.load(_f)
            _ck_K = int(_ac.get('K', P.K)); _ck_M = int(_ac.get('B', P.M))
            if _ck_K != P.K or _ck_M != P.M:
                cfg_overrides['K'] = _ck_K
                cfg_overrides['M'] = _ck_M
                print(f"  ⚠ RESUME CASE AUTO-DETECT: ckpt K={_ck_K} M={_ck_M} "
                      f"≠ params.py K={P.K} M={P.M} → overriding env cfg to match ckpt")
    cfg = make_config(**cfg_overrides)
    if cfg_overrides:
        print(f"  ⚠ CFG OVERRIDE: {cfg_overrides}  (D_k_HYPOTHESIS test)")
    # Carried on cfg rather than threaded through _build_power_state's four call
    # sites (train rollout, warm-up, aux teacher, infer) — every one of them
    # already has cfg, none of them would otherwise need a new argument.
    cfg.power_feat_scale = float(getattr(args, 'power_feat_scale', 1.0) or 1.0)
    cfg.power_aux_beta_cap = bool(getattr(args, 'power_aux_beta_cap', False))
    if cfg.power_aux_beta_cap:
        print(f"  ⚙ power-aux teacher beta grid CAPPED: {_PW_AUX_BETAS_CAPPED} "
              f"(default {_PW_AUX_BETAS})")
    if getattr(args, 'power_scale_feats', False):
        print(f"  ⚙ power-scale-feats ON · margin tanh divisor s="
              f"{cfg.power_feat_scale:g}"
              f"{'  (s=1 saturates ~70% of entries)' if cfg.power_feat_scale == 1.0 else ''}")
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps,
                  reward_noise_avg=getattr(P, 'reward_noise_avg', 1))

    # [--no-ae] ablation: kill the AE reconstruction objective + pretrain so the encoder/λ
    # are driven ONLY by the task through the quantum path (the bypass is dropped in the head).
    if getattr(args, 'no_ae', False):
        P.ae_weight = 0.0
        _keep_pt = getattr(args, 'no_ae_keep_pretrain', False)
        if not _keep_pt:
            args.no_pretrain = True
        print("  🔬 --no-ae: z_t bypass DROPPED (head+sub-actors use o_hat); ae_weight→0, "
              + ("pretrain KEPT (--no-ae-keep-pretrain → encoder bootstrap → vanilla-adaptive, "
                 "no oracle-warmup needed)." if _keep_pt else
                 "pretrain OFF. Encoder/λ forced through quantum path."))

    if getattr(args, 'actor_mode', 'quantum') == 'classical':
        # DNN baseline (paper A2 / R5): pure-MLP actor replacing AE+VQC+head.
        _pol_hidden = (tuple(args.pol_hidden)
                       if getattr(args, 'pol_hidden', None)
                       else getattr(P, 'classical_pol_hidden', (256, 128)))
        actor = ClassicalActor(
            cfg,
            n_latent   = P.n_latent,
            enc_hidden = getattr(P, 'classical_enc_hidden', (128,)),
            pol_hidden = _pol_hidden,
            lr         = getattr(P, 'lr_classical_actor', P.lr_actor_qc),
            seed       = args.seed,
        )
        print(f"  🧮 CLASSICAL ACTOR (DNN baseline): {actor.num_params():,} params "
              f"(enc {list(actor.ENC_HIDDEN)} → z{actor.N_LATENT} → pol {list(actor.POL_HIDDEN)})")
    else:
        # --readout ablation (fresh runs): swap ONLY the VQC readout observable design,
        # keeping every other hyperparameter identical. None → params.py default (r1).
        # Ignored on --resume (the saved actor config wins).
        _ro = getattr(args, 'readout', None)
        if _ro is None:
            _rm  = getattr(P, 'vqc_readout_mode', 'generic')
            _ezz = P.extra_zz_pairs
            _fzz = getattr(P, 'full_zz_pairs', ())
        elif _ro == 'r1':
            _rm, _ezz, _fzz = 'r1', (), ()
        elif _ro == 'single-z':
            _rm, _ezz, _fzz = 'single_z', (), ()
        elif _ro == 'nn-zz':
            _rm, _ezz, _fzz = 'generic', (), ()
        elif _ro == 'full-zz':
            _rm, _ezz, _fzz = 'generic', (), P._b1_zz_pairs(P.n_qubits, P.M)
        else:
            raise ValueError(f"unknown --readout {_ro!r}")
        if _ro is not None:
            print(f"  🔬 READOUT ABLATION: --readout {_ro} → mode={_rm} "
                  f"extra_zz={len(_ezz)} full_zz={len(_fzz)}")
        actor = QuantumActor(
            cfg,
            n_qubits         = P.n_qubits,
            n_latent         = P.n_latent,
            n_hidden_ae      = P.n_hidden_ae,
            n_hidden_post    = P.n_hidden_post,
            n_var_layers     = P.n_var_layers,
            n_shots          = P.n_shots_train,
            lr_ae            = P.lr_actor_ae,
            lr_qc            = P.lr_actor_qc,
            lr_xi            = P.lr_actor_xi,
            data_reuploading = P.data_reuploading,
            ae_pretrain_lr   = P.ae_pretrain_lr,
            spsa_n_reps      = P.spsa_n_reps,
            spsa_epsilon     = P.spsa_epsilon,
            extra_cz_pairs   = P.extra_cz_pairs,
            extra_zz_pairs   = _ezz,
            full_zz_pairs    = _fzz,
            readout_mode     = _rm,
            softmax_head     = getattr(P, 'vqc_softmax_head', False),
            softmax_beta_init= getattr(P, 'vqc_softmax_beta_init', 1.0),
            no_ae            = getattr(args, 'no_ae', False),
            no_entangle      = getattr(args, 'no_entangle', False),
            seed             = args.seed,
        )
        if getattr(args, 'no_entangle', False):
            print("  ⚙ ABLATION --no-entangle: product-state circuit "
                  "(no CZ chain, no cross-block bridges); parameter count and "
                  "R1 readout unchanged")
    # State-value critic V(s) — action-independent baseline (d_action=0).
    # A Q(s,a) baseline cancels the action's own value, giving E[advantage]≈0
    # and a vanishing policy gradient; V(s) is the correct PPO/GAE baseline.
    critic = ClassicalCritic(
        actor.d_s,
        d_action = 0,
        hidden   = P.critic_hidden,
        lr       = P.lr_critic,
        gamma    = args.gamma,
        seed     = args.seed,
        popart       = getattr(P, 'popart_enabled', False),
        popart_beta  = getattr(P, 'popart_beta', 0.1),
        popart_sigma_floor = getattr(P, 'popart_sigma_floor', 1e-2),
    )
    # [--no-ae] sub-actors consume o_hat (N_QUANTUM) instead of the AE latent z_t (n_latent)
    _rep_dim = (actor.N_QUANTUM if getattr(args, 'no_ae', False)
                else P.n_latent)              # "system spatial latent" fed to Phase/Power
    # [--power-clean-input] PowerMLP drops the rep (o_hat|z_t) → reads ONLY h_eff (2K).
    # Power/Ck don't train the encoding, so o_hat is pure shot-noise to them (blocks
    # concentration). h_eff = per-user effective channel = the discriminative signal needed.
    if getattr(args, 'power_clean_input', False) and getattr(args, 'counterfactual_assign', False):
        raise SystemExit('--power-clean-input is not compatible with --counterfactual-assign')
    _pw_rep_dim = 0 if getattr(args, 'power_clean_input', False) else _rep_dim
    phase_net = PhaseMLP(
        d_s      = 5 * cfg.K + _rep_dim,      # c^SRU (2K) + g_SU (2K) + rep + phi_mask (K)
        M        = cfg.M,
        N        = cfg.N,
        n_levels = env.n_phase_levels,
        hidden   = P.n_hidden_phase,
        lr       = P.lr_phase,
        seed     = args.seed,
    )
    # Carried as an attribute rather than a constructor argument so that agents
    # saved before this flag existed still load: getattr defaults it to 0.0,
    # which reproduces the uniform teacher exactly.
    phase_net.aux_hspread_gamma = float(getattr(args, 'phase_aux_hspread', 0.0) or 0.0)
    if phase_net.aux_hspread_gamma > 0.0:
        print(f"  ⚙ phase teacher redistributed by log-hspread, gamma="
              f"{phase_net.aux_hspread_gamma:g} (mean weight unchanged)")
    # --power-scale-feats appends 2K+1 absolute-scale features (see
    # _power_scale_feats). Gated by a flag so the change is one controlled
    # variable against the existing runs rather than a silent redefinition of
    # the head's input.
    _pw_scale_dim = (cfg.K + 1) if getattr(args, 'power_scale_feats', False) else 0
    # --power-mag-feats appends the per-user |h|^2 ordering (K dims). Separate
    # flag from --power-scale-feats because they supply different quantities and
    # measured differently: the scale block is a saturated feasibility flag, this
    # one is the magnitude the teacher target is a power of (linear R² .245→.601).
    _pw_mag_dim = cfg.K if getattr(args, 'power_mag_feats', None) else 0
    power_net = PowerMLP(
        d_s    = 2 * cfg.K + _pw_scale_dim + _pw_mag_dim + _pw_rep_dim,  # h_eff Re+Im (2K) [+ scale] [+ mag] [+ rep]
        K      = cfg.K,
        M      = cfg.M,
        P_S    = cfg.P_S,
        hidden = P.n_hidden_power,
        lr     = P.lr_power,
        seed   = args.seed,
    )
    # [--ck-logit-spread] override the ck-stability floor (module global, read at call-time
    # by _ck_group_softmax → setting it here applies to every C_k softmax this run).
    import RL.sub_actors as _sub_actors
    _ck_spread = float(getattr(args, 'ck_logit_spread', 8.0))
    _sub_actors.CK_LOGIT_SPREAD = _ck_spread
    print(f"  ck-stability: CK_LOGIT_SPREAD = {_ck_spread:g}"
          + ("   ⚠ DISABLED (C_k may collapse to one-hot)" if _ck_spread >= 1e6 else ""))
    ck_net = CkMLP(
        d_s    = 5 * cfg.K,                  # D_k + R_p + R_c + shortfall + phi_float
        K      = cfg.K,
        hidden = P.n_hidden_ck,
        lr     = P.lr_ck,
        seed   = args.seed,
    )

    # ── Resume / curriculum transfer ──────────────────────────────────────────
    # Load the 4 trained actors (same architecture → same K/M/nq) to continue
    # training under new env/reward params (e.g. higher λ_D, larger R_LoS).
    # The critic is ALSO warm-started when a saved critic exists in the resume
    # dir: a freshly-initialised V(s) is blind for the first hundreds of
    # episodes (explVar≈0), and with gae_lambda small its noisy advantages
    # erode the good warm-started actor. Loading V(s) gives calibrated targets
    # from step one. (Minor return-scale shifts from changing λ_D / R_LoS are
    # re-fit far faster than a cold start.) If no critic file is present
    # (e.g. runs saved before critic persistence) we fall back to a fresh one.
    args._resumed_critic = False
    resume_dir = getattr(args, 'resume', None)
    if resume_dir:
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"--resume dir not found: {resume_dir}")
        # Auto-detect actor type from the saved actor_config.json 'mode' field
        # (classical runs write mode='classical'); falls back to quantum.
        _ac_cfg_path = os.path.join(resume_dir, 'actor_config.json')
        _is_classical = False
        if os.path.isfile(_ac_cfg_path):
            with open(_ac_cfg_path) as _f:
                _is_classical = (json.load(_f).get('mode') == 'classical')
        if _is_classical:
            actor = ClassicalActor.from_dir(resume_dir, seed=args.seed)
            print(f"  🧮 CLASSICAL ACTOR resumed: {actor.num_params():,} params")
        else:
            actor = QuantumActor.from_dir(resume_dir, seed=args.seed)
            # Keep training-time shot count / spsa settings from params.py
            actor.n_shots     = P.n_shots_train
            actor.spsa_n_reps = P.spsa_n_reps
        phase_net = PhaseMLP.from_dir(resume_dir, seed=args.seed)
        if getattr(args, 'power_clean_input', False):
            print("  🔧 --power-clean-input: PowerMLP RE-INITIALISED (clean h_eff-only input, "
                  "o_hat dropped); actor/phase/ck/critic resumed as usual.")
        else:
            power_net = PowerMLP.from_dir(resume_dir, seed=args.seed)
        ck_net    = CkMLP.from_dir(resume_dir, seed=args.seed)
        # Warm-start the critic if it was persisted alongside the actors.
        if os.path.isfile(os.path.join(resume_dir, 'critic_config.json')):
            critic = ClassicalCritic.from_dir(resume_dir, seed=args.seed)
            args._resumed_critic = True
            # [Case2 PopArt bug-fix] On a curriculum-ramp resume (λ_D and R_LoS
            # change → return SCALE shifts), the loaded PopArt stats (μ,σ) are
            # STALE for the new ramp. With pa_initialized=True the warm-up only
            # EMA-adapts them (β=0.1 → ~10 rollouts of mis-scaled critic → noisy
            # advantages right when the ramp transition is most fragile). Force a
            # re-snap: reset pa_initialized so the critic warm-up (which rolls the
            # NEW policy in the NEW env) re-initialises μ,σ to the new return scale
            # and re-fits the normalised head before any actor update.
            if getattr(critic, 'popart', False):
                critic.pa_initialized = False

    # ③ power-fairness lever (--power-fairness α): set on power_net for BOTH the
    # fresh and resumed paths — forward() reads it. 0.0 = off (pure learned head).
    # --power-fairness-priv / --power-priv-frac are applied here too so a RESUME
    # picks them up from the CLI rather than the checkpoint's baked-in values.
    power_net.power_fairness = max(0.0, min(1.0, float(getattr(args, 'power_fairness', 0.0) or 0.0)))
    power_net.power_fairness_priv = _pf_priv(args)
    power_net.power_priv_frac = float(getattr(args, 'power_priv_frac', 0.8))

    return cfg, env, actor, critic, phase_net, power_net, ck_net


def train(args) -> None:
    # Ensure the console accepts UTF-8 box-drawing characters on Windows.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    run_dir, run_id = _next_run_dir()
    os.makedirs(run_dir, exist_ok=True)

    # 1. Build all components silently (no prints yet).
    cfg, env, actor, critic, phase_net, power_net, ck_net = \
        _build_components(args)

    # 2. hyperparameters.json — written first, before the log is opened.
    _save_hyperparameters(run_dir, cfg, actor, critic,
                          phase_net, power_net, ck_net, args, run_id)

    # 3. training_log.txt — opened second; captures all subsequent stdout.
    log_path   = os.path.join(run_dir, 'training_log.txt')
    log_f      = open(log_path, 'w', encoding='utf-8', buffering=1)
    tee        = _Tee(log_f)
    sys.stdout = tee
    try:
        _train_body(args, run_dir, run_id,
                    cfg, env, actor, critic, phase_net, power_net, ck_net)
    finally:
        sys.stdout = tee._stdout
        log_f.close()


def _train_body(args, run_dir: str, run_id: int,
                cfg, env, actor, critic,
                phase_net, power_net, ck_net) -> None:
    # ── Derived values ────────────────────────────────────────────────────────
    demand = np.full(cfg.K, cfg.D_k_bps_hz)   # (K,) — demand per user

    # ── Print environment summary ─────────────────────────────────────────────
    W = 130
    _gpu_str = f"GPU (CuPy)" if GPU_BACKEND else "CPU (NumPy)"
    print(f"\n{'═'*W}")
    print(f"  Quantum AC Training  ·  IRS-assisted RSMA Satellite Comm.  ·  QC backend: {_gpu_str}")
    print(f"{'═'*W}")
    print(f"\n{cfg.summary()}\n")

    d_s = actor.d_s
    n_phase_params = cfg.M * cfg.N * env.n_phase_levels

    print(f"  Actor Pipeline  (4 stages per step)")
    print(f"  {'─'*64}")
    print(f"    1. Quantum IRS selection")
    print(f"       State  : a_t ∈ R^{d_s}  =  {cfg.K}×({cfg.M}+2)+2×{cfg.M}  (affinity ‖ D_k ‖ |g_SU|)")
    _dec_arch = ' → '.join(str(h) for h in reversed(actor.N_HIDDEN_AE))
    print(f"       Encoder: {d_s} → B2-DualAE → {actor.N_LATENT}  (z_t)"
          f"  [IRS:{2*cfg.M}D + User:2×{cfg.K}D→P]")
    print(f"       Decoder: {actor.N_LATENT} → {_dec_arch} → {d_s}  (AE regulariser)")
    # actor.EXTRA_CZ_PAIRS keeps the CONFIGURED pairs; --no-entangle clears them
    # in the circuit module, not on the actor, so the summary has to check the
    # ablation flag or it prints CZ bridges that are not being applied.
    if getattr(actor, 'NO_ENTANGLE', False):
        _cz_desc = " NO-CZ(ablation)"
    else:
        _cz_desc = (f"+{len(actor.EXTRA_CZ_PAIRS)}×CZ-bridge"
                    if actor.EXTRA_CZ_PAIRS else "")
    if actor.FULL_ZZ_PAIRS:
        _zz_desc = f"+{len(actor.FULL_ZZ_PAIRS)}×ZZ-full(B1)"
    elif actor.EXTRA_ZZ_PAIRS:
        _zz_desc = f"+{len(actor.EXTRA_ZZ_PAIRS)}×ZZ-cross(B4)"
    else:
        _zz_desc = ""
    print(f"       QC     : {actor.N_LATENT}-dim → {actor.N_QUBITS} qubits "
          f"(H+U_enc+U_var×{actor.N_VAR_LAYERS}{_cz_desc}) → {actor.N_QUANTUM}-dim{_zz_desc}  (o_hat)")
    _post_arch = ' → '.join(str(h) for h in actor.N_HIDDEN_POST)
    print(f"       Post-NN: {actor.N_LATENT+actor.N_QUANTUM} → {_post_arch}"
          f" → {cfg.K}×{actor.n_choices} logits")
    print(f"       Output : φ ∈ {{0,…,{cfg.M}}}^{cfg.K}  "
          f"({actor.n_choices}^{cfg.K} combinations)")
    print(f"    2. PhaseMLP  (IRS phase shifts — active IRS only)")
    _d_phase = phase_net.d_s
    print(f"       State  : per ELEMENT ∈ R^{_d_phase}  "
          f"([∠c^SRU cos+sin({2*cfg.K}) ‖ ∠g_SU cos+sin({2*cfg.K}) ‖ "
          f"z_t({P.n_latent}) ‖ phi_mask({cfg.K})])  unit-phasor")
    print(f"       Net    : {phase_net.architecture}  "
          f"(shared weights, N={cfg.N} elements/IRS, categorical)")
    print(f"       Output : phase_idx ∈ {{0..{env.n_phase_levels-1}}}^{{G×{cfg.N}}}  "
          f"(G ≤ {cfg.M} active IRS, expanded to {cfg.M}×{cfg.N})")
    print(f"    3. PowerMLP  (power allocation)")
    _d_power = 2 * cfg.K + P.n_latent
    print(f"       State  : h_eff+z_t ∈ R^{_d_power}  "
          f"([Re+Im h_eff({2*cfg.K}) ‖ z_t({P.n_latent})])")
    print(f"       Net    : {power_net.architecture}  softmax → ×P_S")
    print(f"       Output : w_c_vec ∈ R^{{G+1}}  (per-group common, G≤{cfg.M}),  "
          f"w_p ∈ R^{cfg.K}  (private)  [Σ=P_S]")
    _a_pf, _a_pv = (getattr(power_net, 'power_fairness', 0.0),
                    getattr(power_net, 'power_fairness_priv', 0.0))
    if _a_pf > 0.0 or _a_pv > 0.0:
        print(f"       ③ POWER-FAIRNESS  α_split/common={_a_pf:.2f}  α_private={_a_pv:.2f}  "
              f"f={getattr(power_net, 'power_priv_frac', 0.8):.2f}")
        if _a_pv < _a_pf:
            print(f"          ↳ private axis left FREE to concentrate — needs a "
                  f"demand-filling C_k or QoS collapses")
    print(f"    4. CkMLP  (common-rate split)")
    _d_ck = 5 * cfg.K
    print(f"       State  : [D_k, R_p_k, R_c_g_k, shortfall_k, phi_k] ∈ R^{_d_ck}  (per user)")
    print(f"       Net    : {ck_net.architecture}  within-group softmax → C_k")
    print(f"       Output : C_k ∈ R^{cfg.K}  (Σ_{{k∈g}} C_k = R_c_g per group)")

    print(f"\n  Critic  (State-Value MLP  V(s))")
    print(f"    Input        : s_t ({critic.d_state})  =  {critic.d_in} total")
    print(f"    Architecture : {critic.architecture_str}  (V)")

    print(f"\n  Training plan")
    print(f"    Episodes       : {args.episodes}")
    print(f"    Steps/episode  : {args.steps}")
    print(f"    PPO epochs     : {P.ppo_epochs}  (ε={P.ppo_epsilon},  batch={P.batch_size})")
    print(f"    Total steps    : {args.episodes * args.steps:,}")
    print(f"    γ (discount)   : {args.gamma}")
    print(f"    AE loss weight : {P.ae_weight}")
    print(f"    β entropy      : {P.beta_entropy} → {P.beta_entropy_min}  "
          f"(anneal to {int(P.beta_entropy_anneal_end*100)}% of training)")
    print(f"    LR schedule    : warm-up {int(P.lr_decay_start*100)}% eps, "
          f"then linear decay to ×{P.lr_min_frac}")
    print(f"    GAE lambda (λ) : {P.gae_lambda}")
    print(f"    AE pretrain    : {P.ae_pretrain_epochs} steps @ lr={P.ae_pretrain_lr}")
    print(f"    Data reupload  : {P.data_reuploading}")
    print(f"    lr  ω/λθ/ξ/ψ  : "
          f"{P.lr_actor_ae}/{P.lr_actor_qc}/{P.lr_actor_xi}/{P.lr_critic}")
    print(f"    Shots (train)  : {P.n_shots_train}")
    print(f"    Seed           : {args.seed}")
    n_batches   = -(-args.steps // P.batch_size)          # ceil div
    n_updates   = P.ppo_epochs * n_batches * P.n_rollout_episodes
    if P.spsa_n_reps > 0:
        # SPSA: 2 * n_reps circuits per update (independent of n_params)
        n_qc_per_update = 2 * P.spsa_n_reps * P.batch_size
        n_qc_per_ep     = n_qc_per_update * P.ppo_epochs * n_batches
        speedup_vs_ps   = (4 * (1 + actor.N_VAR_LAYERS) * actor.N_QUBITS) / (2 * P.spsa_n_reps)
        grad_desc = (f"SPSA  n_reps={P.spsa_n_reps}  eps={P.spsa_epsilon}"
                     f"  ({n_qc_per_update} circuits/update,  {speedup_vs_ps:.0f}x vs param-shift)")
    else:
        n_qc_evals  = 4 * (1 + actor.N_VAR_LAYERS) * actor.N_QUBITS
        n_qc_per_ep = n_qc_evals * P.batch_size * P.ppo_epochs * n_batches
        grad_desc   = (f"parameter-shift  ({n_qc_evals} circuits/sample,  "
                       f"{n_qc_per_ep:,} per episode)")
    print(f"\n{'─'*W}")
    print(f"  QC gradient : {grad_desc}")
    print(f"  Rollout buf : {P.n_rollout_episodes} ep x {args.steps} steps = "
          f"{P.n_rollout_episodes * args.steps} transitions/PPO call")
    print(f"{'─'*W}\n")

    # ── Column header strings (used after optional pre-training) ─────────────
    # NOTE: ClpQ/ClpP/ClpW/ClpC columns removed from the on-screen table for a
    # cleaner display (clip data still saved to metrics.npz). Rollout episodes
    # (no PPO update) show "—" for all loss columns, like L_ae.
    hdr = (f"  {'Ep':>5}  {'Reward':>9}  {'BestReward':>10}  "
           f"{'L_pg':>8}  {'L_ae':>9}  {'L_phs':>7}  {'L_pw':>7}  {'L_ck':>7}  "
           f"{'TD':>8}  "
           f"{'R_tot':>8}  {'R/user':>7}  {'QoS/K':>7}  {'IRS/K':>7}  {'Blk/K':>7}  {'s/ep':>6}")
    sep = (f"  {'─'*5}  {'─'*9}  {'─'*10}  "
           f"{'─'*8}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}  "
           f"{'─'*8}  "
           f"{'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}")

    # ════════════════════════════════════════════════════════════════════════
    # Environment feasibility probe (calibrates λ_D vs R_LoS before training)
    # ════════════════════════════════════════════════════════════════════════
    try:
        _env_feasibility_report(cfg, seed=args.seed)
    except Exception as _e:
        print(f"  (feasibility probe skipped: {_e})")

    # ════════════════════════════════════════════════════════════════════════
    # AE Hot-start pre-training (reconstruction only, no RL signal)
    # ════════════════════════════════════════════════════════════════════════
    if getattr(args, 'resume', None):
        print(f"  ⟳ RESUMED actors from: {args.resume}")
        if getattr(args, '_resumed_critic', False):
            print(f"    (AE pre-training skipped; critic V(s) warm-started from resume dir)")
        else:
            print(f"    (AE pre-training skipped; no saved critic found → critic re-initialised)")
    if getattr(args, 'freeze_phase', False):
        print(f"  ❄ EXP-1 FREEZE-PHASE: PhaseMLP params FROZEN (grad apply skipped). "
              f"Assignment trains vs stationary phase → watch IRS-share catch up to oracle (F2 test).")
    if (P.ae_pretrain_epochs > 0 and not getattr(args, 'no_pretrain', False)
            and not getattr(args, 'resume', None)
            and getattr(args, 'actor_mode', 'quantum') == 'quantum'):  # classical has no AE
        print(f"  Pre-training AE encoder/decoder for {P.ae_pretrain_epochs} steps …")
        rng_pre = np.random.default_rng(args.seed + 9999)
        pre_loss_log = []
        for pre_step in range(P.ae_pretrain_epochs):
            s_rand = rng_pre.standard_normal(actor.d_s)
            # Fix: demand slice is not normalised by _group_norm — inject real value
            s_rand[cfg.K * cfg.M : cfg.K * (cfg.M + 1)] = cfg.D_k_bps_hz
            loss   = actor.pretrain_ae_step(s_rand)
            pre_loss_log.append(loss)
            if (pre_step + 1) % 100 == 0:
                avg = float(np.mean(pre_loss_log[-100:]))
                print(f"    [{pre_step+1:>4}/{P.ae_pretrain_epochs}]  L_ae = {avg:.4f}")
        print(f"  AE pre-training complete  "
              f"(final L_ae ≈ {float(np.mean(pre_loss_log[-20:])):.4f})")

    # ════════════════════════════════════════════════════════════════════════
    # EXP-3 Phase warmup (supervised PhaseMLP pretrain on closed-form oracle)
    # ════════════════════════════════════════════════════════════════════════
    # Closes the +59pp live→oracle alignment gap observed @ result_14 ep_00700.
    # Runs BEFORE critic warmup so the critic fits returns under a warmer phase.
    # Composable with --freeze-phase: warmup runs first (trains PhaseMLP), then
    # freeze-phase takes effect during PPO (prevents PPO from eroding the warmup
    # alignment — see V3 verdict result_15 in docs/Training-Case-2.txt where PPO
    # decayed initial +0.028 reward boost back to -0.029 over 400ep).
    _oracle_wu = getattr(args, 'oracle_warmup', False)
    if _oracle_wu:
        # ① and 🌡 have VERY different cost profiles and budgets, so they get their
        # own flags (defaulting to the phase ones = old behaviour):
        #   ① assign — trains the VQC actor, ~3 min/epoch @K10/M2, saturates ~30 ep
        #   🌡 phase — classical MLP over N elements, cheap/epoch, needs many more
        aw_eps    = int(getattr(args, 'assign_warmup_episodes', None)
                        or getattr(args, 'phase_warmup_episodes', 8))
        aw_epochs = int(getattr(args, 'assign_warmup_epochs', None)
                        or getattr(args, 'phase_warmup_epochs', 30))
        print(f"  🎯 ① ORACLE ASSIGNMENT WARMUP: {aw_eps} rollout ep × {aw_epochs} CE epochs "
              f"on oracle routing (blocked→its-IRS) …")
        _aw = _warmup_assignment(env, actor, cfg, args, demand, aw_eps, aw_epochs)
        print(f"    assignment warmup done: N={_aw['n_samples']}  "
              f"per-user match {_aw['match_initial']:.1f}%→{_aw['match_final']:.1f}%")
    if getattr(args, 'phase_warmup', False) or _oracle_wu:
        pw_eps    = int(getattr(args, 'phase_warmup_episodes', 8))
        pw_epochs = int(getattr(args, 'phase_warmup_epochs', 30))
        compose_note = (" + ❄ FREEZE during PPO (EXP-3c: prevent PPO erosion of warmup)"
                        if getattr(args, 'freeze_phase', False) else "")
        _oa_note = " on ORACLE routing" if _oracle_wu else ""
        print(f"  🌡 EXP-3 PHASE WARMUP: {pw_eps} rollout ep × {pw_epochs} CE epochs "
              f"on oracle_phase_idx targets{_oa_note} (actors frozen){compose_note} …")
        _pw = _warmup_phase(env, actor, phase_net, cfg, args, demand,
                            pw_eps, pw_epochs, use_oracle_assign=_oracle_wu)
        if _pw['n_samples'] > 0:
            print(f"    phase warmup done: N={_pw['n_samples']}  "
                  f"CE μ={_pw['ce_initial']:.3f}→{_pw['ce_final']:.3f}  "
                  f"per-IRS match {_pw['match_initial']:.1f}%→{_pw['match_final']:.1f}%")
        else:
            print(f"    phase warmup skipped: no active-IRS rollout instances "
                  f"(assignment routed everyone to direct)")
        # Dump immediate post-warmup ckpt for clean attribution of warmup vs PPO damage.
        # Per result_15 V3 analysis (probe_assignment_decomp ckpt sweep), oracle ceiling
        # dropped -0.088 between r11 baseline and r15 ep_00100 — but ep_00100 is post
        # 100 PPO ep, so warmup vs PPO contribution unclear. This ckpt enables clean
        # ablation: probe ep_00000 = pure warmup effect (no PPO yet).
        _ep00000 = os.path.join(run_dir, 'checkpoints', 'ep_00000')
        _save_agents(_ep00000, actor, phase_net, power_net, ck_net, cfg, critic)
        print(f"    ⭐ post-warmup ckpt dumped → {_ep00000} (probe with probe_assignment_decomp.py)")

    # ── Critic warm-up (resume only): fit V(s) to the resumed policy ──────────
    # Calibrates the value scale before PPO so advantages are meaningful from
    # episode 1 — essential when the resume dir has no saved critic (e.g.
    # result_6) and the critic would otherwise cold-start blind.
    # Critic warm-up: fit V(s) to MC returns of the (frozen) current policy before PPO.
    # Run for BOTH resume AND fresh runs [Case 2 result_2 fix H2]: on fresh, the RL heads
    # are random (post AE-pretrain) but the rolled returns still give V a calibrated SCALE
    # → better early advantages → bootstrap out of the blind-critic stuck equilibrium.
    # (One-shot on fresh is less durable than on resume since the policy then changes, but
    #  combined with reward_noise_avg↑ it gives the critic a fighting start.)
    wu_eps = getattr(P, 'critic_warmup_episodes', 0)
    if wu_eps > 0:
        wu_epochs = getattr(P, 'critic_warmup_epochs', 30)
        _wu_src = 'resumed' if getattr(args, 'resume', None) else 'fresh (random) '
        print(f"  Warming up critic V(s) [{_wu_src} policy]: {wu_eps} rollout ep × "
              f"{wu_epochs} fit epochs (actors frozen) …")
        _wu = _warmup_critic(env, actor, critic, phase_net, power_net, ck_net,
                             cfg, args, demand, wu_eps, wu_epochs)
        print(f"    critic warm-up done: N={_wu['n_samples']}  "
              f"return μ={_wu['ret_mean']:+.2f} σ={_wu['ret_std']:.2f}  "
              f"V̄={_wu['v_mean']:+.2f}  explVar={_wu['expl_var']:+.2f}")

    print(f"\n{'─'*W}")
    print(hdr)
    print(sep)

    # ── History buffers ───────────────────────────────────────────────────────
    hist: dict = {k: [] for k in (
        'episode_reward', 'mean_L_actor', 'mean_L_pg', 'mean_L_ae', 'mean_td_loss',
        'feasibility_rate', 'mean_o_hat_norm', 'mean_z_norm',
        'mean_sum_rate', 'mean_per_user_rate', 'mean_qos_rate',
        'mean_irs_rate', 'mean_blocked_rate',
        'mean_L_phase', 'mean_L_power', 'mean_L_ck',
        'mean_clip_q', 'mean_clip_phase', 'mean_clip_power', 'mean_clip_ck',
        # divergence diagnostics (carry-forward between PPO updates)
        'mean_kl_q', 'mean_kl_ph', 'mean_kl_pw', 'mean_kl_ck',
        'mean_qp_penalty', 'mean_expl_var', 'mean_v', 'mean_v_std',
        'mean_lam_y', 'mean_lam_z',
        'gnorm_ae', 'gnorm_qc', 'gnorm_xi', 'gnorm_phase',
        'gnorm_power', 'gnorm_ck', 'gnorm_critic',
    )}

    best_reward   = None
    best_ep       = 0     # episode index of the current best (for eps-since-best)
    n_diverge     = 0     # cumulative divergence-signal count (for end summary)
    # carry-forward of the latest PPO-update diagnostics (so npz arrays stay
    # per-episode aligned even though diagnostics are computed every N episodes)
    last_diag = {k: 0.0 for k in (
        'kl_q', 'kl_ph', 'kl_pw', 'kl_ck', 'qp_penalty', 'expl_var', 'v', 'v_std',
        'lam_y', 'lam_z', 'gn_ae', 'gn_qc', 'gn_xi', 'gn_phase', 'gn_power',
        'gn_ck', 'gn_critic')}
    t_train_start = time.perf_counter()
    rng_ppo       = np.random.default_rng(args.seed + 77)   # for ep_buf shuffling
    rollout_buf   = []   # accumulates ep_buf across n_rollout_episodes episodes
    gnorm_ema     = {}   # EMA of per-net grad norms (for spike-based divergence detection)

    # ════════════════════════════════════════════════════════════════════════
    # Episode loop
    # ════════════════════════════════════════════════════════════════════════
    for ep in range(args.episodes):
        obs   = env.reset()
        K     = cfg.K          # fixed K throughout
        t0_ep = time.perf_counter()

        # ── Per-episode schedule ──────────────────────────────────────────────
        beta    = _compute_beta(ep, args.episodes)
        lr_frac = _compute_lr_frac(ep, args.episodes)
        actor.opt_ae.lr      = P.lr_actor_ae * lr_frac
        # ⚠ FIX 2026-06-13: classical-mode opt_qc IS the MLP actor optimizer; respect
        # lr_classical_actor (was silently overwritten by lr_actor_qc → r28≡r29 bug).
        _lr_qc_base = (getattr(P, 'lr_classical_actor', P.lr_actor_qc)
                       if getattr(args, 'actor_mode', 'quantum') == 'classical'
                       else P.lr_actor_qc)
        actor.opt_qc.lr      = _lr_qc_base * lr_frac
        actor.opt_xi.lr      = P.lr_actor_xi * lr_frac
        phase_net.opt.lr     = P.lr_phase    * lr_frac
        power_net.opt.lr     = P.lr_power    * lr_frac
        ck_net.opt.lr        = P.lr_ck       * lr_frac
        critic.opt.lr        = P.lr_critic   * lr_frac

        ep_reward = 0.0
        ep_L_actor = []
        ep_L_pg, ep_L_ae, ep_td = [], [], []
        ep_L_phase, ep_L_power, ep_L_ck = [], [], []
        ep_clip_q, ep_clip_ph, ep_clip_pw, ep_clip_ck = [], [], [], []
        # Divergence diagnostics (populated during PPO update)
        ep_gnorm = {'ae': [], 'qc': [], 'xi': [], 'phase': [], 'power': [], 'ck': [], 'critic': []}
        ep_glam  = []   # [Bước 0] per-mini-batch λ (encoding) grad vectors → frozen-λ diag
        ep_kl    = {'q': [], 'ph': [], 'pw': [], 'ck': []}
        ep_ent   = {'q': [], 'ph': [], 'pw': [], 'ck': []}
        ep_qp    = []   # QoS penalty per step (reward breakdown)
        # Behaviour diagnostics: grouping, power allocation, common-rate split
        ep_n_dir = []   # # users on direct link
        ep_irs_cnt = []  # (M,) # users per IRS
        ep_wc_frac = []; ep_wp_frac = []   # common / private power fraction of P_S
        ep_wp_top  = []  # fraction of private power held by the top-2 users (concentration)
        ep_ck_tot  = []; ep_ck_top = []; ep_ck_active = []  # common-rate total / top-share / #served
        ep_buf = []
        ep_feas      = 0
        ep_o_norm    = []
        ep_z_norm    = []
        ep_sum_rate  = []
        ep_qos_count     = []
        ep_irs_count     = []
        ep_blocked_count = []

        # ── Step loop ────────────────────────────────────────────────────────
        for step_idx in range(args.steps):
            is_last_step = (step_idx == args.steps - 1)

            # ════════════════════════════════════════════════════════════════
            # 1) Forward on the LIVE trajectory
            # ════════════════════════════════════════════════════════════════
            blocked = _compute_irs_favored(env)
            ep_blocked_count.append(int(np.sum(env.channels['su_blocked'])))
            # Routing-shaping target (Option 1): IRS-favored users → their best IRS.
            _route_tgt, _route_msk = _irs_favored_target(env)
            s_t     = actor.extract_state(obs, demand, blocked)
            phi, _, actor_info = actor.forward(s_t)
            z_t = actor_info['z_t']                              # (n_latent,) spatial latent
            # [--no-ae] sub-actors consume o_hat (quantum readout) instead of AE latent z_t
            rep = actor_info['o_hat'] if getattr(args, 'no_ae', False) else z_t

            # Counterfactual-assignment auxiliary (q-head credit fix): per-user R_cf
            # over link choices, by re-running downstream under φ'. Flag-gated (cost).
            _cf_rew = (_cf_assign_rewards(env, phase_net, power_net, ck_net, phi, rep,
                                          cfg, demand, float(cfg.lambda_D))
                       if getattr(args, 'counterfactual_assign', False) else None)

            # 0-based active IRS indices for PhaseMLP; 1-based ids for PowerMLP/rate
            active_irs     = _get_active_irs(phi)        # (G,) 0-based
            active_irs_ids = _get_active_irs_ids(phi)    # [gid, …] 1-based

            s_phase  = _build_phase_state(env.channels, phi, cfg, rep)  # (M, 3K+rep_dim)
            phase_idx, lp_old_ph, _ = phase_net.forward(s_phase, active_irs)
            phases_rad   = env.phase_model.index_to_phase(phase_idx)
            proposed_Phi = env.phase_model.build_phi(phases_rad)

            h_eff   = env.rate_computer.effective_channels_all(
                          phi, proposed_Phi, env.channels)
            s_power = _build_power_state(
                h_eff, None if getattr(args, 'power_clean_input', False) else rep,
                cfg if getattr(args, 'power_scale_feats', False) else None,
                getattr(args, 'power_mag_feats', None))  # (2K [+scale] [+mag] [+rep],)
            w_c_vec, w_p, _, power_action = power_net.forward(s_power, active_irs_ids)

            partial = env.rate_computer.compute_rates_partial(
                phi, proposed_Phi, env.channels, w_p, w_c_vec,
                active_irs_ids=active_irs_ids)
            s_ck = _build_ck_state(
                demand, partial['R_private'], partial['R_c_group'], phi, cfg)
            if getattr(args, 'closed_form_ck', False):
                # Solve C_k exactly instead of sampling it. Marking every group
                # None makes the whole Ck PPO path inert (CkMLP._group_terms
                # skips None groups) → zero log-prob, zero gradient, no update.
                C_k = _closed_form_ck(partial, cfg)
                ck_group_sel = {gid: None for gid in partial['groups']}
            else:
                C_k, _, ck_group_sel = ck_net.forward(
                    s_ck, phi, partial['R_c_group'])

            action = {
                'assignment': phi, 'phase_idx': phase_idx,
                'w_p': w_p, 'w_c_vec': w_c_vec, 'C_k': C_k,
            }
            obs2, reward, _, env_info = env.step(action)

            blocked2 = _compute_irs_favored(env)
            s_t_next = actor.extract_state(obs2, demand, blocked2)

            # ════════════════════════════════════════════════════════════════
            # 2) Collect transition for GAE + PPO
            # ════════════════════════════════════════════════════════════════
            a_t = _build_action_vec(phi, phase_idx, w_c_vec, active_irs_ids,
                                    w_p, C_k, cfg, env.n_phase_levels)
            V_t_collected = float(critic.forward(s_t))
            lp_old_q  = actor.compute_log_prob(s_t, phi, K)
            lp_old_pw = power_net.compute_log_prob(s_power, active_irs_ids, power_action)
            lp_old_ck = ck_net.compute_log_prob(s_ck, phi, K, ck_group_sel)
            ep_buf.append({
                's_t':             s_t.copy(),
                'a_t':             a_t.copy(),
                's_t_next':        s_t_next.copy(),
                'phi':             phi.copy(),
                'q_pi':            actor_info['pi'].copy(),   # Δ7: per-user routing probs (K,M+1)
                'route_tgt':       _route_tgt.copy(),         # Option 1: best IRS id per user (1-based)
                'route_msk':       _route_msk.copy(),         # Option 1: 1.0 where IRS-favored
                'cf_rew':          (_cf_rew.copy() if _cf_rew is not None else None),  # Opt2: R_cf (K,M+1)
                # oracle routing target for --assign-aux-weight (continuous CE teacher)
                'route_oracle':    (_oracle_assign_env(env).copy()
                                    if getattr(args, 'assign_aux_weight', 0.0) > 0.0 else None),
                # oracle phase target for --phase-aux-weight. n_sweeps=0 = the
                # closed-form per-element alignment only (0.08 ms/state vs 0.56
                # with ascent) — cheap enough to run every rollout step. Skipped
                # entirely once the anneal has driven the weight to 0, which is
                # where most of the anneal's wall-clock saving comes from.
                # [--phase-aux-sweeps] n_sweeps=0 was the default here and it is
                # measured HARMFUL: scored on the true channel over 120 states of
                # r275, the target it produces sits at J 1.7022 against the head's
                # own 1.7227 — the teacher was pulling the head toward something
                # WORSE than what the head already did, which is why weight 0.8
                # cost -0.126 J and why per-element agreement decayed to chance
                # (25.2% vs 25.0%) despite warm-up reaching 65.7%. One sweep gives
                # 1.7772 (+0.0545, 56% of the J-oracle's gain) for 0.56 ms/state.
                # NOT monotone -- 3 sweeps 1.7590, 8 sweeps 1.7565 -- because the
                # ascent maximises coherent gain, not J, so more of it drifts
                # further from the objective. Default stays 0 for reproducibility.
                # [--phase-aux-hspread] log spread of the effective channel. The
                # phase teacher is redistributed by this: the oracle-phase swap is
                # worth +0.583 on the states the policy loses and +0.045 on the
                # ones it wins, and the losers are the high-spread ones
                # (corr -0.79). Computed here because h_eff is already in hand.
                'log_hspread':     float(np.log(
                                       (np.abs(h_eff).max() + 1e-30) /
                                       (np.abs(h_eff).min() + 1e-30))),
                'phase_oracle':    (oracle_phase_idx(_po_est(env.channels), phi, cfg,
                                                     n_sweeps=int(getattr(args, 'phase_aux_sweeps', 0))).copy()
                                    if _aux_w(args, 'phase', ep, args.episodes) > 0.0 else None),
                's_phase':         s_phase.copy(),
                'phase_idx':       phase_idx.copy(),
                's_power':         s_power.copy(),
                'active_irs_ids':  list(active_irs_ids),
                's_ck':            s_ck.copy(),
                # Power/Ck actions are the SAMPLED ALLOCATION VECTORS (Dirichlet
                # policy, 2026-07-21) — not categorical indices. They must be
                # copied: the nets return fresh arrays but the buffer outlives
                # the step, and PPO re-scores exactly these vectors.
                'power_a_split':   np.asarray(power_action['split'],   float).copy(),
                'power_a_common':  np.asarray(power_action['common'],  float).copy(),
                'power_a_private': np.asarray(power_action['private'], float).copy(),
                'ck_group_sel':    {g: (None if v is None else np.asarray(v, float).copy())
                                    for g, v in ck_group_sel.items()},
                # teacher share for --ck-aux-weight (per group, on the simplex)
                'ck_aux_tgt':      _ck_aux_target(partial, cfg)
                                   if getattr(args, 'ck_aux_weight', 0.0) > 0.0 else None,
                # teacher allocation for --power-aux-weight (grid (β*,f*) per state)
                'power_aux_tgt':   (_power_aux_target(env.rate_computer, phi,
                                                      proposed_Phi, env.channels,
                                                      active_irs_ids, h_eff, cfg)
                                    if getattr(args, 'power_aux_weight', 0.0) > 0.0
                                    else None),
                'lp_old_q':        float(lp_old_q),
                'lp_old_ph':       float(lp_old_ph),
                'lp_old_pw':       float(lp_old_pw),
                'lp_old_ck':       float(lp_old_ck),
                'reward':          float(reward),
                'is_terminal':     bool(is_last_step),
                'V_t':             V_t_collected,
                # ── Tier-2 diag fields (read by RL/critic_diag.py) ────────
                'blocked_n':       int(np.sum(env.channels['su_blocked'])),
                # σ² is PER-USER (K,) since the per-user-noise change; the Tier-2
                # diag only tracks the ambient level, so log its mean.
                'sigma2':          float(np.mean(env_info.get('sigma2', np.nan))),
            })

            # ── EXP-2 IRS-routing bonus (default OFF) ─────────────────────────
            # Reward shaping to clean credit signal for IRS-routed users — fights F2
            # non-stationarity by giving assignment an explicit reinforcement signal
            # for routing through IRS (counter to the natural risk-avoidance of
            # high-variance IRS channels). Default 0.0 = OFF (no behaviour change).
            # Anneal: full bonus for first 20% of training, linear decay to 0 by 60%.
            # USE when EXP-3 verdict is (3) EXECUTION PROBLEM (oracle has room but
            # policy can't extract it).
            _irs_bonus_w = float(getattr(args, 'irs_bonus', 0.0))
            if _irs_bonus_w > 0.0:
                _bonus_frac = max(0.0, min(1.0,
                                  1.0 - (ep - 0.2 * args.episodes) / (0.4 * args.episodes)))
                _n_irs = int(np.sum(phi > 0))
                reward = reward + _irs_bonus_w * _bonus_frac * _n_irs

            ep_reward += reward
            ep_o_norm.append(float(np.linalg.norm(actor_info['o_hat'])))
            ep_z_norm.append(float(np.linalg.norm(actor_info['z_t'])))
            ep_sum_rate.append(float(env_info['sum_rate']))
            ep_qp.append(float(env_info['qp_penalty']))
            ep_qos_count.append(int(np.sum(env_info['R_tot'] >= cfg.D_k_bps_hz)))
            ep_irs_count.append(int(np.sum(phi > 0)))
            if env_info['feasible']:
                ep_feas += 1

            # ── Behaviour diagnostics (per step) ──────────────────────────────
            ep_n_dir.append(int(np.sum(phi == 0)))
            ep_irs_cnt.append(np.array([int(np.sum(phi == m + 1)) for m in range(cfg.M)]))
            _wc_t = float(np.sum(w_c_vec)); _wp_t = float(np.sum(w_p)); _tot = _wc_t + _wp_t + 1e-9
            ep_wc_frac.append(_wc_t / _tot); ep_wp_frac.append(_wp_t / _tot)
            _wp_sorted = np.sort(w_p)[::-1]
            ep_wp_top.append(float(_wp_sorted[:2].sum() / (_wp_t + 1e-9)))
            _ck = env_info.get('C_k')
            if _ck is not None:
                _ck = np.asarray(_ck, float); _ckt = float(_ck.sum())
                ep_ck_tot.append(_ckt)
                ep_ck_top.append(float(_ck.max() / (_ckt + 1e-9)) if _ckt > 1e-9 else 0.0)
                ep_ck_active.append(int(np.sum(_ck > 1e-6)))

            obs = obs2

        # ════════════════════════════════════════════════════════════════════
        # 3) GAE backward — Â_t = Σ (γλ)^l δ_{t+l},  δ_t = r_t + γV(s_{t+1}) − V(s_t)
        # ════════════════════════════════════════════════════════════════════
        # V(s_{i+1}) is the value stored at the next buffer entry. Episodes are
        # time-truncated (done is always False), so the final step bootstraps the
        # next-state value V(s_T') rather than treating it as a true terminal.
        # Advantages are normalised once per rollout (see PPO block), not here.
        gae = 0.0
        V_next_last = float(critic.forward(ep_buf[-1]['s_t_next'])) if ep_buf else 0.0
        for i in reversed(range(len(ep_buf))):
            t      = ep_buf[i]
            V_next = ep_buf[i + 1]['V_t'] if i + 1 < len(ep_buf) else V_next_last
            delta  = t['reward'] + args.gamma * V_next - t['V_t']
            gae    = delta + args.gamma * P.gae_lambda * gae
            t['gae_adv'] = float(gae)
            t['ret']     = float(gae + t['V_t'])   # GAE return target (for explained-variance)

        rollout_buf.extend(ep_buf)

        # ════════════════════════════════════════════════════════════════════
        # 5) PPO-clip — runs every n_rollout_episodes episodes
        # ════════════════════════════════════════════════════════════════════
        if (ep + 1) % P.n_rollout_episodes == 0 or ep == args.episodes - 1:
          # Per-rollout advantage normalisation (zero-mean, unit-std over all
          # collected transitions), then CLIP to ±adv_clip. Clipping bounds the
          # gradient from extreme-advantage outliers (e.g. high spawn-variance at
          # large R_LoS → reward swings → huge AE/QC grad spikes). At ±5 it only
          # touches ~5σ outliers, so stable runs are unaffected.
          advs     = np.array([t['gae_adv'] for t in rollout_buf])
          adv_mean = float(advs.mean())
          adv_std  = float(advs.std()) + 1e-8
          _norm    = (advs - adv_mean) / adv_std
          _aclip   = getattr(P, 'adv_clip', 0.0) or 0.0
          adv_clip_frac = float(np.mean(np.abs(_norm) > _aclip)) if _aclip > 0 else 0.0
          if _aclip > 0:
              _norm = np.clip(_norm, -_aclip, _aclip)
          for t, _a in zip(rollout_buf, _norm):
              t['gae_adv'] = float(_a)

          ep_indices = np.arange(len(rollout_buf))
          # PopArt: update value-scale stats (μ,σ) ONCE per rollout from this
          # batch's GAE-returns, and PRESERVE the critic's outputs (rescale last
          # layer). Must run BEFORE the PPO epochs so all minibatch regressions
          # use the same normalised target. (no-op when popart disabled.)
          # [T5 diag] POP correctness: V(s) for fixed states must be UNCHANGED across
          # the stats update (POP is algebraically exact → ΔV≈0). A large ΔV ⇒ bug.
          _pa_on = getattr(P, 'popart_enabled', False)
          _pa_s  = [t['s_t'] for t in rollout_buf[:64]]
          _v_pre = np.array([critic.forward(s) for s in _pa_s]) if _pa_on else None
          critic.update_popart_stats(np.array([t['ret'] for t in rollout_buf]))
          pop_dv = (float(np.max(np.abs(
                       np.array([critic.forward(s) for s in _pa_s]) - _v_pre)))
                    if _v_pre is not None else float('nan'))
          # [Tier-2 diag] per-PPO-epoch critic pre-clip grad-norm tracker.
          # Detects whether the critic over-trains across the 6 PPO epochs
          # (epoch-6 grad ≫ epoch-1 grad ⇒ critic is "chasing" its own targets).
          ep_critic_gn_preclip_by_epoch = [[] for _ in range(P.ppo_epochs)]
          # [factored PowerMLP diag] per-axis entropy & split policy bias —
          # detects collapsed axes (low H/Hmax) or runaway uniformisation (high H/Hmax)
          # in the new split/common/private factorisation (Case2 result_10 fix).
          ep_pw_axis_stats: dict = {}
          for _epoch in range(P.ppo_epochs):
            rng_ppo.shuffle(ep_indices)
            for start in range(0, len(rollout_buf), P.batch_size):
                mini = [rollout_buf[i] for i in ep_indices[start : start + P.batch_size]]
                B = len(mini)
                if B == 0:
                    continue

                # Loss accumulators (for logging)
                L_pg_b = L_ae_b = L_ent_b = L_phase_b = L_power_b = L_ck_b = 0.0
                td_b   = 0.0

                # Gradient outputs (assigned by batch methods below)
                g_phase_acc = g_power_acc = g_ck_acc = None

                # ── Phase 1: batch MLP actor gradients (vectorised — no Python loop) ─
                adv_arr     = np.array([t['gae_adv']   for t in mini])
                lp_old_ph_b = np.array([t['lp_old_ph'] for t in mini])
                lp_old_pw_b = np.array([t['lp_old_pw'] for t in mini])
                lp_old_ck_b = np.array([t['lp_old_ck'] for t in mini])

                lp_ph_b = phase_net.compute_log_prob_batch(mini)
                lp_pw_b = power_net.compute_log_prob_batch(mini)
                lp_ck_b = ck_net.compute_log_prob_batch(mini, K)

                # Approx KL(old‖new) per sub-actor: mean(exp(Δ) − 1 − Δ), Δ=lp_new−lp_old
                _d_ph = lp_ph_b - lp_old_ph_b
                _d_pw = lp_pw_b - lp_old_pw_b
                _d_ck = lp_ck_b - lp_old_ck_b
                kl_ph_b = float(np.mean(np.exp(_d_ph) - 1.0 - _d_ph))
                kl_pw_b = float(np.mean(np.exp(_d_pw) - 1.0 - _d_pw))
                kl_ck_b = float(np.mean(np.exp(_d_ck) - 1.0 - _d_ck))

                eff_ph_b, l_ph_arr, clip_ph_b = _ppo_eff_adv_batch(
                    lp_ph_b, lp_old_ph_b, adv_arr, P.ppo_epsilon)
                eff_pw_b, l_pw_arr, clip_pw_b = _ppo_eff_adv_batch(
                    lp_pw_b, lp_old_pw_b, adv_arr, P.ppo_epsilon)
                eff_ck_b, l_ck_arr, clip_ck_b = _ppo_eff_adv_batch(
                    lp_ck_b, lp_old_ck_b, adv_arr, P.ppo_epsilon)

                L_phase_b = float(l_ph_arr.mean())
                L_power_b = float(l_pw_arr.mean())
                L_ck_b    = float(l_ck_arr.mean())

                _, L_ent_ph, g_phase_acc = phase_net.compute_grads_batch(
                    mini, eff_ph_b, beta,
                    counterfactual=getattr(args, 'phase_counterfactual', False),
                    aux_w=_aux_w(args, 'phase', ep, args.episodes))
                _, L_ent_pw, g_power_acc, pw_axis = power_net.compute_grads_batch(
                    mini, eff_pw_b, beta,
                    beta_entropy_private_extra=_beta_pwr_priv(args),
                    aux_w=float(getattr(args, 'power_aux_weight', 0.0) or 0.0))
                for _k, _v in pw_axis.items():
                    ep_pw_axis_stats.setdefault(_k, []).append(float(_v))
                _, L_ent_ck, g_ck_acc    = ck_net.compute_grads_batch(
                    mini, K, eff_ck_b, beta,
                    aux_w=float(getattr(args, 'ck_aux_weight', 0.0) or 0.0))
                L_ent_b += L_ent_ph + L_ent_pw + L_ent_ck

                # ── Critic batch gradient (vectorised — no Python loop) ────
                # V(s) critic: action list is unused (d_action=0).
                # Target = GAE λ-return t['ret'] (= Â_t + V_t), KHÔNG phải TD(0) target.
                # [Case2 result_3 fix B-2]: TD(0) (r+γV_target(s')) là target 1-bước phương
                # sai CAO → grad V kẹt lớn (700-1000), explVar dao động 0.14-0.64, QoS kẹt.
                # GAE-return (λ=0.9, ~7-step avg) ít nhiễu hơn + align với explVar (đo vs ret)
                # + là value-target PPO CHUẨN → grad giảm/settle, explVar ổn.
                td_b, g_critic_acc = critic.compute_grads_batch(
                    [t['s_t']       for t in mini],
                    [None           for _ in mini],
                    np.array([t['ret'] for t in mini]))

                # ── Phase 2: fused quantum log-prob + gradient (1 QC forward) ─
                _shape_coef = float(getattr(args, 'routing_shape_coef', 0.0)) \
                    if getattr(args, 'routing_shape', False) else 0.0
                _cf_coef = float(getattr(args, 'cf_coef', 0.0)) \
                    if getattr(args, 'counterfactual_assign', False) else 0.0
                # --cf-anneal-end: linear decay → 0 at frac·episodes (divergence fix, đĩa r25)
                _cf_frac = float(getattr(args, 'cf_anneal_end', 0.0) or 0.0)
                if _cf_coef > 0.0 and _cf_frac > 0.0:
                    _cf_coef *= max(0.0, 1.0 - ep / (_cf_frac * args.episodes))
                l_q_arr, L_ae_b, L_ent_q, g_ae_acc, g_qc_acc, g_xi_acc, clip_q_b, kl_q_b = \
                    actor.compute_logprobs_grads_batch(
                        [t['s_t']      for t in mini],
                        [t['phi']      for t in mini],
                        np.array([t['lp_old_q'] for t in mini]),
                        np.array([t['gae_adv']  for t in mini]),
                        P.ppo_epsilon, P.ae_weight, beta,
                        routing_target_b=(np.array([t['route_tgt'] for t in mini])
                                          if _shape_coef > 0.0 else None),
                        shape_mask_b=(np.array([t['route_msk'] for t in mini])
                                      if _shape_coef > 0.0 else None),
                        shape_coef=_shape_coef,
                        cf_rew_b=(np.array([t['cf_rew'] for t in mini])
                                  if _cf_coef > 0.0 else None),
                        cf_coef=_cf_coef)
                L_pg_b  = float(l_q_arr.mean())
                L_ent_b += L_ent_q

                # [Bước 0 diag] capture UNCLIPPED λ (encoding-scale) gradient per
                # mini-batch to diagnose the frozen-λ issue [E]. We want to know if
                # the λ gradient is zero-mean sign-cancelling noise (→ Adam stalls →
                # λ frozen) vs a tiny-but-consistent signal. Store the concatenated
                # lam_y/lam_z grad vector; stats computed in the diag panel below.
                ep_glam.append(np.concatenate([
                    np.ravel(g_qc_acc['lam_y']), np.ravel(g_qc_acc['lam_z'])]))

                # Clip non-quantum gradients
                # All batch gradient methods already return B-averaged grads internally

                gn_ae = _clip_grad_norm(g_ae_acc,     P.grad_clip_ae)
                gn_qc = _clip_grad_norm(g_qc_acc,     P.grad_clip_actor)
                gn_xi = _clip_grad_norm(g_xi_acc,     P.grad_clip_actor)
                gn_ph = _clip_grad_norm(g_phase_acc,  P.grad_clip_actor)
                gn_pw = _clip_grad_norm(g_power_acc,  P.grad_clip_actor)
                gn_ck = _clip_grad_norm(g_ck_acc,     P.grad_clip_actor)
                gn_cr = _clip_grad_norm(g_critic_acc, P.grad_clip_critic)
                # [Tier-2 diag] track pre-clip critic grad per PPO epoch
                ep_critic_gn_preclip_by_epoch[_epoch].append(gn_cr)

                # Apply
                actor.apply_grads(g_ae_acc, g_qc_acc, g_xi_acc)

                # ── ASSIGNMENT AUX TEACHER (--assign-aux-weight) ──────────────
                # Continuous CE pull toward the oracle routing (blocked→its-IRS),
                # applied at EVERY update alongside PPO. Targets the measured
                # collapse mechanism: routing IDENTITY drifts (r166 ep2000→10000:
                # mis 0.5→1.8%, blocked→dead-direct 1.1→5.1%) even though the
                # COUNT stays right — F2 assignment-credit failing at scale
                # ((M+1)^K). The oracle routing hits QoS 99.7%, so a strong-enough
                # teacher can pull routing toward it, i.e. bypass the hard credit
                # problem instead of learning it. Unlike a periodic re-warmup this
                # never shocks the critic (no burst). Reuses exactly the warm-up
                # recipe: lp_old set to the current logprob so the PPO ratio is 1
                # (no clipping) → pure -w·log π(oracle|s).
                _aw_q = float(getattr(args, 'assign_aux_weight', 0.0) or 0.0)
                if _aw_q > 0.0:
                    _s_aux   = [t['s_t'] for t in mini]
                    _tgt_aux = [t['route_oracle'] for t in mini]
                    _lp0_aux = actor.compute_logprobs_batch(_s_aux, _tgt_aux)
                    _ax = actor.compute_logprobs_grads_batch(
                        _s_aux, _tgt_aux, _lp0_aux,
                        np.full(len(mini), _aw_q), P.ppo_epsilon, 0.0, 0.0)
                    actor.apply_grads(_ax[3], _ax[4], _ax[5])
                if not getattr(args, 'freeze_phase', False):   # EXP-1: freeze-phase skips this
                    phase_net.apply_grads(g_phase_acc)
                power_net.apply_grads(g_power_acc)
                if not getattr(args, 'closed_form_ck', False):
                    ck_net.apply_grads(g_ck_acc)   # C_k is solved, not learned
                critic.apply_grads(g_critic_acc)

                L_actor_b = (L_pg_b + P.ae_weight * L_ae_b
                             + L_phase_b + L_power_b + L_ck_b + L_ent_b)
                ep_L_actor.append(L_actor_b)
                ep_L_pg.append(L_pg_b)
                ep_L_ae.append(L_ae_b)
                ep_td.append(td_b)
                ep_L_phase.append(L_phase_b)
                ep_L_power.append(L_power_b)
                ep_L_ck.append(L_ck_b)
                ep_clip_q.append(clip_q_b)
                ep_clip_ph.append(clip_ph_b)
                ep_clip_pw.append(clip_pw_b)
                ep_clip_ck.append(clip_ck_b)
                # Divergence diagnostics
                ep_gnorm['ae'].append(gn_ae);  ep_gnorm['qc'].append(gn_qc)
                ep_gnorm['xi'].append(gn_xi);  ep_gnorm['phase'].append(gn_ph)
                ep_gnorm['power'].append(gn_pw);  ep_gnorm['ck'].append(gn_ck)
                ep_gnorm['critic'].append(gn_cr)
                ep_kl['q'].append(kl_q_b);   ep_kl['ph'].append(kl_ph_b)
                ep_kl['pw'].append(kl_pw_b); ep_kl['ck'].append(kl_ck_b)
                _bi = max(beta, 1e-8)
                ep_ent['q'].append(-L_ent_q / _bi);   ep_ent['ph'].append(-L_ent_ph / _bi)
                ep_ent['pw'].append(-L_ent_pw / _bi); ep_ent['ck'].append(-L_ent_ck / _bi)

          # ── Divergence diagnostics panel (once per PPO update) ────────────────
          _retb = np.array([t['ret'] for t in rollout_buf])
          _Vb   = np.array([t['V_t'] for t in rollout_buf])
          _gx   = lambda x: float(np.max(x)) if x else float('nan')   # max → spike detector
          _gm   = lambda x: float(np.mean(x)) if x else float('nan')
          diag = {
              'gnorm':    {k: _gx(v) for k, v in ep_gnorm.items()},
              'kl':       {k: _gm(v) for k, v in ep_kl.items()},
              'ent':      {k: _gm(v) for k, v in ep_ent.items()},
              'v_mean':   float(_Vb.mean()), 'v_std': float(_Vb.std()),
              'expl_var': _explained_variance(_retb, _Vb),
              'lam_y':    float(np.max(np.abs(actor.lam_y))),
              'lam_z':    float(np.max(np.abs(actor.lam_z))),
              'sum_rate': float(np.mean(ep_sum_rate)), 'qp': float(np.mean(ep_qp)),
          }
          diag['scalars'] = {'L_pg': _gm(ep_L_pg), 'TD': _gm(ep_td),
                             'V_mean': diag['v_mean'], 'expl_var': diag['expl_var'],
                             **{f"gn_{k}": v for k, v in diag['gnorm'].items()}}
          _signals = _divergence_signals(diag, gnorm_ema)
          _print_diag_panel(ep, diag, _signals)      # screen + file (concise)
          if _signals:
              n_diverge += 1

          # ── Verbose diagnostics → log FILE ONLY (screen stays concise) ───────
          # (1) entropy-domination ratio β·H/|L_pg| per actor — the direct
          #     predictor of the result_3 collapse (entropy overwhelming the
          #     policy gradient → policy drifts to random → QoS collapse).
          _pg  = lambda xs: max(abs(_gm(xs)), 1e-6)
          _ed  = {a: beta * diag['ent'][a] / _pg(xs) for a, xs in
                  (('q', ep_L_pg), ('ph', ep_L_phase), ('pw', ep_L_power), ('ck', ep_L_ck))}
          _edf = '  ⚠ ENTROPY-DOMINATED (↓β)' if max(_ed.values()) > 1.0 else ''
          _acf = '  ⚠ HIGH-VARIANCE' if adv_clip_frac > 0.02 else ''
          # ⚠ pw/ck use a DIRICHLET entropy, which is differential and routinely
          # NEGATIVE once the head concentrates — so pw/ck here can be negative
          # and that is normal, not a fault. Only q/ph are categorical entropies
          # bounded ≥0; the ENTROPY-DOMINATED check keys on >1.0 and is unaffected.
          _flog(f"  · ent/pg (β·H/|L_pg|): q={_ed['q']:.2f} ph={_ed['ph']:.2f} "
                f"pw={_ed['pw']:.2f} ck={_ed['ck']:.2f}{_edf}  │ adv-clip {adv_clip_frac*100:.1f}%{_acf}")
          # [Dirichlet PowerMLP] largest MEAN share on each axis, max(α)/Σα.
          # flat = 1/n (split 50%, private 1/K), fully concentrated = 100%.
          # Rising private% = the head is learning to concentrate power — which
          # is what the Pareto measurement says it should do. NOT comparable to
          # the pre-2026-07-21 categorical H/log(n) figures in older logs.
          if ep_pw_axis_stats:
              _ax = {k: float(np.mean(v)) for k, v in ep_pw_axis_stats.items()}
              _flog(f"  · pw axis (max share): split={_ax['share_split_max']*100:.0f}% "
                    f"common={_ax['share_common_max']*100:.0f}% "
                    f"private={_ax['share_private_max']*100:.0f}% "
                    f"(flat={100.0/max(cfg.K,1):.0f}%)  │ "
                    f"E[common]={_ax['pi_split_common']*100:.0f}%")
          # [Bước 0] frozen-λ diagnostic: across this update's mini-batch λ grads,
          #   mag = mean |g| (typical push size);  net = mean_c |mean_s g| (directional
          #   component, per-component signed-mean then abs-averaged);  r = net/mag.
          #   r≈1 → consistent direction (λ would move);  r≈0 → zero-mean sign-cancelling
          #   noise → Adam's 1st moment ≈0 → λ FROZEN [E]. Compare mag vs gn_qc too.
          if ep_glam:
              _G = np.stack(ep_glam)                       # (n_mb, 2*nq)
              _mag = float(np.mean(np.abs(_G)))
              _net = float(np.mean(np.abs(_G.mean(axis=0))))
              _r   = _net / (_mag + 1e-12)
              _flog(f"  · λgrad: mag={_mag:.2e} net={_net:.2e} r={_r:.2f}"
                    f"  ({'consistent→should move' if _r > 0.5 else 'zero-mean noise→frozen [E]'})")
          if getattr(args, 'counterfactual_assign', False):
              _cfc = float(getattr(args, 'cf_coef', 0.0))
              _cff = float(getattr(args, 'cf_anneal_end', 0.0) or 0.0)
              if _cff > 0.0:
                  _cfc *= max(0.0, 1.0 - ep / (_cff * args.episodes))
              _flog(f"  · cf: coef_eff={_cfc:.3f}"
                    + (f" (anneal→0 @ep{int(_cff * args.episodes)})" if _cff > 0.0
                       else " (no anneal)"))
          # (2) rolling-mean trend (reward/QoS/R_tot) + episodes-since-best —
          #     reveals slow drift hidden by per-episode noise.
          _rw = hist['episode_reward']; _w = min(50, len(_rw))
          if _w > 0:
              _flog(f"  · rolling-{_w}: reward μ={float(np.mean(_rw[-_w:])):8.1f}  "
                    f"QoS μ={float(np.mean(hist['mean_qos_rate'][-_w:]))*100:4.0f}%  "
                    f"Rtot μ={float(np.mean(hist['mean_sum_rate'][-_w:])):.2f}  │ "
                    f"best={best_reward:.1f} ({(ep+1)-best_ep} ep ago)")
          # ── Lagrangian λ_D dual ascent (constrained-RL: max R_tot s.t. QoS ≥ target) ──
          #   λ_D = Lagrange multiplier on the QoS constraint, updated ONCE per rollout from
          #   the SMOOTHED rolling QoS so the ramp is GENTLE — a fast λ bump cascades into
          #   [K] LAMBDA-BUMP-COLLAPSE (Ck softmax collapse). dual ascent:
          #     viol = qos_target − QoS_rolling   (>0 under target → tighten λ_D; PopArt re-scales)
          #     λ_D ← clip(λ_D + lambda_lr·viol, λ_min, λ_max)
          if getattr(args, 'lagrangian', False) and _w > 0:
              _qos_roll = float(np.mean(hist['mean_qos_rate'][-_w:]))
              _viol     = args.qos_target - _qos_roll
              _lam_old  = cfg.lambda_D
              cfg.lambda_D = float(np.clip(_lam_old + args.lambda_lr * _viol,
                                           args.lambda_min, args.lambda_max))
              _flog(f"  · Lagrangian: QoS_roll={_qos_roll*100:4.1f}% "
                    f"target={args.qos_target*100:.0f}% viol={_viol:+.3f} → "
                    f"λ_D {_lam_old:.3f}→{cfg.lambda_D:.3f} "
                    f"[{args.lambda_min:.1f},{args.lambda_max:.1f}]")
          # (3) Behaviour: grouping / power allocation / common-rate split (this episode).
          #     Reveals HOW the agent acts — e.g. power concentrated on few strong users
          #     (wp-top2 high) = abuse; how common rate is shared; group balance.
          _gm2     = lambda xs: float(np.mean(xs)) if xs else 0.0
          _irs_mean = np.mean(np.array(ep_irs_cnt), axis=0) if ep_irs_cnt else np.zeros(cfg.M)
          _irs_str  = "[" + " ".join(f"{x:.1f}" for x in _irs_mean) + "]"
          _flog(f"  · behaviour: grp dir={_gm2(ep_n_dir):.1f} irs={_irs_str}  │ "
                f"pwr wc={_gm2(ep_wc_frac)*100:.0f}% wp={_gm2(ep_wp_frac)*100:.0f}% "
                f"(wp-top2={_gm2(ep_wp_top)*100:.0f}%)  │ "
                f"Ck tot={_gm2(ep_ck_tot):.2f} top={_gm2(ep_ck_top)*100:.0f}% on {_gm2(ep_ck_active):.1f}/{cfg.K}")
          # Update grad-norm EMA AFTER signalling (so a spike is judged vs history)
          for _net, _gn in diag['gnorm'].items():
              if np.isfinite(_gn):
                  gnorm_ema[_net] = _gn if _net not in gnorm_ema else 0.8 * gnorm_ema[_net] + 0.2 * _gn
          # Carry-forward diagnostics for per-episode npz persistence
          last_diag.update(
              kl_q=diag['kl']['q'], kl_ph=diag['kl']['ph'],
              kl_pw=diag['kl']['pw'], kl_ck=diag['kl']['ck'],
              qp_penalty=diag['qp'], expl_var=diag['expl_var'],
              v=diag['v_mean'], v_std=diag['v_std'],
              lam_y=diag['lam_y'], lam_z=diag['lam_z'],
              gn_ae=diag['gnorm']['ae'], gn_qc=diag['gnorm']['qc'],
              gn_xi=diag['gnorm']['xi'], gn_phase=diag['gnorm']['phase'],
              gn_power=diag['gnorm']['power'], gn_ck=diag['gnorm']['ck'],
              gn_critic=diag['gnorm']['critic'])

          # ── Tier-2 critic diagnostic (read-only, ~50ms / PPO update) ─────────
          try:
              crit_diag = compute_critic_diag(
                  rollout_buf, critic,
                  ep_critic_gn_preclip_by_epoch,
                  P.grad_clip_critic, args.gamma, env=env,
                  pop_dv=pop_dv)
              for line in format_critic_diag_lines(ep, crit_diag):
                  _flog(line)
              write_critic_diag_jsonl(
                  os.path.join(run_dir, 'critic_diag.jsonl'),
                  ep, crit_diag)
          except Exception as _exc:  # noqa — never let diag kill training
              _flog(f"  ┄ crit-diag[{ep}]  ERROR: {_exc}")

          rollout_buf = []   # clear after PPO update

        # ── Episode stats ─────────────────────────────────────────────────────
        ep_time      = time.perf_counter() - t0_ep
        feas_rate     = ep_feas / args.steps
        mean_sum_rate = float(np.mean(ep_sum_rate))
        mean_qos_n    = float(np.mean(ep_qos_count))          # avg # users meeting D_k
        mean_qos_rate = mean_qos_n / K                        # fraction (for hist/plots)
        mean_irs_n        = float(np.mean(ep_irs_count))      # avg # users on IRS
        mean_irs_rate     = mean_irs_n / K                   # fraction (for hist/plots)
        mean_blocked_n    = float(np.mean(ep_blocked_count)) # avg # users blocked by buildings
        mean_per_user_rate = mean_sum_rate / K               # avg rate per user

        # Loss means — guard against warmup episodes where no update happened
        _mean = lambda xs: float(np.mean(xs)) if len(xs) > 0 else 0.0

        hist['episode_reward'].append(ep_reward)
        hist['mean_L_actor'].append(_mean(ep_L_actor))
        hist['mean_L_pg'].append(_mean(ep_L_pg))
        hist['mean_L_ae'].append(_mean(ep_L_ae))
        hist['mean_td_loss'].append(_mean(ep_td))
        hist['feasibility_rate'].append(feas_rate)
        hist['mean_o_hat_norm'].append(float(np.mean(ep_o_norm)))
        hist['mean_z_norm'].append(float(np.mean(ep_z_norm)))
        hist['mean_sum_rate'].append(mean_sum_rate)
        hist['mean_per_user_rate'].append(mean_per_user_rate)
        hist['mean_qos_rate'].append(mean_qos_rate)
        hist['mean_irs_rate'].append(mean_irs_rate)
        hist['mean_blocked_rate'].append(mean_blocked_n / K)
        hist['mean_L_phase'].append(_mean(ep_L_phase))
        hist['mean_L_power'].append(_mean(ep_L_power))
        hist['mean_L_ck'].append(_mean(ep_L_ck))
        hist['mean_clip_q'].append(_mean(ep_clip_q))
        hist['mean_clip_phase'].append(_mean(ep_clip_ph))
        hist['mean_clip_power'].append(_mean(ep_clip_pw))
        hist['mean_clip_ck'].append(_mean(ep_clip_ck))
        # Carry-forward PPO-update diagnostics (per-episode aligned for npz)
        hist['mean_kl_q'].append(last_diag['kl_q']);   hist['mean_kl_ph'].append(last_diag['kl_ph'])
        hist['mean_kl_pw'].append(last_diag['kl_pw']); hist['mean_kl_ck'].append(last_diag['kl_ck'])
        hist['mean_qp_penalty'].append(last_diag['qp_penalty'])
        hist['mean_expl_var'].append(last_diag['expl_var'])
        hist['mean_v'].append(last_diag['v']);         hist['mean_v_std'].append(last_diag['v_std'])
        hist['mean_lam_y'].append(last_diag['lam_y']); hist['mean_lam_z'].append(last_diag['lam_z'])
        hist['gnorm_ae'].append(last_diag['gn_ae']);   hist['gnorm_qc'].append(last_diag['gn_qc'])
        hist['gnorm_xi'].append(last_diag['gn_xi']);   hist['gnorm_phase'].append(last_diag['gn_phase'])
        hist['gnorm_power'].append(last_diag['gn_power']); hist['gnorm_ck'].append(last_diag['gn_ck'])
        hist['gnorm_critic'].append(last_diag['gn_critic'])

        is_new_best = (best_reward is None) or (ep_reward > best_reward)
        best_reward = ep_reward if best_reward is None else max(best_reward, ep_reward)
        if is_new_best:
            best_ep = ep + 1
        best_str    = f"{best_reward:>10.3f}"

        qos_str = f"{mean_qos_n:.1f}/{K}"
        irs_str = f"{mean_irs_n:.1f}/{K}"
        blk_str = f"{mean_blocked_n:.1f}/{K}"
        flag    = "*" if is_new_best else " "
        # A PPO update ran this episode iff loss lists were populated. On rollout
        # episodes (no update) show "—" for every loss column (L_pg/L_ae/L_phs/
        # L_pw/L_ck/TD), consistent with how L_ae already displays.
        _upd = len(ep_L_pg) > 0
        _lf  = lambda xs, w, p: (f"{_mean(xs):>{w}.{p}f}" if _upd else f"{'—':>{w}}")
        _L_ae_str = f"{_mean(ep_L_ae):9.2e}" if _upd else f"{'—':>9}"
        print(f"{flag} {ep+1:>5}  {ep_reward:>9.3f}  {best_str}  "
              f"{_lf(ep_L_pg,8,3)}  {_L_ae_str}  "
              f"{_lf(ep_L_phase,7,3)}  {_lf(ep_L_power,7,3)}  {_lf(ep_L_ck,7,3)}  "
              f"{_lf(ep_td,8,4)}  "
              f"{mean_sum_rate:>8.3f}  {mean_per_user_rate:>7.3f}  "
              f"{qos_str:>7}  {irs_str:>7}  {blk_str:>7}  "
              f"{ep_time:>5.1f}s")

        # ── Save best agent (overwrite) whenever reward improves ──────────────
        if is_new_best:
            best_dir = os.path.join(run_dir, 'best')
            _save_agents(best_dir, actor, phase_net, power_net, ck_net, cfg, critic)

        # ── Periodic checkpoint every checkpoint_interval episodes ────────────
        ci = getattr(P, 'checkpoint_interval', 500)
        if ci > 0 and (ep + 1) % ci == 0:
            ckpt_dir = os.path.join(run_dir, 'checkpoints', f'ep_{ep+1:05d}')
            _save_agents(ckpt_dir, actor, phase_net, power_net, ck_net, cfg, critic)
            print(f"  [ckpt ep {ep+1}] → {os.path.relpath(ckpt_dir)}"
                  f"  (best reward so far: {best_reward:.3f})")

    # ── Total training time ───────────────────────────────────────────────────
    t_total  = time.perf_counter() - t_train_start
    final_ma = float(np.mean(hist['episode_reward'][-10:]))
    print(f"\n  {'─'*W}")
    print(f"  Training complete in {t_total/60:.1f} min  "
          f"({t_total/args.episodes:.1f} s/ep)")
    print(f"  Final 10-ep avg reward   : {final_ma:.4f}")
    print(f"  Final feasibility rate   : "
          f"{np.mean(hist['feasibility_rate'][-10:])*100:.1f}%")
    print(f"  Final avg Σ R_tot        : "
          f"{np.mean(hist['mean_sum_rate'][-10:]):.4f} bps/Hz"
          f"  ({np.mean(hist['mean_per_user_rate'][-10:]):.4f} bps/Hz/user)")
    print(f"  Final avg QoS fraction   : "
          f"{np.mean(hist['mean_qos_rate'][-10:])*100:.1f}%")

    # End-of-run analysis + recommendations for the next run
    _analysis_summary(hist, n_diverge, args)

    # ════════════════════════════════════════════════════════════════════════
    # Save results  (hyperparameters.json already written before training)
    # ════════════════════════════════════════════════════════════════════════
    np.savez(os.path.join(run_dir, 'metrics.npz'), **hist)
    _save_agents(run_dir, actor, phase_net, power_net, ck_net, cfg, critic)
    saved_plots = _save_plots(run_dir, hist, args)

    print(f"\n{'═'*W}")
    print(f"  Results  →  {os.path.abspath(run_dir)}")
    print(f"{'─'*W}")
    print(f"  hyperparameters.json   — all params for this run")
    print(f"  metrics.npz            — raw episode arrays (reload with np.load)")
    print(f"  training_log.txt       — full CLI output from this run")
    print(f"  agents/                — final actor weights (load with infer.py)")
    print(f"  best/                  — best-reward snapshot (overwrites on improvement)")
    ci = getattr(P, 'checkpoint_interval', 500)
    if ci > 0:
        n_ckpts = args.episodes // ci
        print(f"  checkpoints/ep_XXXXX/  — periodic snapshots every {ci} ep "
              f"({n_ckpts} total)")
    if saved_plots:
        for p in saved_plots:
            print(f"  {os.path.basename(p):<26} — performance plot")
    elif not HAVE_MPL or args.no_plots:
        print(f"  (plots skipped — pass --no-plots=false or install matplotlib)")
    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Quantum AC agent for IRS-assisted RSMA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--episodes', type=int,   default=P.n_episodes,
                        help='Number of training episodes')
    parser.add_argument('--steps',    type=int,   default=P.n_steps_per_ep,
                        help='Environment steps per episode')
    parser.add_argument('--gamma',    type=float, default=P.gamma,
                        help='TD discount factor γ')
    parser.add_argument('--seed',     type=int,   default=P.seed_default,
                        help='Global random seed')
    parser.add_argument('--no-plots',    action='store_true',
                        help='Skip matplotlib plot generation')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='Skip AE hot-start pre-training (fast pipeline inspection)')
    parser.add_argument('--no-ae-keep-pretrain', dest='no_ae_keep_pretrain', action='store_true',
                        help='With --no-ae, KEEP the AE encoder pre-training (decouple no-ae from '
                             'no-pretrain). The pretrained encoder bootstraps the representation so '
                             'routing converges under VANILLA PPO (no --oracle-warmup needed) — '
                             'recovering r39-style ADAPTIVE drop-for-R_tot that oracle-warmup '
                             'suppresses (it commits to serve-everyone). Head still reads o_hat (no-ae); '
                             'decoder discarded at inference so param-eff unchanged.')
    parser.add_argument('--full-phase-warmup', dest='full_phase_warmup', action='store_true',
                        help='Disable phase-warmup early-stop → run ALL --phase-warmup-epochs '
                             '(test whether more CE epochs unlock a higher phase-match ceiling).')
    parser.add_argument('--phase-warmup-target', dest='phase_warmup_target', type=float, default=None,
                        help='Train phase-warmup until per-IRS match ≥ TARGET%% (else run the full '
                             '--phase-warmup-epochs cap). Disables plateau early-stop. Use a HIGH '
                             'epoch cap (phase-warmup is classical/cheap) to give the actor max '
                             'chance to reach DNN-level phase: hits target → speed-limited (capable); '
                             'caps below → ceiling-limited representation. e.g. --phase-warmup-target 95')
    parser.add_argument('--readout', dest='readout', default=None,
                        choices=['r1', 'single-z', 'nn-zz', 'full-zz'],
                        help='ABLATION (fresh runs only): VQC readout observable design. '
                             'r1=structured per-action (params default); single-z=Z-only (no ZZ); '
                             'nn-zz=Z+nearest-neighbor ZZ (generic); full-zz=Z+cross-block ZZ (every user-IRS pair plus user-block nearest neighbours; NOT all pairwise -- see params._b1_zz_pairs). '
                             'Only the readout differs — every other hyperparameter identical. '
                             'Ignored on --resume (saved actor config wins).')
    parser.add_argument('--no-ae', dest='no_ae', action='store_true',
                        help='ABLATION: drop the z_t classical bypass [B]. The assignment head '
                             'and sub-actors (Phase/Power) consume o_hat (quantum readout) instead '
                             'of the AE latent z_t. Auto-sets ae_weight=0 + --no-pretrain so the '
                             'encoder/λ receive gradient ONLY through the quantum path. Tests '
                             'whether λ un-freezes once the AE no longer represents for it.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to an agents/ dir to load the 4 actors from and continue '
                             'training (curriculum transfer; critic warm-start). E.g. '
                             'results/result_2/checkpoints/ep_00900/agents. '
                             'OR a RUN-DIR (results/result_N) → AUTO-PICK best sweet-spot ckpt '
                             '(pick_resume_ckpt) = chọn agents tốt nhất tự động.')
    parser.add_argument('--phase-aux-anneal-end', dest='phase_aux_anneal_end',
                        type=float, default=None,
                        help='Linearly decay --phase-aux-weight to 0 by this FRACTION of '
                             'training (e.g. 0.5 = gone by the halfway point); default None '
                             'keeps it constant. Saves wall-clock two ways: the per-step '
                             'oracle_phase_idx call is skipped once the weight hits 0, and '
                             'the phase term leaves every PPO epoch (updates are 58-65%% of '
                             'run time). Safe candidate because the phase gap is only 5-7%% '
                             'of the remaining J gap once the head is trained. ⚠ There is '
                             'deliberately no equivalent for --assign-aux-weight: that '
                             'teacher is what prevents the routing-identity collapse.')
    parser.add_argument('--no-entangle', dest='no_entangle', action='store_true',
                        help='ABLATION: build the variational circuit with NO '
                             'two-qubit gates -- the nearest-neighbour CZ chain and '
                             'every cross-block bridge are dropped, leaving a '
                             'product state. Parameter count is UNCHANGED (CZ '
                             'carries none) and the R1 readout is untouched, so '
                             'this is a one-variable test of whether the quantum '
                             'correlations matter. On a product state the '
                             'two-qubit observables satisfy <Z_i Z_j>=<Z_i><Z_j> '
                             'exactly (measured residual 3.3e-07 vs 0.351 with '
                             'the chain), so R1-b and R1-c -- two thirds of the '
                             'readout at Case 2 -- become products of R1-a/R1-d '
                             'rather than vanishing. Answers the question Bowles '
                             'et al. 2024 raise: removing entanglement often '
                             'costs nothing, which would make "quantumness" not '
                             'the operative ingredient.')
    parser.add_argument('--phase-aux-hspread', dest='phase_aux_hspread',
                        type=float, default=0.0,
                        help='Redistribute the --phase-aux-weight teacher toward '
                             'high channel-spread states, WITHOUT changing its '
                             'total strength: w_i = w*(1 + gamma*tanh(z_i)) with '
                             'z the batch-standardised log(max|h|/min|h|). '
                             'Measured on 120 Case-2 states: the oracle-phase '
                             'swap is worth +0.045 on states the policy already '
                             'wins and +0.583 on the 20%% it loses (more than '
                             "AO's own +0.434 lead there), and the losers are the "
                             'high-spread ones (corr -0.79). A uniform teacher '
                             'spends its weight where it buys +0.045, which is '
                             'why fixing the target alone (r280) moved nothing. '
                             '0 = uniform, bit-identical to before. Try 0.5-1.0.')
    parser.add_argument('--phase-aux-sweeps', dest='phase_aux_sweeps',
                        type=int, default=0,
                        help='Coordinate-ascent sweeps used to build the '
                             '--phase-aux-weight target (oracle_phase_idx '
                             'n_sweeps). Default 0 reproduces every run so far, '
                             'and is measured HARMFUL: on the true channel that '
                             'target scores J 1.7022 vs the head own 1.7227, so '
                             'the teacher aims BELOW the head. 1 gives 1.7772 '
                             '(+0.0545, 56%% of the J-oracle gain) for 0.56 vs '
                             '0.09 ms/state. Do not raise it further: 3 and 8 '
                             'sweeps fall back to 1.7590 and 1.7565, because the '
                             'ascent maximises coherent gain rather than J.')
    parser.add_argument('--power-aux-anneal-end', dest='power_aux_anneal_end',
                        type=float, default=None,
                        help='Linearly decay --power-aux-weight to 0 by this '
                             'FRACTION of training (e.g. 0.5 = gone by halfway); '
                             'default None keeps it constant. The teacher costs '
                             '12 rate evaluations per rollout step and a term in '
                             'every PPO epoch, and the power axis has no measured '
                             'headroom left once oracles are chosen on g-hat and '
                             'paid on g, so decaying it is a pure wall-clock '
                             'saving. Same machinery as --phase-aux-anneal-end; '
                             'as there, --assign-aux-weight can never anneal.')
    parser.add_argument('--phase-aux-weight', dest='phase_aux_weight', type=float, default=0.0,
                        help='CONTINUOUS phase teacher: add w·(-log π(oracle_phase|s)) to the '
                             'PhaseMLP loss at EVERY update. The supervised phase warm-up is applied '
                             'once and then ERODED by PPO (measured r154: coherent-gain alignment '
                             '15.6%% → 7.0%% over 1300 ep); this cannot decay. Targets the largest '
                             'remaining gap once routing is fixed — on r177, swapping in the oracle '
                             'phase is +0.138 J = 42%% of the whole gap to AO. Costs ~0.08 ms/step '
                             '(closed-form per-element target, no coordinate ascent). Try 0.1-0.5.')
    parser.add_argument('--assign-aux-weight', dest='assign_aux_weight', type=float, default=0.0,
                        help='CONTINUOUS assignment teacher: add w·(-log π(oracle_routing|s)) to the '
                             'actor loss at EVERY update, alongside PPO. Targets the measured collapse '
                             'cause — routing IDENTITY drift (count stays right, WHO gets each IRS '
                             'drifts wrong: r166 mis 0.5→1.8%%, blocked→dead-direct 1.1→5.1%% over '
                             'training). The oracle routing {blocked→its-IRS} reaches QoS 99.7%%, so a '
                             'strong teacher can bypass the (M+1)^K credit-assignment problem instead '
                             'of learning it. Not a periodic re-warmup → never shocks the critic. '
                             'Try 0.1-0.5; anneal to 0 once routing holds. Keeps the actor LEARNED.')
    parser.add_argument('--power-aux-weight', dest='power_aux_weight', type=float, default=0.0,
                        help='AUXILIARY TEACHER for PowerMLP: add w·(-log p(oracle_alloc | α)) to its '
                             'loss at EVERY update, alongside PPO. Power was the ONLY head without a '
                             'teacher and it carries essentially the whole remaining gap — measured on '
                             'r181 @D_k=0.05: routing (taught) 0%%, phase (taught) 9.7%%, POWER 90.5%%. '
                             '--power-fairness only BLENDS toward uniform, it never teaches the head '
                             'to concentrate; this does. Target is the (β*,f*) maximising the design-'
                             'view objective on a 4×3 grid per state (~12 rate evals). Try 0.3-0.5, '
                             'and pair with a LOW --power-fairness (a high α damps the teacher too, '
                             'since both act on the same blended concentrations).')
    parser.add_argument('--ck-aux-weight', dest='ck_aux_weight', type=float, default=0.0,
                        help='AUXILIARY TEACHER for CkMLP: add w·(-log p(oracle_share | α)) to its '
                             'loss at EVERY update, alongside PPO. Keeps C_k LEARNED (it still gets '
                             'the full reward gradient and still adapts) while supplying the credit '
                             'signal one shared advantage across four heads fails to deliver — the '
                             'head is provably capable (isolated demand-fill test -0.683 → -0.061) '
                             'but does not learn it in situ. Unlike a warm-up this cannot be eroded: '
                             'the phase warm-up decayed 15.6%% → 7.0%% alignment over 1300 ep in '
                             'r154, because it was only applied once. Try 0.05-0.3 and anneal to 0. '
                             'Mutually exclusive in spirit with --closed-form-ck (that one removes '
                             'the head entirely; this one teaches it).')
    parser.add_argument('--closed-form-ck', dest='closed_form_ck', action='store_true',
                        help='Replace the learned CkMLP with the exact demand-fill allocator '
                             '(cheapest-unmet-first, then spend the leftover budget). C_k is the '
                             'one sub-problem whose optimum is CLOSED FORM and provable, so '
                             'learning it is pure overhead — unlike assignment ((M+1)^K), phase '
                             '(L^N) or the power simplex, which stay learned. Measured on r159: '
                             'learned C_k gives QoS 77.4%% vs 89.7%% for this allocator on the SAME '
                             'routing/phase/power. Also removes CkMLP from the PPO graph (no '
                             'log-prob, no gradient, no update).')
    parser.add_argument('--freeze-phase', dest='freeze_phase', action='store_true',
                        help='EXPERIMENT-1: freeze PhaseMLP params (skip its grad apply) so the '
                             'assignment head trains against a STATIONARY phase. Diagnostic for '
                             'F2 non-stationarity: does IRS-share catch up to oracle when phase '
                             'stops moving? Use with --resume from a trained ckpt.')
    parser.add_argument('--oracle-warmup', dest='oracle_warmup', action='store_true',
                        help='⭐ DUAL oracle warm-up (2026-06-14, improvement ①): supervised warm-up '
                             'of BOTH the VQC Q-head (assignment → oracle blocked-routing, '
                             'Common-Knowledge [N]) AND PhaseMLP (oracle phase FOR that routing). '
                             'Removes the post-warmup KL_ph/KL_q yank (phase/actor no longer warmed '
                             'for random routing) and injects the routing the agent never learns at '
                             'scale. Reuses --phase-warmup-episodes/--phase-warmup-epochs for both '
                             'stages. Supersedes --phase-warmup (phase-only) — use this instead.')
    parser.add_argument('--phase-warmup', dest='phase_warmup', action='store_true',
                        help='EXPERIMENT-3: supervised PhaseMLP pretrain on closed-form oracle '
                             '(analysis/phase_oracle.oracle_phase_idx) BEFORE PPO begins. Closes '
                             'the +59pp live->oracle alignment gap observed at r14 ep_00700 '
                             '(per-IRS index match 26%%, ~= random 1/L -> PhaseMLP picks wrong-of-4 '
                             'index ~75%% of the time). Reduces F2 init bias for assignment training.')
    parser.add_argument('--assign-warmup-episodes', dest='assign_warmup_episodes', type=int, default=None,
                        help='Rollout episodes for the ① ORACLE ASSIGNMENT warm-up only. '
                             'Defaults to --phase-warmup-episodes. Split out because ① trains the '
                             'VQC ACTOR (~3 min/epoch at K10/M2 — PQC forward over the whole buffer), '
                             'while 🌡 phase is a cheap classical MLP fit: they want very different budgets.')
    parser.add_argument('--assign-warmup-epochs', dest='assign_warmup_epochs', type=int, default=None,
                        help='CE epochs for the ① ORACLE ASSIGNMENT warm-up only. Defaults to '
                             '--phase-warmup-epochs. ① typically SATURATES at ~100%% match by ~30 '
                             'epochs (r151), and now exits immediately at ≥99.5%%, so a small cap '
                             '(30-50) costs nothing and caps the downside.')
    parser.add_argument('--phase-warmup-episodes', dest='phase_warmup_episodes', type=int, default=8,
                        help='Rollout episodes to collect (s_phase, oracle_idx) supervised buffer.')
    parser.add_argument('--phase-warmup-epochs', dest='phase_warmup_epochs', type=int, default=30,
                        help='Supervised CE epochs over the warmup buffer (PPO batch size).')
    parser.add_argument('--irs-bonus', dest='irs_bonus', type=float, default=0.0,
                        help='EXP-2: reward shaping bonus per IRS-routed user. Annealed from '
                             'full at ep=0.2*N_eps to 0 at ep=0.6*N_eps. Default 0.0 = OFF. '
                             'Use when EXP-3 verdict = execution problem (oracle has room but '
                             'policy cannot extract). Suggested first try: 0.05 (mild).')
    parser.add_argument('--D-k', dest='D_k', type=float, default=None,
                        help='Override per-user QoS demand D_k (bps/Hz). Default = params.py value '
                             '(0.10). D_k_HYPOTHESIS test: 0.15 creates regime where Direct-only '
                             'still solves with equal power but smart RSMA+IRS allocation needed.')
    parser.add_argument('--R-LoS', dest='R_LoS_km', type=float, default=None,
                        help='Override satellite LoS coverage radius R_LoS (km). Default = params.py '
                             'value (0.2). Curriculum ramps: 0.2 → 0.3 → 0.4 → 0.5 progressively '
                             'increasing blockage. Use with --resume for ramp transition.')
    parser.add_argument('--power-scale-feats', dest='power_scale_feats',
                        action='store_true',
                        help='Append 2K+1 absolute-scale features to the power '
                             'head input: per-user single-user capacity '
                             'log2(1+Gamma_k), demand pressure D_k/that, and the '
                             'mean log SNR. The h-block is rescaled by its own '
                             'mean to survive LayerNorm, which removes every '
                             'absolute quantity — so the head currently knows '
                             'the per-user ranking but not whether anyone can '
                             'reach D_k, and cannot react to a P_S change at '
                             'all. Power is 73%% of the remaining gap to the '
                             'oracle.')
    parser.add_argument('--power-feat-scale', dest='power_feat_scale',
                        type=float, default=1.0,
                        help='Divisor inside the feasibility-margin tanh of '
                             '--power-scale-feats: tanh(log2(cap/D_k) / s). The '
                             'raw ratio has sd ~6 and spans -9.5..+7.0, so s=1 '
                             '(the default, kept for reproducibility) saturates '
                             '70%% of entries and reduces the margin to a sign '
                             'bit. s=4 measured 2.2-2.5%% saturation while '
                             'keeping ~90%% of the variance. No effect unless '
                             '--power-scale-feats is also given.')
    parser.add_argument('--power-aux-beta-cap', dest='power_aux_beta_cap',
                        action='store_true',
                        help='Restrict the --power-aux-weight teacher grid to '
                             'beta <= 1 (0, 0.25, 0.5, 1.0) instead of the '
                             'default (0, 1, 4, 16). Measured on 100 states, '
                             'choosing on g-hat and scoring on true g: J peaks '
                             'at beta=1 (1.5765) and collapses at beta=4 '
                             '(1.3752), below even equal power (1.5312). The '
                             'default grid therefore teaches over-concentration '
                             'that only pays on the estimate — the likely cause '
                             'of the 21 net-harmful power-aux runs.')
    parser.add_argument('--power-mag-feats', dest='power_mag_feats',
                        choices=['sq', 'log'], default=None,
                        help='Append the per-user magnitude ordering (K dims) to '
                             'the power head input: sq = |h|^2/mean, log = '
                             'log|h|^2 - mean. The teacher target is private ∝ '
                             '|h|^{2b} with b up to 16, and from the rescaled '
                             'Re/Im block alone the head must square its own '
                             'inputs to get there. Fitting the teacher log-shares '
                             'by least squares on 600 states: current input R^2 '
                             '0.245, +|h|^2 0.601, +log|h| 0.559. Independent of '
                             '--power-scale-feats, which supplies a saturated '
                             'feasibility flag rather than the magnitude.')
    parser.add_argument('--P-S', dest='P_S_dBm', type=float, default=None,
                        help='Override P_S (satellite TX power, dBm). Default = per-case auto (K≤5:50, '
                             'K≤10:70, else:100). R3 scale-K sweep: FIX P_S=50 across K=5..9 so the '
                             'scale-K axis is clean (no P_S tier-jump confound).')
    parser.add_argument('--lambda-D', dest='lambda_D_fixed', type=float, default=None,
                        help='Override FIXED λ_D (QoS penalty weight). Default = params.py value. '
                             '⚠ λ_D bump on --resume (e.g. 1.5→3) = [K] LAMBDA-BUMP-COLLAPSE trigger; '
                             'watch Ck-tot/ent_ck/explVar early. Mutually exclusive với --lagrangian.')
    parser.add_argument('--irs-spawn-frac', dest='irs_spawn_frac', type=float, default=None,
                        help='Override irs_spawn_radius_frac (IRS spawn region as fraction of R_LoS). '
                             '=1.0 → IRS can spawn anywhere in the LoS disk (unlock full problem).')
    parser.add_argument('--user-free-frac', dest='user_free_frac', type=float, default=None,
                        help='Override user_free_radius_frac (free-user spawn region as fraction of '
                             'R_LoS). =1.0 → free users spread to the LoS edge (unlock full problem).')
    # ── Lagrangian / constrained-RL: adapt λ_D as a dual variable on a QoS target ──
    parser.add_argument('--actor-mode', dest='actor_mode', choices=['quantum', 'classical'],
                        default='quantum',
                        help='High-level actor type. "quantum" (default) = AE+VQC+SoftmaxPQC actor; '
                             '"classical" = pure-MLP DNN baseline (paper A2/R5 VQC-vs-DNN, param-eff). '
                             'Sub-actors (phase/power/Ck) stay classical in both. ADDITIVE — default '
                             'quantum unchanged.')
    parser.add_argument('--pol-hidden', dest='pol_hidden', type=int, nargs='+', default=None,
                        help='ClassicalActor policy-MLP hidden widths for the A1 width-sweep '
                             '(iso-performance grid), e.g. "--pol-hidden 40" or "--pol-hidden 64 32". '
                             'Default None = params.py / (256, 128). Ignored for --actor-mode quantum.')
    parser.add_argument('--counterfactual-assign', dest='counterfactual_assign', action='store_true',
                        help='Q-HEAD CREDIT FIX (Option 2, COMA-style): per-user counterfactual baseline '
                             'A_cf(k)=R_cf[k,a_k]−Σ_c π_k(c)·R_cf[k,c], re-running downstream under each '
                             'φ_k=c. De-confounds assignment credit + discourages IRS-crowding (load-'
                             'balance). Cost: K·(M+1) downstream forwards/step. Default OFF.')
    parser.add_argument('--cf-coef', dest='cf_coef', type=float, default=0.3,
                        help='Strength of counterfactual-assignment aux (--counterfactual-assign). '
                             'A_cf standardized → coef scale-free. 0.3 default; raise to push harder.')
    parser.add_argument('--cf-anneal-end', dest='cf_anneal_end', type=float, default=0.0,
                        help='Linearly decay cf_coef → 0 by this FRACTION of --episodes (e.g. 0.5 = '
                             'cf off at half-run). FIX for late-run Q-head divergence (đĩa r25 @ep768 '
                             'KL_q=1.71): the cf aux bypasses PPO clipping and A_cf is standardized '
                             'per-minibatch, so once the policy commits the standardization re-amplifies '
                             'noise to unit scale and keeps pushing outside the trust region. Annealing '
                             'keeps cf during exploration (where its credit signal is real) and releases '
                             'the policy as it commits. 0.0 = no anneal (legacy).')
    parser.add_argument('--routing-shape', dest='routing_shape', action='store_true',
                        help='Q-HEAD CREDIT FIX (Option 1): add a directed CE shaping gradient that '
                             'pulls the assignment policy toward the IRS-favored link for users where '
                             'IRS beats direct (the under-routed +Δgain users → IRS≥Blk). Resume-safe '
                             '(gradient-only, no arch change). Default OFF.')
    parser.add_argument('--routing-shape-coef', dest='routing_shape_coef', type=float, default=0.1,
                        help='Strength of routing shaping (--routing-shape). 0.1 = gentle nudge; raise '
                             'to push IRS-routing harder. Too high → overrides RL objective. ')
    parser.add_argument('--power-fairness', dest='power_fairness', type=float, default=0.0,
                        help='③ POWER-FAIRNESS α∈[0,1]: blend the applied private-power split toward '
                             'EQUAL-SPLIT (α·uniform + (1-α)·learned). probe_power_qos proved equal-split '
                             'recovers +28pp QoS at ~zero sum-rate cost (PowerMLP concentrates power on '
                             'already-served users, starving unmet ~3.3×). 0 = off (learned head); 1 = exact '
                             'equal-split; ~0.6-0.8 = sweet-spot guess. Applied to w_p only (PPO path '
                             'unchanged). Distinct from beta_entropy_pwr_private (soft entropy, insufficient). '
                             'NOTE: drives the SPLIT + COMMON axes; the private axis is driven by '
                             '--power-fairness-priv (defaults to this value).')
    parser.add_argument('--power-fairness-priv', dest='power_fairness_priv', type=float, default=None,
                        help='③b α_priv∈[0,1] for the PRIVATE distribution only. Defaults to '
                             '--power-fairness (unchanged behaviour). Set LOW (0-0.3) to let the private '
                             'power CONCENTRATE while --power-fairness still pins the split/common axes: '
                             'the 07-21 Pareto measurement (per-element physics, oracle C_k) shows a '
                             'concentrated private distribution buys large sum-rate at ~unchanged QoS, '
                             'because the common stream carries the QoS. Requires a demand-filling C_k — '
                             'with equal C_k this COLLAPSES QoS (100%%→60%%).')
    parser.add_argument('--power-priv-frac', dest='power_priv_frac', type=float, default=0.8,
                        help='③c target private fraction f for the --power-fairness split blend '
                             '(split → [1-f, f]). 0.8 (default) was QoS-optimal under EQUAL C_k; with a '
                             'demand-filling C_k the reward-optimal f is much lower — measured ≈0.68 for '
                             'K5/M1 and ≈0.28 for K10/M2 at P_S=50dBm, λ_D=1.5, R_LoS=0.5.')
    parser.add_argument('--power-clean-input', dest='power_clean_input', action='store_true',
                        help='Feed PowerMLP ONLY the clean per-user effective channels h_eff (Re+Im, 2K) '
                             'and DROP the o_hat/z_t block. Power/Ck do not train the quantum encoding, so '
                             'o_hat only adds shot-noise that blocks power-concentration (under-extracts '
                             'R_tot vs AE). Fully no-AE (assignment head still reads o_hat). On RESUME the '
                             'PowerMLP is RE-INITIALISED (input dim 2K+rep -> 2K). Incompatible with '
                             '--counterfactual-assign.')
    parser.add_argument('--ck-logit-spread', dest='ck_logit_spread', type=float, default=8.0,
                        help='ck-stability floor (default 8.0): clamp the within-group C_k softmax '
                             'logit spread so it cannot collapse to a one-hot (ent_ck->0, grad-explosion, '
                             'common-rate dumped on 1 user -> QoS crash; r40/r44). Set LARGE (e.g. 1e9) '
                             'to DISABLE ck-stability for ablation (lets C_k collapse like pre-06-18).')
    parser.add_argument('--lagrangian', dest='lagrangian', action='store_true',
                        help='CONSTRAINED-RL: treat λ_D as a Lagrange multiplier and adapt it via '
                             'dual ascent on a QoS target (maximize sum-rate s.t. QoS ≥ --qos-target) '
                             'instead of a FIXED λ_D trade-off. λ_D updates ONCE per rollout from the '
                             'smoothed rolling QoS (gentle ramp — avoids [K] LAMBDA-BUMP-COLLAPSE). '
                             'Default OFF (fixed λ_D unchanged).')
    parser.add_argument('--qos-target', dest='qos_target', type=float, default=0.80,
                        help='Lagrangian target QoS fraction (0.80 = 80%% of users meet D_k). '
                             '--lagrangian only.')
    parser.add_argument('--lambda-lr', dest='lambda_lr', type=float, default=0.05,
                        help='Lagrangian dual-ascent step for λ_D per rollout. KEEP SMALL = gentle '
                             'ramp; a fast λ bump cascades into [K] Ck-collapse. --lagrangian only.')
    parser.add_argument('--lambda-min', dest='lambda_min', type=float, default=1.0,
                        help='Lower clip for adaptive λ_D. --lagrangian only.')
    parser.add_argument('--lambda-max', dest='lambda_max', type=float, default=3.0,
                        help='Upper clip for adaptive λ_D (cap to bound penalty-dom [I-2] / drift; '
                             'watch for drift as λ_D rises past ~2). --lagrangian only.')
    # ── Δ7: PhaseMLP counterfactual gradient (M≥2 lever — Case 2/3) ──
    parser.add_argument('--phase-counterfactual', dest='phase_counterfactual',
                        action='store_true',
                        help='Δ7: weight each active IRS phase-gradient by its π_q expected '
                             'occupancy (F2 mitigation — phase credit tracks routing dist, not '
                             'the jumpy sampled-active set). Only meaningful for M≥2 (Case 2/3); '
                             '≈ no-op at M=1. Default OFF.')
    # ── Entropy-schedule overrides (per-run, no params.py edit) ──
    parser.add_argument('--beta-entropy', dest='beta_entropy', type=float, default=None,
                        help='Override params.py beta_entropy (initial entropy coeff). Higher = '
                             'more exploration → avoid premature assignment commit (esp. high cases).')
    parser.add_argument('--beta-entropy-pwr-private', dest='beta_entropy_pwr_private',
                        type=float, default=None,
                        help='Extra entropy bonus on the PRIVATE power axis, ON TOP of --beta-entropy '
                             '(params.py default 0.003 = 4x the global beta on that axis alone). It was '
                             'a crutch from when PowerMLP could not learn from reward; with the Dirichlet '
                             'policy it now FIGHTS the head — result_150 showed the private max-share '
                             'frozen at 29%% (flat 20%%) while split/common moved 43->81%%. Pass 0.0 to drop it.')
    parser.add_argument('--beta-entropy-min', dest='beta_entropy_min', type=float, default=None,
                        help='Override params.py beta_entropy_min (entropy floor after anneal).')
    parser.add_argument('--beta-entropy-anneal-end', dest='beta_entropy_anneal_end',
                        type=float, default=None,
                        help='Override params.py beta_entropy_anneal_end (fraction of training over '
                             'which β anneals; raise 0.1→0.3 to keep entropy high LONGER).')
    return parser.parse_args()


if __name__ == '__main__':
    train(_parse_args())
