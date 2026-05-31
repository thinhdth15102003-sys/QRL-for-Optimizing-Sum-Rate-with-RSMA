"""
config.py
---------
System-wide hyperparameters for the Multi-IRS ISTN environment.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    # ------------------------------------------------------------------ #
    # Network topology
    # ------------------------------------------------------------------ #
    K: int = 15      # ground users (fixed)
    M: int = 5       # number of IRS (each on a building rooftop)
    N: int = 24      # reflection elements per IRS

    # ------------------------------------------------------------------ #
    # Satellite / RF parameters  (ITU-R model)
    # ------------------------------------------------------------------ #
    P_S_dBm: float       = 50.0   # max transmit power (dBm)
    f_GHz: float         = 1.58   # carrier frequency (GHz)
    h_SR_km: float       = 800.0  # satellite altitude (km)
    G_S_dBi: float       = 60.0   # satellite TX antenna gain (dBi)
    G_U_dBi: float       = 60.0   # user RX antenna gain (dBi)
    noise_mean_dBW: float = -30.0  # 0 dBm = -30 dBW = 10^-3 W (paper uses dBm scale)
    noise_var_dBW: float  = 10.0  # noise variance in dB²; per-step n_0 ~ N(mean, sqrt(var)) dBW

    # ------------------------------------------------------------------ #
    # Rain attenuation  (lognormal, per ITU-R)
    # ------------------------------------------------------------------ #
    rain_mean: float    = 4.0   # μ  (dB)
    rain_var_SR: float  = 0.1   # σ²  satellite → IRS
    rain_var_SU: float  = 0.1   # σ²  satellite → user
    rain_var_RU: float  = 0.1   # σ²  IRS → user

    # ------------------------------------------------------------------ #
    # IRS–user small-scale fading  (Rayleigh + distance path loss)
    # ------------------------------------------------------------------ #
    g_sf_dB: float       = 10.0   # small-scale fading gain (dB)
    path_loss_exp: float = 3.5    # urban path-loss exponent for IRS→user hop
    d0_ref_km: float     = 0.01   # reference distance (10 m) for IRS→user PL

    # ------------------------------------------------------------------ #
    # Imperfect CSI  (multiplicative: Δg = κ|g|ε, ε ~ CN(0,1))
    # ------------------------------------------------------------------ #
    kappa: float = 0.05   # multiplicative CSI error coefficient; applied to all links

    # ------------------------------------------------------------------ #
    # Discrete phase-shift  (2-bit → 4 levels: {0, π/2, π, 3π/2})
    # ------------------------------------------------------------------ #
    quantization_bits: int = 2

    # ------------------------------------------------------------------ #
    # QoS constraint & demand-aware reward  (quadratic shortfall penalty)
    # QoS  : R_k^tot = R_private[k] + C_k[k]  ≥  D_k  for every user k
    # r_t  = Σ R_k^tot  -  λ_D · Σ ( [D_k - R_k^tot]+ / (D_k + ε) )²
    # ------------------------------------------------------------------ #
    D_k_bps_hz: float = 0.10   # per-user QoS demand D_k (bps/Hz)
                                # feasibility: R_private[k] + C_k[k] ≥ D_k
    lambda_D:   float = 1.0    # penalty weight λ_D
    epsilon_qp: float = 0.001  # positive error tolerance ε for QoS denominator

    # ------------------------------------------------------------------ #
    # Ground geometry & satellite LoS zone
    # ------------------------------------------------------------------ #
    R_LoS_km: float = 0.5    # satellite LoS coverage radius on ground (km)
    h_IRS_km: float = 0.02   # IRS/building rooftop height (20 m)
    # Spawn-region bounds as a FRACTION of R_LoS (0→centre, 1→full LoS disk).
    # Concentrating IRS centrally (urban core) + keeping free users inner-half
    # avoids the 'IRS/user at the LoS edge → no coverage' pathology, esp. at
    # large R_LoS. IRS also keep a minimum mutual separation (no clumping).
    irs_spawn_radius_frac:  float = 0.667      # IRS placed within this·R_LoS
    user_free_radius_frac:  float = 0.5        # free users within this·R_LoS

    # ------------------------------------------------------------------ #
    # User mobility  (walking users, random-walk each time step)
    # ------------------------------------------------------------------ #
    user_speed_mps: float = 1.5   # pedestrian speed (m/s)
    dt_s: float           = 1.0   # time step duration (s)

    # ------------------------------------------------------------------ #
    # Building blocking & IRS reflection efficiency  (two distinct roles)
    # ------------------------------------------------------------------ #
    beta_blocking: float = 0.05  # power attenuation of sat→user direct link when
                                  # a building intercepts the path (0 < β ≤ 1)
    beta_IRS: float      = 0.8   # IRS reflection efficiency per element (0 < β ≤ 1);
                                  # ideal passive IRS = 1, hardware losses reduce this
    d_block_km:   float = 0.05   # building footprint half-width (km): each
                                  # building occupies a 3-D box of size
                                  # (2·d_block_km × 2·d_block_km × h_IRS_km).
                                  # The satellite is at (0,0,h_SR_km); blocking
                                  # uses 3-D ray–box intersection to project the
                                  # building's shadow on the ground.

    # ------------------------------------------------------------------ #
    # Derived fields  (computed automatically)
    # ------------------------------------------------------------------ #
    P_S: float              = field(init=False)
    sigma2: float           = field(init=False)
    c: float                = field(init=False)
    phase_levels: np.ndarray = field(init=False)
    user_step_km: float     = field(init=False)   # walking distance per step (km)

    def __post_init__(self):
        self.c            = 3e8
        self.P_S          = 10 ** ((self.P_S_dBm - 30) / 10)
        self.sigma2       = 10 ** (self.noise_mean_dBW / 10)   # nominal σ² at mean dBW
        self.phase_levels = np.array([
            2 * np.pi * k / (2 ** self.quantization_bits)
            for k in range(2 ** self.quantization_bits)
        ])
        self.user_step_km = self.user_speed_mps * self.dt_s / 1000.0

    def summary(self) -> str:
        lines = [
            "SystemConfig",
            f"  Topology  : K={self.K} users (fixed), M={self.M} IRS, N={self.N} elements/IRS",
            f"  Power     : P_S={self.P_S_dBm} dBm ({self.P_S:.2f} W)",
            f"  Frequency : {self.f_GHz} GHz,  satellite alt={self.h_SR_km} km",
            f"  Noise     : N({self.noise_mean_dBW},{self.noise_var_dBW}) dBW per-step  →  nominal σ²={self.sigma2:.2e} W",
            f"  Phase     : {self.quantization_bits}-bit → levels={np.round(self.phase_levels, 4)} rad",
            f"  CSI error : multiplicative κ={self.kappa}  (Δg = κ|g|ε, ε~CN(0,1))",
            f"  QoS/Demand: D_k={self.D_k_bps_hz} bps/Hz (R_tot ≥ D_k), λ_D={self.lambda_D}, ε={self.epsilon_qp}",
            f"  Geometry  : LoS radius={self.R_LoS_km} km, "
            f"h_IRS={self.h_IRS_km*1000:.0f} m (IRS on building rooftop)",
            f"  Mobility  : speed={self.user_speed_mps} m/s, dt={self.dt_s} s "
            f"({self.user_step_km*1000:.1f} m/step)",
            f"  Blocking  : β_block={self.beta_blocking}, β_IRS={self.beta_IRS}, "
            f"r_block={self.d_block_km*1000:.0f} m",
            f"  Path loss : IRS→user exponent={self.path_loss_exp}, "
            f"d0={self.d0_ref_km*1000:.0f} m",
        ]
        return "\n".join(lines)
