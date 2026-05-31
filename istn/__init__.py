"""
istn  —  Multi-IRS ISTN Environment Package
============================================
Extensions over Tan et al. (IEEE TCCN 2026):
  • M IRS (each on a rooftop) with per-IRS blocking coefficient β_m
  • 2-bit discrete phase-shift quantisation
  • Imperfect CSI  Δg ~ CN(0, σ_e²)
  • User grouping generalised to M+1 groups (1 per IRS + direct link)
"""

from istn.config    import SystemConfig
from istn.channel   import ChannelModel

__all__ = [
    "SystemConfig",
    "ChannelModel",
]
