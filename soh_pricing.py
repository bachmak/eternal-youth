# soh_pricing.py
from __future__ import annotations
import numpy as np


def _calendar_rate_multiplier_from_soc(soc: np.ndarray) -> np.ndarray:
    """
    Calendar-aging *shape* proxy vs SOC at 25°C.
    - Minimum at 20% SOC
    - Convex increase towards 100% SOC (stronger near the top)
    - Mild increase below 20% (kept small; LFP low-SOC storage is often not worse)

    This returns a dimensionless multiplier >= ~1.
    """
    s = np.clip(soc.astype(float), 0.0, 1.0)

    # Anchor points (SOC -> multiplier). Chosen to be:
    # - min around 0.2
    # - clearly steeper above 0.8 and especially >0.95
    x = np.array([0.00, 0.20, 0.50, 0.80, 0.95, 1.00], dtype=float)
    y = np.array([1.15, 1.00, 1.35, 2.40, 4.20, 5.00], dtype=float)

    mult = np.interp(s, x, y)

    # Ensure convex-ish behavior in the high SOC end by applying a tiny extra curvature
    # (still very mild; helps reflect “gets worse near 100%”)
    high = s > 0.85
    mult[high] *= (1.0 + 0.15 * ((s[high] - 0.85) / 0.15) ** 2)

    return mult


def simulate_soh_from_soc(
    soc: np.ndarray,
    dt_hours: float,
    base_loss_pct_per_day_at_opt: float = 0.010,
) -> np.ndarray:
    """
    Compute a simple SOH proxy from a SOC time series using a calendar-aging proxy.

    - base_loss_pct_per_day_at_opt: average %SOH loss per day at the optimal SOC (~20%) at 25°C.
      This is a *proxy rate* (not battery-specific), intended mainly for RELATIVE comparison.

    Returns:
      soh array in [0..1], same length as soc, starting at 1.0.
    """
    soc = np.asarray(soc, dtype=float)
    n = len(soc)
    if n == 0:
        return np.array([], dtype=float)

    dt_days = float(dt_hours) / 24.0
    mult = _calendar_rate_multiplier_from_soc(soc)

    # loss per day in %-points:
    loss_pct_per_day = base_loss_pct_per_day_at_opt * mult

    # convert to fraction per step:
    loss_frac_per_step = (loss_pct_per_day / 100.0) * dt_days

    soh = np.empty(n, dtype=float)
    soh[0] = 1.0
    # accumulate
    for k in range(1, n):
        soh[k] = max(0.0, soh[k - 1] - loss_frac_per_step[k - 1])

    return soh


def soc_histogram(
    soc_base: np.ndarray,
    soc_mpc: np.ndarray,
    bin_width: float = 0.05,
):
    """
    Returns histogram data for overlay plots.
    """
    soc_base = np.clip(np.asarray(soc_base, dtype=float), 0.0, 1.0)
    soc_mpc = np.clip(np.asarray(soc_mpc, dtype=float), 0.0, 1.0)

    nbins = int(round(1.0 / bin_width))
    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    count_base, _ = np.histogram(soc_base, bins=edges)
    count_mpc, _ = np.histogram(soc_mpc, bins=edges)

    return {
        "bin_edges": edges,
        "bin_center": centers,
        "count_base": count_base,
        "count_mpc": count_mpc,
    }
