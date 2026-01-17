# optimizer.py
import cvxpy as cp
import numpy as np


class MPCOptimizer:
    """
    Deterministic convex MPC (QP via OSQP) for residential PV-battery HEMS.

    Hard rule INSIDE the QP (convex):
    - Charging is limited to forecast PV surplus:
        p_ch[t] <= max(pv_hat[t] - load_hat[t], 0)
      -> prevents "charging from grid" in the PLAN (w.r.t. forecasts).

    IMPORTANT:
    - Strict "no simultaneous charge & discharge" is non-convex (needs binaries).
      We keep the QP convex and enforce hard actuation feasibility in pipeline.py:
        (1) no measured grid-charging (clamp to measured surplus),
        (2) no simultaneous ch/dis (projection preserving net power).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = int(cfg.HORIZON_SAMPLES)

        # Parameters (forecasts / initial state)
        self.param_pv = cp.Parameter(self.n, nonneg=True)
        self.param_load = cp.Parameter(self.n, nonneg=True)
        self.param_soc_start = cp.Parameter(1, nonneg=True)

        # Forecast PV surplus upper bound for charging (hard in plan)
        self.param_pv_surplus = cp.Parameter(self.n, nonneg=True)

        # Decision variables
        self.p_ch = cp.Variable(self.n, nonneg=True)
        self.p_dis = cp.Variable(self.n, nonneg=True)
        self.p_gr_im = cp.Variable(self.n, nonneg=True)
        self.p_gr_ex = cp.Variable(self.n, nonneg=True)
        self.soc = cp.Variable(self.n + 1)

        cfg = self.cfg
        dt = float(cfg.DT)

        constraints = []

        # initial state
        constraints += [self.soc[0] == self.param_soc_start]

        # bounds
        constraints += [self.soc >= cfg.SOC_MIN_HARD, self.soc <= cfg.SOC_MAX_HARD]
        constraints += [self.p_ch <= cfg.P_CH_MAX]
        constraints += [self.p_dis <= cfg.P_DIS_MAX]

        # HARD (plan): no grid-charging (within forecast): charge only from PV surplus
        constraints += [self.p_ch <= self.param_pv_surplus]

        # Soft convex anti-simultaneous (still convex; hard version in actuation)
        p_max = float(max(cfg.P_CH_MAX, cfg.P_DIS_MAX))
        constraints += [self.p_ch + self.p_dis <= p_max]

        # battery dynamics
        constraints += [
            self.soc[1:] == self.soc[:-1]
            + (self.p_ch * cfg.ETA_CH - self.p_dis / cfg.ETA_DIS) * (dt / cfg.CAPACITY)
        ]

        # energy balance (import - export = load + charge - discharge - pv)
        constraints += [
            self.p_gr_im - self.p_gr_ex == self.param_load + self.p_ch - self.p_dis - self.param_pv
        ]

        # Objective: energy cost
        energy_cost = (cfg.PRICE_BUY * cp.sum(self.p_gr_im) - cfg.PRICE_SELL * cp.sum(self.p_gr_ex)) * dt

        # Cyclic wear proxy (throughput)
        cyclic_wear_cost = (cfg.COST_WEAR * cp.sum(self.p_ch + self.p_dis)) * dt

        # Calendar aging proxy (hinge penalties around SOC_TARGET, plus high-SoC zones)
        soc = self.soc  # length n+1
        below = cp.pos(cfg.SOC_TARGET - soc)
        above = cp.pos(soc - cfg.SOC_TARGET)
        high85 = cp.pos(soc - 0.85)
        high95 = cp.pos(soc - 0.95)

        cal_cost = cfg.COST_SOC_HOLDING * (
            cfg.W_BELOW_TARGET * cp.sum_squares(below)
            + cfg.W_ABOVE_TARGET * cp.sum_squares(above)
            + cfg.W_HIGH_85 * cp.sum_squares(high85)
            + cfg.W_HIGH_95 * cp.sum_squares(high95)
        )

        # Small quadratic regularization for numerical stability
        reg = 1e-4 * (cp.sum_squares(self.p_ch) + cp.sum_squares(self.p_dis))

        total_cost = energy_cost + cyclic_wear_cost + cal_cost + reg
        self.problem = cp.Problem(cp.Minimize(total_cost), constraints)

    def solve_plan(self, pv_forecast, load_forecast, curr_soc, n_apply: int):
        """
        Solve the QP and return the first n_apply battery actions (arrays).
        """
        epsilon = 1e-4

        # Keep SOC strictly inside bounds for solver stability
        curr_soc = float(curr_soc)
        curr_soc = max(curr_soc, self.cfg.SOC_MIN_HARD + epsilon)
        curr_soc = min(curr_soc, self.cfg.SOC_MAX_HARD - epsilon)

        pv_f = np.asarray(pv_forecast, dtype=float)
        ld_f = np.asarray(load_forecast, dtype=float)

        self.param_pv.value = pv_f
        self.param_load.value = ld_f
        self.param_soc_start.value = np.atleast_1d(curr_soc)

        # forecast PV surplus for hard no-grid-charge constraint (in the plan)
        pv_surplus = np.maximum(pv_f - ld_f, 0.0)
        self.param_pv_surplus.value = pv_surplus

        self.problem.solve(
            solver=cp.OSQP,
            verbose=False,
            warm_start=True,
            max_iter=40000,
            eps_abs=1e-3,
            eps_rel=1e-3,
        )

        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"MPC solve failed: {self.problem.status}")

        # Extract first n_apply actions and sanitize tiny numeric violations
        n_apply = int(max(1, min(n_apply, self.n)))
        p_ch = np.asarray(self.p_ch.value[:n_apply], dtype=float)
        p_dis = np.asarray(self.p_dis.value[:n_apply], dtype=float)

        # Clip to physical bounds
        p_ch = np.clip(p_ch, 0.0, float(self.cfg.P_CH_MAX))
        p_dis = np.clip(p_dis, 0.0, float(self.cfg.P_DIS_MAX))

        # Enforce PV surplus bound again (numerical safety, still forecast-based)
        pv_surplus_apply = np.maximum(pv_f[:n_apply] - ld_f[:n_apply], 0.0)
        p_ch = np.minimum(p_ch, pv_surplus_apply)

        # Enforce convex sum-bound again (numerical safety)
        p_max = float(max(self.cfg.P_CH_MAX, self.cfg.P_DIS_MAX))
        s = p_ch + p_dis
        mask = s > p_max + 1e-9
        if np.any(mask):
            scale = (p_max / s[mask])
            p_ch[mask] *= scale
            p_dis[mask] *= scale

        return p_ch, p_dis

    def solve(self, pv_forecast, load_forecast, curr_soc):
        """
        Backward-compatible: return only first-step action.
        """
        p_ch, p_dis = self.solve_plan(pv_forecast, load_forecast, curr_soc, n_apply=1)
        return float(p_ch[0]), float(p_dis[0])
