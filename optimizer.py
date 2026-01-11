import cvxpy as cp
import numpy as np


class MPCOptimizer:
    """
    Minimal change vs your original MPC:
    - Still convex QP (OSQP).
    - Energy balance and battery dynamics unchanged.
    - Replaces the old "SOC_REF quadratic parking fee" with a convex hinge-based
      calendar-aging proxy that:
        * has minimum around SOC_TARGET (=20%)
        * stays relatively mild up to ~70-80%
        * rises much more strongly above ~85% and especially >95%
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = cfg.HORIZON_SAMPLES

        # Parameters (forecasts / initial state)
        self.param_pv = cp.Parameter(self.n, nonneg=True)
        self.param_load = cp.Parameter(self.n, nonneg=True)
        self.param_soc_start = cp.Parameter(1, nonneg=True)

        # Decision variables
        self.p_ch = cp.Variable(self.n, nonneg=True)
        self.p_dis = cp.Variable(self.n, nonneg=True)
        self.p_gr_im = cp.Variable(self.n, nonneg=True)
        self.p_gr_ex = cp.Variable(self.n, nonneg=True)
        self.soc = cp.Variable(self.n + 1)

        # Constraints
        self.constraints = [
            # initial state
            self.soc[0] == self.param_soc_start,

            # limits
            self.soc >= cfg.SOC_MIN_HARD,
            self.soc <= cfg.SOC_MAX_HARD,
            self.p_ch <= cfg.P_CH_MAX,
            self.p_dis <= cfg.P_DIS_MAX,

            # battery dynamics
            self.soc[1:] == self.soc[:-1] +
            (self.p_ch * cfg.ETA_CH - self.p_dis / cfg.ETA_DIS)
            * cfg.DT / cfg.CAPACITY,

            # energy balance (import - export = load + charge - discharge - pv)
            self.p_gr_im - self.p_gr_ex ==
            self.param_load + self.p_ch - self.p_dis - self.param_pv
        ]

        # -------------------------
        # Objective
        # -------------------------
        dt = cfg.DT

        energy_cost = (
            cfg.PRICE_BUY * cp.sum(self.p_gr_im)
            - cfg.PRICE_SELL * cp.sum(self.p_gr_ex)
        ) * dt

        cyclic_wear_cost = (cfg.COST_WEAR * cp.sum(self.p_ch + self.p_dis)) * dt

        # Calendar aging proxy (convex hinge penalties)
        # Goal: encourage staying near ~20%, punish high SoC strongly (late rising)
        soc = self.soc  # length n+1

        # "Stay near SOC_TARGET=20%" softly
        # - mild penalty for being below 20% (to avoid hugging SOC_MIN)
        # - mild penalty for being above 20% (keeps around target unless PV/price says otherwise)
        below = cp.pos(cfg.SOC_TARGET - soc)
        above = cp.pos(soc - cfg.SOC_TARGET)

        # High SoC "danger zones"
        high85 = cp.pos(soc - 0.85)
        high95 = cp.pos(soc - 0.95)

        # Convex penalties (squares)
        cal_cost = (
            cfg.COST_SOC_HOLDING * (
                cfg.W_BELOW_TARGET * cp.sum_squares(below)
                + cfg.W_ABOVE_TARGET * cp.sum_squares(above)
                + cfg.W_HIGH_85 * cp.sum_squares(high85)
                + cfg.W_HIGH_95 * cp.sum_squares(high95)
            )
        )

        total_cost = energy_cost + cyclic_wear_cost + cal_cost

        self.problem = cp.Problem(cp.Minimize(total_cost), self.constraints)

    def solve(self, pv_forecast, consumption_forecast, curr_soc):
        epsilon = 1e-4

        if curr_soc < self.cfg.SOC_MIN_HARD + epsilon:
            curr_soc = self.cfg.SOC_MIN_HARD + epsilon

        if curr_soc > self.cfg.SOC_MAX_HARD - epsilon:
            curr_soc = self.cfg.SOC_MAX_HARD - epsilon

        self.param_pv.value = np.array(pv_forecast, dtype=float)
        self.param_load.value = np.array(consumption_forecast, dtype=float)
        self.param_soc_start.value = np.atleast_1d(float(curr_soc))

        try:
            self.problem.solve(
                solver=cp.OSQP,
                verbose=False,
                warm_start=True,
                max_iter=40000,
                eps_abs=1e-3,
                eps_rel=1e-3
            )
        except cp.SolverError:
            pass

        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            raise Exception(f"Problem {self.problem.status} is not optimal")

        return (
            float(self.p_ch.value[0]),
            float(self.p_dis.value[0]),
            float(self.p_gr_im.value[0]),
            float(self.p_gr_ex.value[0])
        )
