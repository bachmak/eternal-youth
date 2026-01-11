# optimizer.py
import cvxpy as cp
import numpy as np

from soh_pricing import calendar_cost_pwl_segments


class MPCOptimizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n = cfg.HORIZON_SAMPLES

        self.param_pv = cp.Parameter(self.n, nonneg=True)
        self.param_load = cp.Parameter(self.n, nonneg=True)
        self.param_soc_start = cp.Parameter(1, nonneg=True)

        self.p_ch = cp.Variable(self.n, nonneg=True)
        self.p_dis = cp.Variable(self.n, nonneg=True)
        self.p_gr_im = cp.Variable(self.n, nonneg=True)
        self.p_gr_ex = cp.Variable(self.n, nonneg=True)
        self.soc = cp.Variable(self.n + 1)

        # --------------------
        # Constraints
        # --------------------
        self.constraints = [
            self.soc[0] == self.param_soc_start,

            self.soc >= cfg.SOC_MIN_HARD,
            self.soc <= cfg.SOC_MAX_HARD,
            self.p_ch <= cfg.P_CH_MAX,
            self.p_dis <= cfg.P_DIS_MAX,

            # dynamics
            self.soc[1:] == self.soc[:-1] +
            (self.p_ch * cfg.ETA_CH - self.p_dis / cfg.ETA_DIS)
            * cfg.DT / cfg.CAPACITY,

            # balance
            self.p_gr_im - self.p_gr_ex ==
            self.param_load + self.p_ch - self.p_dis - self.param_pv
        ]

        # --------------------
        # Base energy + wear costs
        # --------------------
        base_cost = (
            (
                cfg.PRICE_BUY * cp.sum(self.p_gr_im)
                - cfg.PRICE_SELL * cp.sum(self.p_gr_ex)
                + cfg.COST_WEAR * cp.sum(self.p_ch + self.p_dis)
            ) * cfg.DT
        )

        # --------------------
        # Calendar aging cost (SoC pricing) - convex PWL on soc in [0.20..1.00]
        # --------------------
        if getattr(cfg, "USE_CALENDAR_AGING_COST", True):
            seg = calendar_cost_pwl_segments(
                dt_minutes=cfg.DT * 60.0,
                T_days=getattr(cfg, "AGING_T_DAYS", 160.0),
                battery_cost_eur=getattr(cfg, "BATTERY_COST_EUR", 4000.0),
                eol_soh=getattr(cfg, "EOL_SOH", 0.80),
            )

            a = seg["a"]
            b = seg["b"]

            # Apply cost to soc[1:] (state during each interval)
            z = cp.Variable(self.n)  # z[k] = calendar cost per step at interval k

            # z[k] >= a_i*soc[k+1] + b_i for all segments i  (max-of-lines)
            for i in range(len(a)):
                self.constraints.append(z >= a[i] * self.soc[1:] + b[i])

            # Deep discharge extra penalty for soc < 0.20 (smooth convex)
            # This is *not* from the literature curve; it's a safety preference.
            soc_opt_min = getattr(cfg, "SOC_OPT_MIN", 0.20)
            low_pen = getattr(cfg, "LOW_SOC_PENALTY", 50.0)  # tune
            low_soc_penalty = low_pen * cp.sum_squares(cp.pos(soc_opt_min - self.soc[1:]))

            calendar_cost = cp.sum(z) + low_soc_penalty
        else:
            calendar_cost = 0.0

        self.problem = cp.Problem(cp.Minimize(base_cost + calendar_cost), self.constraints)

    def solve(self, pv_forecast, consumption_forecast, curr_soc):
        epsilon = 1e-4

        if curr_soc < self.cfg.SOC_MIN_HARD + epsilon:
            curr_soc = self.cfg.SOC_MIN_HARD + epsilon
        if curr_soc > self.cfg.SOC_MAX_HARD - epsilon:
            curr_soc = self.cfg.SOC_MAX_HARD - epsilon

        self.param_pv.value = np.array(pv_forecast)
        self.param_load.value = np.array(consumption_forecast)
        self.param_soc_start.value = np.atleast_1d(curr_soc)

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
