import cvxpy as cp
import numpy as np


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

            # energy balance
            self.p_gr_im - self.p_gr_ex ==
            self.param_load + self.p_ch - self.p_dis - self.param_pv
        ]

        cost_terms = (
                (
                        cfg.PRICE_BUY * cp.sum(self.p_gr_im)
                        - cfg.PRICE_SELL * cp.sum(self.p_gr_ex)
                        + cfg.COST_WEAR * cp.sum(self.p_ch + self.p_dis)
                ) * cfg.DT
                + cfg.COST_SOC_HOLDING * cp.sum_squares(self.soc - cfg.SOC_REF)
        )

        self.problem = cp.Problem(cp.Minimize(cost_terms), self.constraints)

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
