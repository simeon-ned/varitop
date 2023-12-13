import casadi as cs
import numpy as np
from typing import Dict


class DelIntegrator:
    def __init__(self):
        self._del: cs.Function = None
        self._q_0: cs.SX = None

        self._time: [float, float] = None
        self._steps: int = None
        self._dt: float = None

        self.history: Dict = {}
        self.metrics: [cs.Function] = []

    def set_del(self, del_: cs.Function):
        self._del = del_

    def set_time(self, span: [float, float], steps: int):
        self._time = span
        self._steps = steps
        self._dt = (span[1] - span[0]) / steps

    def set_initial(self, initial: cs.SX):
        self._q_0 = initial

    def add_metric(self, metrics: [cs.Function]):
        self.metrics.extend(metrics)

    def solve(self, backend="newton"):
        # derive number of lagrange multipliers
        # probably nnz_out not the best way
        lagrange_mult_dim = self._del.nnz_out() - self._q_0.shape[0]

        self.history = {}
        self.history["state"] = [self._q_0, self._q_0]
        self.history["multipliers"] = [
            np.zeros(lagrange_mult_dim),
            np.zeros(lagrange_mult_dim),
        ]
        for metric in self.metrics:
            q1 = self.history["state"][-2]
            q2 = self.history["state"][-1]
            dq1 = np.zeros_like(q1)
            dq2 = (q2 - q1) / self._dt
            self.history[metric.name()] = [
                metric(q1, dq1),
                metric(q2, dq2),
            ]

        rf = cs.rootfinder("rf", backend, self._del)

        for _ in range(2, self._steps):
            q1 = self.history["state"][-2]
            q2 = self.history["state"][-1]

            sol = np.array(rf([*q2, *np.zeros(lagrange_mult_dim)], q2, q1, self._dt))

            q3 = sol[: self._q_0.shape[0]].ravel()
            mult = sol[self._q_0.shape[0] :].ravel()

            self.history["state"].append(q3)
            self.history["multipliers"].append(mult)

            for metric in self.metrics:
                dq = (q3 - q2) / self._dt
                self.history[metric.name()].append(metric(q3, dq))

        for key, value in self.history.items():
            self.history[key] = np.array(value).squeeze()
        return self.history
