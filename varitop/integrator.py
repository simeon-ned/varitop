"""This module contains variational integrators"""

from typing import Dict
import casadi as cs
import numpy as np

# TODO: unify the interface for integrator, merge p_res and p_next via vcat


class Integrator:
    """Abstract integrator class
    Contains the basic initialization"""

    def __init__(self):
        self._q_0: cs.SX = None

        self._time: [float, float] = None
        self._steps: int = None
        self._dt: float = None

        self.history: Dict = {}
        self.metrics: list[cs.Function] = []

    def set_time(self, span: [float, float], steps: int):
        """Set the time span and number of steps"""
        self._time = span
        self._steps = steps
        self._dt = (span[1] - span[0]) / steps

    def set_initial(self, initial: cs.SX):
        """Set the initial state"""
        self._q_0 = initial

    def add_metric(self, metrics: [cs.Function]):
        """Add a metric to the integrator"""
        self.metrics.extend(metrics)

    def solve(self, backend="newton"):
        """Solve the problem"""
        raise NotImplementedError


class DelIntegrator(Integrator):
    """Discrete Euler-Lagrange integrator"""

    def __init__(self):
        super().__init__()
        self._del: cs.Function = None

    def set_del(self, del_: cs.Function):
        """Specify the DEL residual"""
        self._del = del_

    def solve(self, backend="newton"):
        """Solve the DEL residual"""
        # derive number of lagrange multipliers
        # probably nnz_out not the best way
        # TODO: check if nnz_out() is ok
        lagrange_mult_dim = self._del.nnz_out() - self._q_0.shape[0]

        # initialize the history
        self.history = {}
        self.history["state"] = [self._q_0, self._q_0]
        self.history["multipliers"] = [
            np.zeros(lagrange_mult_dim),
            np.zeros(lagrange_mult_dim),
        ]
        # include required metrics
        for metric in self.metrics:
            q1 = self.history["state"][-2]
            q2 = self.history["state"][-1]
            dq1 = np.zeros_like(q1)
            dq2 = (q2 - q1) / self._dt
            self.history[metric.name()] = [
                metric(q1, dq1),
                metric(q2, dq2),
            ]

        # define the rootfinder problem
        rf = cs.rootfinder("rf", backend, self._del)

        for _ in range(2, self._steps):
            q1 = self.history["state"][-2]
            q2 = self.history["state"][-1]

            # solve the residual and obtain
            # the next state of the system
            # and the lagrange multipliers
            sol = np.array(rf([*q2, *np.zeros(lagrange_mult_dim)], q2, q1, self._dt))
            q3 = sol[: self._q_0.shape[0]].ravel()
            mult = sol[self._q_0.shape[0] :].ravel()

            # update the history
            self.history["state"].append(q3)
            self.history["multipliers"].append(mult)

            # update the metrics
            for metric in self.metrics:
                dq = (q3 - q2) / self._dt
                self.history[metric.name()].append(metric(q3, dq))

        # numpy is more convenient and effective
        for key, value in self.history.items():
            self.history[key] = np.array(value).squeeze()
        return self.history


class DelmIntegrator(Integrator):
    """Discrete Euler-Lagrange integrator in Momentum form"""

    def __init__(self):
        super().__init__()
        self._p_0: cs.SX = None
        self._p_residual: cs.Function = None
        self._p_next: cs.Function = None

    def set_delm(self, p_residual: cs.Function, p_next: cs.Function):
        """Specify momentum residual and next momentum functions"""
        self._p_residual = p_residual
        self._p_next = p_next

    def set_initial(self, initial: [cs.SX, cs.SX]):
        """Specify initial state of the system in terms of state and momentum"""
        self._q_0 = initial[0]
        self._p_0 = initial[1]

    def solve(self, backend="newton"):
        """Solve the DELM"""
        # derive number of lagrange multipliers
        # probably nnz_out not the best way
        # TODO: check if nnz_out() is ok
        lagrange_mult_dim = self._p_residual.nnz_out() - self._q_0.shape[0]

        # Initialize the history
        self.history = {}
        self.history["state"] = [self._q_0]
        self.history["p"] = [self._p_0]
        self.history["multipliers"] = [
            np.zeros(lagrange_mult_dim),
        ]
        # Include required metrics
        for metric in self.metrics:
            q1 = self.history["state"][0]
            dq1 = np.zeros_like(q1)
            self.history[metric.name()] = [metric(q1, dq1)]

        # Define the rootfinder problem
        rf = cs.rootfinder("rf", backend, self._p_residual)

        for _ in range(1, self._steps):
            q1 = self.history["state"][-1]
            p1 = self.history["p"][-1]

            # solve the residual and obtain
            # the next state of the system
            # and the next momentum, as well
            # as the lagrange multipliers
            sol = np.array(rf([*q1, *np.zeros(lagrange_mult_dim)], p1, q1, self._dt))
            q2 = sol[: self._q_0.shape[0]].ravel()
            p2 = self._p_next(q1, q2, self._dt)
            mult = sol[self._q_0.shape[0] :].ravel()

            # update the history
            self.history["state"].append(q2)
            self.history["multipliers"].append(mult)
            self.history["p"].append(p2)

            # update the metrics
            for metric in self.metrics:
                dq = (q2 - q1) / self._dt
                self.history[metric.name()].append(metric(q2, dq))

        # numpy is more convenient and effective
        for key, value in self.history.items():
            self.history[key] = np.array(value).squeeze()
        return self.history
