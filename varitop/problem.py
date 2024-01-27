"""Problem module"""
from typing import Dict, List
import casadi as cs
import numpy as np

from varitop.integrator import VariationalIntegrator
from .variables import Variable, Momentum, State, Velocity, Control


class VaritopProblem:
    """Variational Optimization Problem"""

    def __init__(self):
        self._variables: Dict[str, Variable] = {}
        self._nodes: int = None
        self._nq: int = None
        self._nu: int = None

        self._q: cs.MX = None
        self._u: cs.MX = None

        self._problem = None
        self._residuals = None
        self._constraints = None
        self._limits = None
        self._cost = None

        self._integrator: VariationalIntegrator = None

    @property
    def integrator(self) -> VariationalIntegrator:
        """Problem integrator"""
        return self._integrator

    @integrator.setter
    def integrator(self, integrator: VariationalIntegrator):
        self._integrator = integrator

    @property
    def nodes(self) -> int:
        """Number of solution nodes"""
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: int):
        self._nodes = nodes

    @property
    def nq(self) -> int:
        """Number of generalized coordinates"""
        return self._nq

    @nq.setter
    def nq(self, nq: int):
        self._nq = nq

    @property
    def nu(self) -> int:
        """Number of controls"""
        return self._nu

    @nu.setter
    def nu(self, nu: int):
        self._nu = nu

    def add_intermidiate_cost(self, cost):
        raise NotImplementedError

    def add_terminal_cost(self, cost):
        raise NotImplementedError

    def add_initial_constraint(self, q_init):
        raise NotImplementedError

    def add_intermidiate_constraint(self, expr):
        raise NotImplementedError

    def add_terminal_constraint(self, q_final):
        raise NotImplementedError

    def add_limits(self, lb, expr, ub):
        raise NotImplementedError


class OptiProblem(VaritopProblem):
    """Optimization through cs.Opti"""

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt

    @property
    def residuals(self):
        if self._residuals is None:
            self._residuals = []

        return self._residuals

    @residuals.setter
    def residuals(self, residuals):
        self._residuals = residuals

    def construct_residuals(self):
        if self.integrator is None:
            raise ValueError("Integrator is not set")

        residuals = []
        q = self.q
        u = self.u
        DEL_residual = self.integrator.get_residual()

        for i in range(1, self.nodes - 1):
            residuals.append(
                DEL_residual(
                    q[i - 1, :].T, q[i, :].T, q[i + 1, :].T, self.dt, u[i - 1, :]
                )
                == 0
            )

        self.residuals = residuals

    @property
    def cost(self):
        if self._cost is None:
            self._cost = 0

        return self._cost

    @cost.setter
    def cost(self, cost):
        self._cost = cost

    def add_intermidiate_cost(self, cost):
        self.cost = self.cost + cost

    def add_terminal_cost(self, cost):
        self.cost = self.cost + cost

    @property
    def constraints(self):
        if self._constraints is None:
            self._constraints = []

        return self._constraints

    def add_initial_constraint(self, q_init):
        self.constraints.append(self.q[0, :].T == q_init)

    def add_intermidiate_constraint(self, expr):
        self.constraints.append(expr)

    def add_terminal_constraint(self, q_final):
        self.constraints.append(self.q[-1, :].T == q_final)

    @property
    def limits(self):
        if self._limits is None:
            self._limits = []

        return self._limits

    def add_limits(self, lb, expr, ub):
        self.limits.append(self.problem.bounded(lb, expr, ub))

    @property
    def problem(self):
        if self._problem is None:
            self._problem = cs.Opti()

        return self._problem

    @property
    def q(self):
        if self._q is None:
            self._q = self.problem.variable(self.nodes, self.nq)

        return self._q

    @property
    def u(self):
        if self._u is None:
            self._u = self.problem.variable(self.nodes - 1, self.nu)

        return self._u

    def build(self):
        self.construct_residuals()
        for residual in self.residuals:
            self.problem.subject_to(residual)

        for constraint in self.constraints:
            self.problem.subject_to(constraint)

        for limit in self.limits:
            self.problem.subject_to(limit)

        self.problem.minimize(self.cost)
