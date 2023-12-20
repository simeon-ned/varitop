"""Integrators modules"""

from typing import List
import casadi as cs


class VariationalIntegrator:
    """Abstract variational integrator class"""

    def __init__(self) -> None:
        self._lagrangian: cs.Function = None
        self._rule: cs.Function = None
        self._nq: int = None
        self._constrained: bool = False
        self._dynamics_constraint: cs.Function = None

    @property
    def lagrangian(self):
        """Getter for continuous lagrangian"""
        return self._lagrangian

    @lagrangian.setter
    def lagrangian(self, lagrangian: cs.Function):
        """Setter for continuous lagrangian"""
        self._lagrangian = lagrangian

    @property
    def rule(self):
        """Getter for approximation rule"""
        return self._rule

    @rule.setter
    def rule(self, rule: cs.Function):
        """Setter for approximation rule"""
        self._rule = rule

    @property
    def nq(self):
        """Getter for number of generalized coordinates"""
        return self._nq

    @nq.setter
    def nq(self, nq: int):
        """Setter for number of generalized coordinates"""
        self._nq = nq

    def _discrete_lagrangian(self) -> cs.Function:
        """Discretize the Lagrangian
        Ld(q1, q2, dt) = L(q, dq) * dt"""

        if self.nq is None:
            raise RuntimeError("Number of generalized coordinates not set.")

        q0 = cs.SX.sym("q0", self.nq)
        q1 = cs.SX.sym("q1", self.nq)
        dt = cs.SX.sym("dt")

        q, dq = self._rule(q0, q1, dt)

        l = self._lagrangian(q, dq)

        variables = [q0, q1, dt]

        # Construct augmented Lagrangian
        if self._constrained:
            phi = self._dynamics_constraint
            phi_dim = phi.nnz_out()
            lambdas = cs.SX.sym("lambda", phi_dim)

            l += lambdas.T @ phi(q1)

            variables = [q0, q1, lambdas, dt]

        ld = cs.Function("Ld", variables, [l * dt])
        return ld

    def step(self):
        """Get a system for a next step of integration"""
        raise NotImplementedError

    def _append_dynamics_constraint(self, constr: cs.Function):
        """Compose a dynamics constraint function
        phi(q) = residual [number of constraints x 1]"""
        if self._dynamics_constraint is None:
            self._dynamics_constraint = constr
        else:
            q = cs.SX.sym("q", self.nq)
            self._dynamics_constraint = cs.Function(
                "phi",
                [q],
                [cs.vcat([self._dynamics_constraint(q), constr(q)])],
                ["q"],
                ["phi"],
            )

    def add_dynamics_constraint(self, constraints: List[cs.Function]):
        """Add a constraint on dynamics of the system"""
        self._constrained = True
        for constr in constraints:
            self._append_dynamics_constraint(constr)


class DelIntegrator(VariationalIntegrator):
    """Discrete Euler-Lagrange integrator"""

    def step(self):
        """(q0, q1) -> (q1, q2)"""

        if self.lagrangian is None:
            raise RuntimeError("Continuous Lagrangian not set.")

        if self.rule is None:
            raise RuntimeError("Approximation rule not set.")

        if self.nq is None:
            raise RuntimeError("Number of generalized coordinates not set.")

        q0 = cs.SX.sym("q0", self.nq)
        q1 = cs.SX.sym("q1", self.nq)
        q2 = cs.SX.sym("q2", self.nq)
        dt = cs.SX.sym("dt")

        # Variables and arguments
        # for the DEL residual
        variables = [q0, q1, q2, dt]
        titles = ["q-1", "q", "q+1", "dt"]

        arguments1 = [q1, q2, dt]
        arguments2 = [q0, q1, dt]

        if self._constrained:
            phi = self._dynamics_constraint
            lambdas = cs.SX.sym("lambda", phi.nnz_out())

            arguments1 = [q1, q2, lambdas, dt]
            arguments2 = [q0, q1, lambdas, dt]

            variables = [q0, q1, q2, lambdas, dt]
            titles = ["q-1", "q", "q+1", "lambda", "dt"]

        ld = self._discrete_lagrangian()
        d1ld = cs.jacobian(ld(*arguments1), q1)
        d2ld = cs.jacobian(ld(*arguments2), q1)

        residual = d1ld.T + d2ld.T

        # If lagrangian is augmented, we need
        # to add the constraints
        if self._constrained:
            phi = self._dynamics_constraint
            residual = cs.vcat([residual, phi(q2)])

        return cs.Function(
            "del",
            variables,
            [residual],
            titles,
            ["DEL Residual"],
        )


class DelmIntegrator(VariationalIntegrator):
    def step(self):
        """(q0, p0) -> (q1, p1)"""
        raise NotImplementedError
