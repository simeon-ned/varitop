"""Integrators"""

from typing import List
from .misc import skew_quaternion
import casadi as cs
import numpy as np


class VariationalIntegrator:
    """Abstract variational integrator class"""

    def __init__(self) -> None:
        self._lagrangian: cs.Function = None
        self._rule: cs.Function = None
        self._nq: int = None
        self._nu: int = None
        self._constrained: bool = False
        self._forced: bool = False
        self._dynamics_constraint: cs.Function = None
        self._generalized_force: cs.Function = None
        self._free: bool = False

    @property
    def free(self) -> bool:
        """Determines if body is free-floating"""
        return self._free

    @free.setter
    def free(self, free: bool):
        self._free = free

    @property
    def lagrangian(self) -> cs.Function:
        """System's continuous lagrangian"""
        return self._lagrangian

    @lagrangian.setter
    def lagrangian(self, lagrangian: cs.Function):
        self._lagrangian = lagrangian

    @property
    def rule(self) -> cs.Function:
        """Midpoint rule"""
        return self._rule

    @rule.setter
    def rule(self, rule: cs.Function):
        self._rule = rule

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

    def _discrete_lagrangian(self) -> cs.Function:
        """Discretization of system's lagrangian

        :return: Ld(q0, q1, h)
        :rtype: casadi.Function"""

        if self.nq is None:
            raise RuntimeError("Number of generalized coordinates not set.")

        q0 = cs.SX.sym("q0", self.nq)
        q1 = cs.SX.sym("q1", self.nq)
        dt = cs.SX.sym("dt")

        q, dq = self._rule(q0, q1, dt)

        l = self._lagrangian(q, dq)

        variables = [q0, q1, dt]

        # Augment with constraints
        if self._constrained:
            phi = self._dynamics_constraint
            phi_dim = phi.nnz_out()
            lambdas = cs.SX.sym("lambda", phi_dim)

            l += lambdas.T @ phi(q1)

            variables.append(lambdas)

        # Augment with forces
        # TODO: can not augment with forces
        # even though it is exactly the same as
        # augmenting with constraints forces
        # it does not affect final result
        # if self._forced:
        #     u = cs.SX.sym("u", self.nu)
        #     f = self._generalized_force
        #     l += f(q, dq, u).T @ (q0 + q1)

        #     variables.append(u)

        ld = cs.Function("Ld", variables, [l * dt])
        return ld

    def get_residual(self):
        """Formulate the residual"""
        raise NotImplementedError

    def get_rf_residual(self):
        """Reformulate the residual
        to casadi.rootfinder problem"""
        raise NotImplementedError

    def step(self):
        """Perform a step of integration"""
        raise NotImplementedError

    def _append_generalized_force(self, force: cs.Function):
        """Compose a generalized force

        :param force: force to generalize and add
        :type force: casadi.Function"""
        self._forced = True
        if self.free:
            # Fd = 2qF
            q0 = cs.SX.sym("q", self.nq)
            dq = cs.SX.sym("dq", self.nq)
            u = cs.SX.sym("u", self.nu)
            lq = skew_quaternion(q0)

            nf = lq @ force(q0, dq, u)
            pf = (
                0
                if self._generalized_force is None
                else self._generalized_force(q0, dq, u)
            )

            self._generalized_force = cs.Function(
                "F",
                [q0, dq, u],
                [pf + nf],
                ["q", "dq", "u"],
                ["F"],
            )
        else:
            raise NotImplementedError

    def add_forces(self, forces: List[cs.Function]):
        """Wrapper for forces lists

        :param forces: A list of forces
        :type forces: List[casadi.Function]"""
        for force in forces:
            self._append_generalized_force(force)

    def _append_dynamics_constraint(self, constr: cs.Function):
        """Add phi(q) for constrained dynamics

        :param constr: constraint residual
        :type constr: casadi.Function"""
        self._constrained = True
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

    def add_dynamics_constraints(self, constraints: List[cs.Function]):
        """Wrapper for constraints lists

        :param constraints: A list of constraints
        :type constraints: List[casadi.Function]"""
        for constr in constraints:
            self._append_dynamics_constraint(constr)


class DelIntegrator(VariationalIntegrator):
    """Discrete Euler-Lagrange integrator"""

    def get_residual(self) -> cs.Function:
        """Formulate Discrete Euler-Lagrange residual"""

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

            arguments1.append(lambdas)
            arguments2.append(lambdas)

            variables.append(lambdas)
            titles.append("lambda")

        ld = self._discrete_lagrangian()
        d1ld = cs.jacobian(ld(*arguments1), q1)
        d2ld = cs.jacobian(ld(*arguments2), q1)

        f = self._generalized_force
        residual = d1ld.T + d2ld.T

        if self._forced:
            # Add external forces
            # For now only left force is considered
            # Somehow example with quadrotor diverges
            # if two forces used simultaneously
            u = cs.SX.sym("u", self.nu)
            arguments1.append(u)
            arguments2.append(u)

            variables.append(u)
            titles.append("u")

            p1 = self.rule(q0, q1, dt)
            # p2 = self.rule(q1, q2, dt) # No right force

            residual += f(*p1, u) * dt

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

    def get_rf_residual(self) -> cs.Function:
        """Reformulate residual to match rootfinder signature"""
        res = self.get_residual()
        nq = self.nq
        nv = self.nv
        nu = self.nu

        q0 = cs.SX.sym("q", nq)
        q1 = cs.SX.sym("q", nq)
        q2 = cs.SX.sym("q", nq)
        dt = cs.SX.sym("dt")
        lambdas = cs.SX.sym("lambda", 1)
        u = cs.SX.sym("u", nu)

        x = cs.vertcat(q2, lambdas)
        rfr = cs.Function("rfr", [x, q0, q1, dt, u], [res(q0, q1, q2, dt, lambdas, u)])
        rf = cs.rootfinder("rf", "newton", rfr)
        return rf

    def step(
        self, q0: np.ndarray, q1: np.ndarray, dt: float, u: np.ndarray
    ) -> np.ndarray:
        """
        :param q0: :math:`q_{k-1}`
        :type q0: np.ndarray
        :param q1: :math:`q_{k}`
        :type q1: np.ndarray
        :param dt: :math:`\\mathrm{d}t`
        :type dt: float
        :param u: :math:`u_{k}`
        :type u: np.ndarray
        :return: :math:`q_{k+1}`
        :rtype: np.ndarray


        DEL Residual evolution described by: :math:`(q_0, q_1) \\rightarrow (q_1, q_2)`
        """
        rf = self.get_rf_residual()
        guess = q1

        if self._constrained:
            # Guess lambdas if any
            guess = np.hstack([guess, np.zeros(self._dynamics_constraint.nnz_out())])

        sol = np.array(rf(guess, q0, q1, dt, u))[:4].ravel()

        return sol


class DelmIntegrator(VariationalIntegrator):
    """Discrete Euler-Lagrange in Momentum form"""

    def step(self):
        """(q0, p0) -> (q1, p1)"""
        raise NotImplementedError
