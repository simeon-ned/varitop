"""Integrators"""

from typing import List
from .misc import quat_prod, qconj, euler_rule
import casadi as cs
import numpy as np


class VariationalIntegrator:
    """Abstract variational integrator class"""

    def __init__(
            self, 
            nq: int = None, 
            nu: int = None,
            free_body: bool = False,
            lagrangian: cs.Function = None,
            rule: cs.Function = euler_rule,
            selector: cs.SX = None
        ) -> None:

        self.nq = nq
        self.nu = nu
        
        self.create_variables()
        self._constrained: bool = False
        self._forced: bool = False
        self._dynamics_constraint: cs.Function = None
        self._generalized_force: cs.Function = cs.Function('F_ext', [self.q, self.dq, self.u], [0], ['q', 'dq', 'u'], ['F_ext'])

        self.free_body = free_body
        self.lagrangian = lagrangian
        self.rule = rule
        self.selector = selector
        

    def create_variables(self):
        self.q = cs.SX.sym("q", self.nq)
        self.dq = cs.SX.sym("dq", self.nq)
        self.u = cs.SX.sym("u", self.nu)
        self.lmbds = cs.SX.sym("lambda", 0)

    def add_force(self, expr: cs.SX):
        self._forced = True
        self._generalized_force = cs.Function(
            'F_ext',
            [self.q, self.dq, self.u],
            [self._generalized_force(self.q, self.dq, self.u) + expr],
            ['q', 'dq', 'u'],
            ['F_ext']
        )

    def add_constraint(self, expr: cs.SX):
        self._constrained = True
        if self._dynamics_constraint is None:
            self._dynamics_constraint = cs.Function(
                'phi',
                [self.q],
                [expr],
                ['q'],
                ['phi']
            )
        else:
            self._dynamics_constraint = cs.Function(
                'phi',
                [self.q],
                [cs.vertcat(self._dynamics_constraint(self.q), expr)],
                ['q'],
                ['phi']
            )

        self.lmbds = cs.SX.sym("lambda", self.lmbds.shape[0] + expr.nnz())

    @property
    def selector(self) -> cs.SX:
        """Selector matrix"""
        return self._selector
    
    @selector.setter
    def selector(self, selector: cs.SX):
        if selector is None:
            return 
        
        if not self.free_body:
            raise ValueError("Tried to define selector for fixed body.")
        self._selector = selector


        force = selector @ self.u
        force_r = cs.vcat([force[3:], 0])
        force_t = force[:3]

        self.add_force(cs.vcat([force_t, 2 * quat_prod(self.q[3:], force_r)]))

    @property
    def free_body(self) -> bool:
        """Index of quaternion in generalized coordinates"""
        return self._free_body

    @free_body.setter
    def free_body(self, free_body: bool):
        self._free_body = free_body

    def v(self, q: cs.SX, dq: cs.SX) -> cs.SX:
        if not self.free_body:
            return dq

        qi = 3
        quat = q[qi : qi + 4]
        quat_dot = dq[qi : qi + 4]
        w = self.q2w(quat, quat_dot)  # angular velocity
        w = w[:3]  # imaginary part, scalar should be zero
        v = cs.vertcat(dq[:qi], w, dq[qi + 4 :])
        return v

    def q2w(self, quat: cs.SX, quat_dot: cs.SX) -> cs.SX:
        """Angular velocity from quaternion and its derivative

        :math:`\\omega = 2 L(\\bar{q}) \\dot{q}`

        :param quat: quaternion
        :type quat: casadi.SX
        :param quat_dot: quaternion derivative
        :type quat_dot: casadi.SX
        :return: angular velocity 4x1
        :rtype: casadi.SX
        """
        return 2 * quat_prod(qconj(quat), quat_dot)

    @property
    def lagrangian(self) -> cs.Function:
        """System's continuous lagrangian"""
        if self._lagrangian is None:
            raise RuntimeError("Continuous Lagrangian not set.")
        return self._lagrangian

    @lagrangian.setter
    def lagrangian(self, lagrangian: cs.Function):
        self._lagrangian = lagrangian

    @property
    def rule(self) -> cs.Function:
        """Midpoint rule"""
        if self._rule is None:
            raise RuntimeError("Approximation rule not set.")
        return self._rule

    @rule.setter
    def rule(self, rule: cs.Function):
        self._rule = rule

    @property
    def nq(self) -> int:
        """Number of generalized coordinates"""
        if self._nq is None:
            raise RuntimeError("Number of generalized coordinates not set.")
        return self._nq

    @nq.setter
    def nq(self, nq: int):
        self._nq = nq

    @property
    def nu(self) -> int:
        """Number of controls"""
        if self._nu is None:
            raise RuntimeError("Number of controls not set.")
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
        v = self.v(q, dq)
        l = self._lagrangian(q, v)

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
        """Compose a force, acting on a body

        :param force: force to add (required to be in the state-space)
        :type force: casadi.Function"""
        self._forced = True

        q0 = cs.SX.sym("q", self.nq)
        dq = cs.SX.sym("dq", self.nq)
        u = cs.SX.sym("u", self.nu)

        new_force = force(q0, dq, u)
        current_force = (
            0 if self._generalized_force is None else self._generalized_force(q0, dq, u)
        )

        self._generalized_force = cs.Function(
            "F",
            [q0, dq, u],
            [current_force + new_force],
            ["q", "dq", "u"],
            ["F"],
        )

    def add_generalized_forces(self, forces: List[cs.Function]):
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
        nu = self.nu

        q0 = cs.SX.sym("q", nq)
        q1 = cs.SX.sym("q", nq)
        q2 = cs.SX.sym("q", nq)
        dt = cs.SX.sym("dt")

        x = q2
        variables = [x, q0, q1, dt]
        arguments = [q0, q1, q2, dt]

        if self._constrained:
            lambdas = cs.SX.sym("lambda", self._dynamics_constraint.nnz_out())
            arguments.append(lambdas)
            x = cs.vertcat(q2, lambdas)
            variables[0] = x

        if self._forced:
            u = cs.SX.sym("u", nu)
            arguments.append(u)
            variables.append(u)

        rfr = cs.Function("rfr", variables, [res(*arguments)])
        rf = cs.rootfinder("rf", "newton", rfr)
        return rf

    def step(
        self, q0: np.ndarray, q1: np.ndarray, dt: float, u: np.ndarray = None
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
        lambda_len = 0
        guess = q1

        if self._constrained:
            # Guess lambdas if any
            lambda_len = self._dynamics_constraint.nnz_out()
            guess = np.hstack([guess, np.zeros(lambda_len)])

        arguments = [guess, q0, q1, dt]

        if self._forced:
            arguments.append(u)

        sol = np.array(rf(*arguments))[: len(guess) - lambda_len].ravel()

        return sol


class DelmIntegrator(VariationalIntegrator):
    """Discrete Euler-Lagrange in Momentum form"""

    def step(self):
        """(q0, p0) -> (q1, p1)"""
        raise NotImplementedError
