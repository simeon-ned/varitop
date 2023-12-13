"""Problem module"""
from typing import Dict, Callable
import numpy as np
import casadi as cs
from .variables import Variable, Momentum, State, Velocity


class VaritopProblem:
    """Problem is going to be here"""

    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.quadrature: Callable = None
        self.lagrangian: cs.Function = None
        self.nodes: int = 0
        self.eq_constraints: list[cs.Function] = []

    def set_nodes(self, nodes: int):
        """Set number of nodes"""
        self.nodes = nodes

    def create_variable(
        self, variable: type[Variable], name: str, dim: int, active: list[int] = None
    ) -> Variable:
        """Create a variable"""
        if active is None:
            active = np.ones(self.nodes, dtype=int)
        self.variables[variable] = variable(name, dim, active)
        return self.variables[variable]

    def create_state(self, name: str, dim: int, active: list[int] = None) -> Variable:
        """Create a state variable"""
        return self.create_variable(State, name, dim, active)

    def create_velocity(self, name: str, active: list[int] = None) -> Variable:
        """Create a velocity variable"""
        return self.create_variable(Velocity, name, active)

    def create_momentum(self, name: str, active: list[int] = None) -> Variable:
        """Create a momentum variable"""
        return self.create_variable(Momentum, name, active)

    def set_quadrature(self, quadrature: Callable):
        """Set the quadrature rule"""
        self.quadrature = quadrature

    def set_continuous_lagrangian(self, lagrangian: cs.Function):
        """Set the continuous lagrangian"""
        self.lagrangian = lagrangian

    def get_discrete_lagrangian(self) -> [cs.Function, cs.Function]:
        """Get the discretization"""

        if self.quadrature is None:
            raise RuntimeError("Quadrature rule not set.")

        if self.lagrangian is None:
            raise RuntimeError("Continuous Lagrangian not set.")

        # Fix the dimensions
        dim = self.variables[State].shape[0]

        q1 = cs.SX.sym("q1", dim)
        q2 = cs.SX.sym("q2", dim)
        dt = cs.SX.sym("dt")

        q, dq = self.quadrature(q1, q2, dt)
        return cs.Function("L", [q1, q2, dt], [self.lagrangian(q, dq) * dt])

    def add_constraint(self, constr_type: str, constraint: cs.Function):
        """Add a constraint"""
        if constr_type == "=":
            self.eq_constraints.append(constraint)
        else:
            raise NotImplementedError()

    def get_del_residual(self) -> cs.Function:
        """Discrete Euler-Lagrange residual
        D1L + D2L + lambda * phi * dt = 0
        I am hoping that having lambdas as
        free variables will allow solution
        with respect to them and next step"""
        discrete_lagrangian = self.get_discrete_lagrangian()

        q1 = cs.SX.sym("q1", self.variables[State].shape[0])
        q2 = cs.SX.sym("q2", self.variables[State].shape[0])
        q3 = cs.SX.sym("q3", self.variables[State].shape[0])
        dt = cs.SX.sym("dt")
        eq_lambdas = cs.SX.sym("eq_lambda", len(self.eq_constraints))

        # First slot derivative
        d1l = cs.Function(
            "D1L", [q1, q2, dt], [cs.jacobian(discrete_lagrangian(q1, q2, dt), q1)]
        )
        # Second slot derivative
        d2l = cs.Function(
            "D2L", [q1, q2, dt], [cs.jacobian(discrete_lagrangian(q1, q2, dt), q2)]
        )

        del_residual = d1l(q2, q3, dt).T + d2l(q1, q2, dt).T

        for i, phi in enumerate(self.eq_constraints):
            del_residual += cs.jacobian(phi(q2), q2).T * eq_lambdas[i] * dt

        variable = cs.vertcat(q3, eq_lambdas)
        eq_constraints = cs.vcat([constr(q3) for constr in self.eq_constraints])
        del_residual = cs.vertcat(del_residual, eq_constraints)

        # Discrete Euler-Lagrange residual
        del_residual = cs.Function(
            "DEL",
            [variable, q2, q1, dt],
            [del_residual],
            ["q-1", "q", "q+1", "dt"],
            ["DEL_residual"],
        )
        return del_residual

    def get_delm_residual(self) -> [cs.Function, cs.Function]:
        discrete_lagrangian = self.get_discrete_lagrangian()

        q1 = cs.SX.sym("q1", self.variables[State].shape[0])
        q2 = cs.SX.sym("q2", self.variables[State].shape[0])
        p = cs.SX.sym("p", self.variables[Momentum].shape[0])
        dt = cs.SX.sym("dt")
        eq_lambdas = cs.SX.sym("eq_lambda", len(self.eq_constraints))

        # First slot derivative
        d1l = cs.Function(
            "D1L", [q1, q2, dt], [cs.jacobian(discrete_lagrangian(q1, q2, dt), q1)]
        )
        # Second slot derivative
        d2l = cs.Function(
            "D2L", [q1, q2, dt], [cs.jacobian(discrete_lagrangian(q1, q2, dt), q2)]
        )

        p_residual = p + d1l(q1, q2, dt).T
        for i, phi in enumerate(self.eq_constraints):
            p_residual += cs.jacobian(phi(q1), q1).T * eq_lambdas[i] * dt

        variable = cs.vertcat(q2, eq_lambdas)
        eq_constraints = cs.vcat([constr(q2) for constr in self.eq_constraints])
        p_residual = cs.vertcat(p_residual, eq_constraints)

        p_residual = cs.Function(
            "P_residual",
            [variable, p, q1, dt],
            [p_residual],
            ["[q+1, lambdas]", "p", "q", "dt"],
            ["P_residual"],
        )

        return p_residual, d2l
