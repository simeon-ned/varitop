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
        self.eq_constraint: cs.Function = None

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

    def _merge_eq_contraints(self, constraints: list[cs.Function]) -> cs.Function:
        q = cs.SX.sym("q", self.variables[State].shape[0])
        return cs.Function(
            "phi", [q], [cs.vcat(list(map(lambda c: c(q), constraints)))]
        )

    def add_constraints(self, constr_type: str, constraints: list[cs.Function]):
        """Add a constraint"""
        if constr_type == "=":
            merged = self._merge_eq_contraints(constraints)
            if self.eq_constraint is None:
                self.eq_constraint = merged
            else:
                self.eq_constraint = self._merge_eq_contraints(
                    [self.eq_constraint, merged]
                )
        else:
            raise NotImplementedError()

    def get_del_residual(self) -> cs.Function:
        """Discrete Euler-Lagrange solution
        D1L(q1, q3) + D2L(q1, q2) + lambda * phi(q2) * dt = 0
        This equation can be solved to obtain the next state q3
        """
        discrete_lagrangian = self.get_discrete_lagrangian()

        if self.variables.get(State) is None:
            raise RuntimeError("State variable not created. Use create_state()")

        q1 = cs.SX.sym("q1", self.variables[State].shape[0])
        q2 = cs.SX.sym("q2", self.variables[State].shape[0])
        q3 = cs.SX.sym("q3", self.variables[State].shape[0])
        dt = cs.SX.sym("dt")

        # Lagrange multipliers
        eq_lambdas = cs.SX.sym("eq_lambda", self.eq_constraint.nnz_out())

        # First slot derivative
        d1l = cs.Function(
            "D1L", [q1, q2, dt], [cs.jacobian(discrete_lagrangian(q1, q2, dt), q1)]
        )
        # Second slot derivative
        d2l = cs.Function(
            "D2L", [q1, q2, dt], [cs.jacobian(discrete_lagrangian(q1, q2, dt), q2)]
        )
        # Initial residual

        # lambda @ phi(q2) * dt
        ljcdt = eq_lambdas.T @ cs.jacobian(self.eq_constraint(q2), q2) * dt
        # D1L(q1, q2) + D2L(q1, q2) + lambda @ phi(q2) * dt
        del_residual = d1l(q2, q3, dt).T + d2l(q1, q2, dt).T + ljcdt.T
        # system of equations include constraints
        # on the next state of the system q3
        del_residual = cs.vcat([del_residual, self.eq_constraint(q3)])

        # variable is the vector of unknowns
        variable = cs.vertcat(q3, eq_lambdas)
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
        """Discrete Euler Lagrange in Momentum form solution
        p_residual = p_prev + D1L(q1, q2) + lambda * phi(q2) * dt
        p_next = D2L(q1, q2)
        Residual can be solved to obtain the next state q2
        p_next provides the momentum on the next state
        """
        discrete_lagrangian = self.get_discrete_lagrangian()

        q1 = cs.SX.sym("q1", self.variables[State].shape[0])
        q2 = cs.SX.sym("q2", self.variables[State].shape[0])
        p = cs.SX.sym("p", self.variables[Momentum].shape[0])
        dt = cs.SX.sym("dt")

        # Lagrange multipliers
        eq_lambdas = cs.SX.sym("eq_lambda", self.eq_constraint.nnz_out())

        # First slot derivative
        d1l = cs.Function(
            "D1L", [q1, q2, dt], [cs.jacobian(discrete_lagrangian(q1, q2, dt), q1)]
        )
        # Second slot derivative
        d2l = cs.Function(
            "D2L", [q1, q2, dt], [cs.jacobian(discrete_lagrangian(q1, q2, dt), q2)]
        )

        # lambda @ phi(q2) * dt
        ljcdt = eq_lambdas.T @ cs.jacobian(self.eq_constraint(q1), q1) * dt

        # initial residual
        p_residual = p + d1l(q1, q2, dt).T + ljcdt.T

        # variable is the vector of unknowns
        variable = cs.vertcat(q2, eq_lambdas)

        p_residual = cs.vertcat(p_residual, self.eq_constraint(q2))

        p_residual = cs.Function(
            "P_residual",
            [variable, p, q1, dt],
            [p_residual],
            ["[q+1, lambdas]", "p", "q", "dt"],
            ["P_residual"],
        )

        # p_next = D2L(q1, q2)
        return p_residual, d2l
