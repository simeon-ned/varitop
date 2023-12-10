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
        return cs.Function("L", [q1, q2, dt], [self.lagrangian(q, dq)])

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

        # First slot derivative
        D1L = cs.Function(
            "D1L", [q1, q2, dt], [cs.jacobian(discrete_lagrangian(q1, q2, dt), q1)]
        )
        # Second slot derivative
        D2L = cs.Function(
            "D2L", [q1, q2, dt], [cs.jacobian(discrete_lagrangian(q1, q2, dt), q2)]
        )
        # Discrete Euler-Lagrange residual
        DEL = cs.Function(
            "DEL",
            [q1, q2, q3, dt],
            [D1L(q1, q2, dt).T + D2L(q2, q3, dt).T],
            ["q-1", "q", "q+1", "dt"],
            ["DEL_residual"],
        )

        # Create the Lagrange multipliers
        eq_lambdas = cs.SX.sym("eq_lambda", len(self.eq_constraints))

        # For each of the constraints, create a residual
        for index, constraint in enumerate(self.eq_constraints):
            residual = cs.Function(
                f"eq_res_{index}", [q2], [cs.jacobian(constraint(q2), q2).T]
            )

            # Allow free is required to make it possible
            # for lambda to be a free from argumenting
            DEL = cs.Function(
                "DEL",
                [q1, q2, q3, dt],
                [DEL(q1, q2, q3, dt) + eq_lambdas[index] * residual(q2) * dt],
                ["q-1", "q", "q+1", "dt"],
                ["DEL_residual"],
                {"allow_free": True},
            )

        return DEL
