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
        self._nv: int = None
        self._nu: int = None

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
        self.create_state("q", nq)

    @property
    def nv(self) -> int:
        """Number of generalized velocities"""
        return self._nv

    @nv.setter
    def nv(self, nv: int):
        self._nv = nv
        self.create_velocity("v", nv)

    @property
    def nu(self) -> int:
        """Number of controls"""
        return self._nu

    @nu.setter
    def nu(self, nu: int):
        self._nu = nu
        self.create_control("u", nu)

    def create_variable(
        self, variable: type[Variable], name: str, dim: int, active: list[int] = None
    ) -> Variable:
        """Create a variable
        
        :param variable: which variable to create
        :param name: varaible name
        :param dim: variable dimensionality
        :param active: at which nodes variable is active
        :type variable: type[Variable]
        :type name: str
        :type dim: int
        :type active: list[int]"""
        if active is None:
            active = np.ones(self.nodes, dtype=int)

        self._variables[variable] = variable(name, dim, active)
        return self._variables[variable]

    def create_state(self, name: str, dim: int, active: list[int] = None) -> Variable:
        """Create a state variable"""
        return self.create_variable(State, name, dim, active)

    def create_velocity(self, name: str, active: list[int] = None) -> Variable:
        """Create a velocity variable"""
        return self.create_variable(Velocity, name, active)

    def create_momentum(self, name: str, active: list[int] = None) -> Variable:
        """Create a momentum variable"""
        return self.create_variable(Momentum, name, active)

    def create_control(self, name: str, active: list[int] = None) -> Variable:
        """Create a control variable"""
        return self.create_variable(Control, name, active)

    @property
    def state(self) -> Variable:
        """System state (projected)"""
        return self._variables[State]

    @property
    def velocity(self) -> Variable:
        """System velocty (projected)"""
        return self._variables[Velocity]

    @property
    def control(self) -> Variable:
        """System control (projected)"""
        return self._variables[Control]
