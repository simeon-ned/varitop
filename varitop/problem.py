"""Problem module"""
from typing import Dict, Callable
import numpy as np
from .variables import Variable, Momentum, State, Velocity, Control


class VaritopProblem:
    """Problem is going to be here"""

    def __init__(self):
        self._variables: Dict[str, Variable] = {}
        self._nodes: int = 0

    @property
    def nodes(self) -> int:
        """Set number of nodes"""
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: int):
        """Set number of nodes"""
        self._nodes = nodes

    def create_variable(
        self, variable: type[Variable], name: str, dim: int, active: list[int] = None
    ) -> Variable:
        """Create a variable"""
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
