"""Problem module"""
from typing import Dict, List
import casadi as cs
import numpy as np

from varitop.integrator import VariationalIntegrator
from .variables import Variable, Momentum, State, Velocity, Control


class VaritopProblem:
    """Problem is going to be here"""

    def __init__(self):
        self._variables: Dict[str, Variable] = {}
        self._nodes: int = None
        self._nq: int = None
        self._nv: int = None
        self._nu: int = None

        self._rule: cs.Function = None
        self._dynamics_constraints: List[cs.Function] = []
        self._forces: List[cs.Function] = []
        self._dynamics: cs.Function = None
        self._integrator: VariationalIntegrator = None
        self._free: bool = False

    @property
    def free(self) -> bool:
        """Get free"""
        return self._free

    @free.setter
    def free(self, free: bool):
        """Set free"""
        self._free = free

        if self._integrator is not None:
            self.integrator.free = free

    @property
    def rule(self) -> cs.Function:
        """Get rule"""
        return self._rule

    @rule.setter
    def rule(self, rule: cs.Function):
        """Set rule"""
        self._rule = rule

        if self._integrator is not None:
            self._integrator.rule = rule

    @property
    def dynamics(self) -> cs.Function:
        """Get dynamics"""
        return self._dynamics

    @dynamics.setter
    def dynamics(self, dynamics: cs.Function):
        """Set dynamics"""
        self._dynamics = dynamics

    @property
    def integrator(self) -> VariationalIntegrator:
        """Get integrator"""
        return self._integrator

    @integrator.setter
    def integrator(self, integrator: VariationalIntegrator):
        """Setup the integrator"""

        if self.nq is None:
            raise RuntimeError("Number of generalized coordinates not set.")

        if self.dynamics is None:
            raise RuntimeError("Dynamics not set.")

        if self.rule is None:
            raise RuntimeError("Rule not set.")

        self._integrator = integrator()
        self._integrator.nq = self.nq
        self._integrator.nu = self.nu
        self._integrator.lagrangian = self.dynamics
        self._integrator.rule = self.rule
        self._integrator.free = self.free
        self._integrator.add_dynamics_constraints(self._dynamics_constraints)
        self._integrator.add_generalized_forces(self._forces)

    def add_dynamics_constraints(self, constraints: List[cs.Function]):
        """Add dynamics constraint"""
        self._dynamics_constraints.extend(constraints)

        if self._integrator is not None:
            self._integrator.add_dynamics_constraints(constraints)

    def add_forces(self, forces: List[cs.Function]):
        """Add forces"""
        self._forces.extend(forces)

        if self._integrator is not None:
            self._integrator.add_generalized_forces(forces)

    @property
    def nodes(self) -> int:
        """Set number of nodes"""
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: int):
        """Set number of nodes"""
        self._nodes = nodes

    @property
    def nq(self) -> int:
        """Set number of generalized coordinates"""
        return self._nq

    @nq.setter
    def nq(self, nq: int):
        """Set number of generalized coordinates"""
        self._nq = nq
        self.create_state("q", nq)

    @property
    def nv(self) -> int:
        """Set number of generalized velocities"""
        return self._nv

    @nv.setter
    def nv(self, nv: int):
        """Set number of generalized velocities"""
        self._nv = nv
        self.create_velocity("v", nv)

    @property
    def nu(self) -> int:
        """Set number of controls"""
        return self._nu

    @nu.setter
    def nu(self, nu: int):
        """Set number of controls"""
        self._nu = nu
        self.create_control("u", nu)

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

    @property
    def state(self) -> List[Variable]:
        """Get state variables"""
        return self._variables[State]

    @property
    def velocity(self) -> List[Variable]:
        """Get velocity variables"""
        return self._variables[Velocity]

    @property
    def control(self) -> List[Variable]:
        """Get control variables"""
        return self._variables[Control]
