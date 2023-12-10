"""Variables are here"""
import casadi as cs


class Variable:
    """Varaibles are here"""

    def __init__(self, name: str, dim: int, active: list[int]):
        self._name: str = name
        self._dim: int = dim
        self._active: list[int] = active
        self._nodes: int = len(active)

        self._var_prj = cs.SX.sym(name, dim, self._nodes)

    @property
    def shape(self):
        return [self._dim, 1]

    @property
    def var(self):
        """Variable"""
        if self._nodes > 0:
            return self[0]

    def __getitem__(self, key):
        return self._var_prj[:, key]


class Momentum(Variable):
    """Momentum is here"""

    def __init__(self, name: str, dim: int, active: list[int]):
        super().__init__(name, dim, active)


class State(Variable):
    """State is here"""

    def __init__(self, name: str, dim: int, active: list[int]):
        super().__init__(name, dim, active)


class Velocity(Variable):
    """Velocity is here"""

    def __init__(self, name: str, dim: int, active: list[int]):
        super().__init__(name, dim, active)


class Control(Variable):
    """Control is here"""

    def __init__(self, name: str, dim: int, active: list[int]):
        super().__init__(name, dim, active)
