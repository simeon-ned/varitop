import casadi as cs
from darli.backend import CasadiBackend
from darli.modeling.functional import Functional


class Model:
    valid_backends = ["darli"]

    def __init__(self, urdf_path: str):
        self._nq: int = None
        self._nv: int = None
        self._nu: int = None
        self._lagrangian: cs.Function = None
        self._urdf_path: str = urdf_path

    @property
    def nq(self) -> int:
        return self._nq

    @property
    def nv(self) -> int:
        return self._nv

    @property
    def nu(self) -> int:
        return self._nu

    @property
    def lagrangian(self) -> cs.Function:
        return self._lagrangian

    def create_data(self):
        raise NotImplementedError


class DarliModel(Model):
    def create_data(self):
        model = Functional(CasadiBackend(self._urdf_path))

        self._nq = model.nq
        self._nv = model.nv
        self._nu = model.nu
        self._lagrangian = model.lagrangian
