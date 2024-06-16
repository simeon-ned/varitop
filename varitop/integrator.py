import logging

backends = []
try:
    import jax
    from jaxopt import ScipyRootFinding

    logging.getLogger().setLevel(logging.WARNING)

    backends.append('jax')
except ImportError:
    pass


def midpoint(q1, q2, dt):
    return (q1 + q2) / 2, (q2 - q1) / dt


class DelIntegrator:

    def __init__(self, rule=midpoint):
        self.interpolate = rule
        self._lagrangian = None
        self._ld = None
        self._dld1 = None
        self._dld2 = None
        self._residual = None

    @property
    def backend(self):
        """The backend property."""
        return self._backend

    @backend.setter
    def backend(self, value):
        if value in backends:
            self._backend = value
        else:
            raise RuntimeError(f'Backend not avaiable: {value}. Avaiable backends are: {backends}')

    @property
    def residual(self):
        """The reisudal property."""
        if self._residual is None:
            def _residual_fun(q3, q1, q2, dt):
                dld1 = self.dld1
                dld2 = self.dld2
                return dld2(q1, q2, dt) + dld1(q2, q3, dt)
            self._residual = _residual_fun

        return self._residual

    def step(self, q1, q2, dt):
        if self.backend == 'jax':
            # bn = Broyden(fun=self.residual, verbose=False)
            # sol = bn.run(init_params=q1, q1=q1, q2=q1, dt=dt)
            sc = ScipyRootFinding(method="lm", optimality_fun=self.residual)
            sol = sc.run(init_params=q1, q1=q1, q2=q2, dt=dt)
            return sol.params

    @residual.setter
    def residual(self, value):
        self._residual = value

    @property
    def lagrangian(self):
        """The lagrangian property."""
        return self._lagrangian

    @lagrangian.setter
    def lagrangian(self, value):
        self._lagrangian = value

    def _derivate(self, function, argnum):
        if self.backend == 'jax':
            return jax.grad(function, argnums=argnum)

    @property
    def ld(self):
        """The ld property."""
        if self._ld is None:
            def _ld_function(q1, q2, dt):
                q, dq = self.interpolate(q1, q2, dt)
                return self.lagrangian(q, dq)
            self._ld = _ld_function
        return self._ld

    @ld.setter
    def ld(self, value):
        self._ld = value

    @property
    def dld1(self):
        """The ld1 property."""
        if self._dld1 is None:
            self._dld1 = self._derivate(self.ld, 0)
        return self._dld1

    @dld1.setter
    def dld1(self, value):
        self._dld1 = value

    @property
    def dld2(self):
        """The ld2 property."""
        if self._dld2 is None:
            self._dld2 = self._derivate(self.ld, 1)
        return self._dld2

    @dld2.setter
    def dld2(self, value):
        self._dld2 = value
