from varitop.integrator import VariationalIntegrator
from casadi import Opti
import casadi as cs

class Varitop:
    def __init__(self, integrator: VariationalIntegrator = None, nsteps: int = None, dt: float = None, custom_dynamics: bool = False):
        """
            Optimization class
        """

        if integrator is None:
            raise ValueError("Integrator must be provided")
        
        if nsteps is None:
            raise ValueError("Number of steps must be provided")

        self.integrator = integrator
        self.problem = Opti()
        self.ns = nsteps
        self.dt = dt
        self.custom_dynamics = custom_dynamics

        self.create_variables()

    @property
    def ns(self):
        return self._ns
    
    @ns.setter
    def ns(self, value):
        self._ns = value

    @property
    def dt(self):
        return self._dt
    
    @dt.setter
    def dt(self, value):
        self._dt = value

    def create_variables(self):
        self.q = self.problem.variable(self.integrator.nq, self.ns+1)
        self.dq = self.problem.variable(self.integrator.nq, self.ns)
        self.u = self.problem.variable(self.integrator.nu, self.ns)
        self.lmbds = self.problem.variable(self.integrator.lmbds.shape[0], self.ns)

    def parameter(self, shape: int):
        return self.problem.parameter(shape)
    
    def set_parameter(self, parameter: cs.SX, value):
        self.problem.set_value(parameter, value)

    def subject_to(self, expr):
        self.problem.subject_to(expr)

    def set_cost(self, cost):
        self.problem.minimize(cost)

    def set_initial(self, variable: cs.SX, value: cs.SX):
        self.problem.set_initial(variable, value)

    def residual(self, step):
        args = [
            self.q[:, step - 1], 
            self.q[:, step], 
            self.q[:, step + 1], 
            self.dt
        ]

        if self.integrator._constrained:
            args.append(self.lmbds[:, step])

        if self.integrator._forced:
            args.append(self.u[:, step])

        return self.integrator.get_residual()(*args)

    def _dynamics_constraint(self):
        for i in range(1, self.ns):
            self.subject_to(
                self.residual(i) == 0
            )

    def bounded(self, lb, variable, ub):
        self.problem.subject_to(
            self.problem.bounded(lb, variable, ub)
        )

    def solve(self):
        if not self.custom_dynamics:
            self._dynamics_constraint()

        self.problem.solver('ipopt')

        return self.problem.solve()
    
    def value(self, variable: cs.SX):
        return self.problem.value(variable)

    def set_initial_configuration(self, q0):
        self.subject_to(self.q[:, 0] == q0)

    def set_terminal_configuration(self, qf):
        self.subject_to(self.q[:, -1] == qf)