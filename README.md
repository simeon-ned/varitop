# Varitop 
Varitop (stands for **var**iational **i**ntegration **t**rajectory **op**timization) is a Python library for variational integration-based optimal motion planning. The library inclined towards robotic applications namely articulated robots and use 

### Variational Integration Based Planning
Variational integrators are numerical methods used to approximate the dynamics of continuous systems while preserving important geometric and structural properties. They minimize a variational action functional defined by the Lagrangian function $\mathcal{L}(q, \dot{q})$.

To discretize the system's behavior, the action functional is approximated over a finite time interval $[t_0, t_1]$ divided into $N$ subintervals with time step $\Delta t$. The discrete action functional is in some way discritized i.e as Euler quadrature:


$$S(q, \dot{q}) = \sum_{k=0}^{N-1} \mathcal{L}(q_k, \dot{q}_k) \Delta t$$

Solving these equations iteratively over each time step allows for numerical integration which preserve symplecticity, energy conservation, and the system's geometric structure and thus make it extremelly beneficial in optimal control and motion planning

<!-- Optimal Trajectory Planning (Finite horizon OCP) with Variational integrators -->

### Basic Example

```python
import casadi as cs
from varitop import VaritopProblem 
from varitop.solvers import Solver
from varitop.action_quadrature import euler

nodes = 50
prob = VaritopProblem(nodes)

# Add the configuration variables 
q = prob.state_variable('q', nq)
v = prob.state_variable('v', nv)
u = prob.control_variable('u', nu)

# Prepare variational integrator
# Create a lagrangian as function of the q, dq 
lagrangian = ...

varitop.set_continues_lagrangian(lagrangian)
varitop.set_quadrature('euler')
# One can get the resulting discrite lagrangian and integrator
# which can be used later to integrate the dynamics
discrete_lagrangian = varitop.get_discrete_lagrangian()
residual_constraints = varitop.get_residual()
variational_integrator = varitop.get_integrator()
prob.set_dynamics(variational_integrator)

# add cost
prob.add_intermediate_cost('min_effort', cs.sumsqr(u))
prob.add_intermediate_cost('min_velocity', cs.sumsqr(dq))
prob.add_terminal_cost('min_velocity', cs.sumsqr(dq))

# add constraints
prob.add_constraint('eq_const', )
prob.add_initial_constraint('init_cons', g(q))
prob.add_terminal_constraint('term_const', h(q))
prob.add_intermediate_constraint('term_const', q[0], lb = 0)

solver = Solver.make_solver(prb, 'ipopt', options)
# set initial solution
solver.solve()
solution = solver.solution_dict()

q_opt = solution['q']
dq_opt = solution['dq']
u_opt = solution['u']
```



<!-- ### Planning for Articulated Robots

```python

```

### Variational Integration

```python
 -->
<!-- ``` -->
