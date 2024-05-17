"""Temporary fix for importing modules from parent directory"""

import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from varitop.integrator import DelIntegrator as DI
from varitop.opti import Varitop

import numpy as np
from time import sleep

from darli.model import Functional
from darli.backend import CasadiBackend
import casadi as cs

from robot_descriptions.skydio_x2_description import URDF_PATH

Q_INIT = np.array([0, 0, 1.0, 0, 0, 0,1])

model = Functional(CasadiBackend(URDF_PATH))
selector = np.array([[0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0],
                     [1.0, 1.0, 1.0, 1.0],
                     [-0.18, 0.18, 0.18, -0.18],
                     [0.14, 0.14, -0.14, -0.14],
                     [-0.0201, 0.0201, 0.0201, -0.0201]])

model.update_selector(selector)

di = DI(
    nq = model.nq,
    nu = 4,
    free_body = True,
    selector = selector,
    lagrangian=model.lagrangian,
)
di.add_constraint(di.q[3:].T @ di.q[3:] - 1)

vp = Varitop(integrator=di, nsteps=40)

tf = vp.parameter(1)
dt = tf / vp.ns
vp.dt = dt

q0 = vp.parameter(model.nq)
qf = vp.parameter(model.nq)

position_weight = vp.parameter(1)
acceleration_weight = vp.parameter(1)
velocity_weight = vp.parameter(1)
control_weight = vp.parameter(1)
control_max = vp.parameter(model.nu)

q = vp.q
u = vp.u
ns = vp.ns
dt = vp.dt

cost = 0 
for i in range(1, vp.ns):
    cost += position_weight*cs.sumsqr(q[:3,i] - qf[:3])*dt
    cost += control_weight*cs.sumsqr(u[:,i])*dt

for i in range(1, vp.ns):
    vp.bounded(np.zeros(model.nu), u[:, i], control_max)
    vp.subject_to(q[2, i] >= 0.4)
    
vp.set_initial_configuration(q0)
vp.set_terminal_configuration(qf)

vp.set_cost(cost)

Q_INIT = np.array([0, 0, 1.5, 0,0,0, 1])
quat_random = np.random.rand(4)
quat_random /= np.linalg.norm(quat_random)
Q_INIT[3:] = quat_random
Q_FINAL = np.array([0, 0., 2.5, 0, 0, 0, 1])
Q_FINAL[:3] += np.random.randn(3)

t_final = 3

vp.set_parameter(tf, t_final)
vp.set_parameter(control_max, 50*np.ones(model.nu) )
vp.set_parameter(q0, Q_INIT)
vp.set_parameter(qf, Q_FINAL)

vp.set_parameter(position_weight[0], 5)
vp.set_parameter(velocity_weight[0], 10)
vp.set_parameter(acceleration_weight[0], 5)
vp.set_parameter(control_weight[0], 5)

vp.set_initial(q[3:,:], np.array([Q_INIT[3:]]*(vp.ns+1)).T)
vp.set_initial(q[:3,:], np.linspace(Q_INIT[:3], Q_FINAL[:3], vp.ns+1).T)

ren.set_state(Q_INIT)
ren.markers[vp.ns+1](position=Q_FINAL[:3],
                color=[1, 0, 0, 0.5],
                size=0.04)
ren.markers[vp.ns+2](position=Q_INIT[:3],
                color=[0, 1, 0, 0.5],
                size=0.04)


sol = vp.solve()