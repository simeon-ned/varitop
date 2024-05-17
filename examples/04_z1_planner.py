"""Temporary fix for importing modules from parent directory"""

import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from varitop.integrator import DelIntegrator as DI
from varitop.opti import Varitop
import casadi as cs
import numpy as np

from darli.model import Functional
from darli.backend import CasadiBackend

from robot_descriptions import z1_description

darli_model = Functional(CasadiBackend(z1_description.URDF_PATH))
darli_model.add_body({'ee': 'link06'})
nq = darli_model.nq
nv = darli_model.nv
nu = darli_model.nu

di = DI(
    nq = nq,
    nu = nu,
    lagrangian = darli_model.lagrangian,
)
di.add_force(di.u)

ns = 150  # number of shooting nodes
tf = 1.5  # [s]
dt = tf/ns

vp = Varitop(
    integrator=di,
    nsteps=ns,
    dt=dt,
)

initial_configuration = vp.parameter(6)
desired_pos = vp.parameter(3)

q = vp.q
u = vp.u

# initial point (joint space)
vp.subject_to(q[:, 0] == initial_configuration)
# vp.set_initial_configuration(initial_configuration)
vp.subject_to(q[:, 0] == q[:, 1])
# Terminal point (cartesian space)
vp.subject_to(darli_model.body('ee').position(q[:, -1]) == desired_pos)
vp.subject_to(q[:, -1] == q[:, -2])

# cost
cost = 0 

# stage cost 
for i in range(ns-1):
    cost += 1e-2*cs.sumsqr(q[:, i])*dt
    cost += 5e-2*cs.sumsqr((q[:, i + 1] - q[:, i])/dt)*dt
    cost += 5e-3*cs.sumsqr(u[:, i])*dt
    
# Terminal cost 
cost += 5e-2*cs.sumsqr(q[:, -1])
cost += 10e-2*cs.sumsqr((q[:, -1] - q[:, -2])/dt)

vp.set_cost(cost)

# Limits 
# joint limits 
vp.bounded(darli_model.q_min, q, darli_model.q_max)
# velocity limits
v_max = 1.5
qdot_lims = np.full((nv), v_max)
vp.bounded(-qdot_lims,(q[:, 1:] - q[:, :-1])/dt,  qdot_lims)
# torque limits
u_max = 5
u_lims = np.full((nu), u_max) 
vp.bounded(-u_max, u,  u_max)

vp.set_parameter(initial_configuration,np.zeros(6))
vp.set_parameter(desired_pos,[0.2, 0.2, 0.36])

sol = vp.solve()