"""Temporary fix for importing modules from parent directory"""

from darli.robots import biped
from darli.backend import CasadiBackend, JointType
from darli.model import Functional
from robot_descriptions.bolt_description import URDF_PATH
import pinocchio as pin
from time import sleep
from tqdm import tqdm
import mediapy as media
import casadi as cs
import numpy as np
import sys
import os

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from varitop.integrator import DelIntegrator as DI
from varitop.opti import Varitop
from varitop.misc import quat_prod
import matplotlib.pyplot as plt
import meshcat

# VISUALISATION
pinmodel, gpinmodel, cpinmodel = pin.buildModelsFromUrdf(
    URDF_PATH,
    package_dirs="/home/m8dotpie/.cache/robot_descriptions",
    geometry_types=[pin.GeometryType.VISUAL, pin.GeometryType.COLLISION],
    root_joint=pin.JointModelFreeFlyer(),
)

mvis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
vis = pin.visualize.MeshcatVisualizer(pinmodel, gpinmodel, cpinmodel)
vis.initViewer(loadModel=True, viewer=mvis)
vis.display(pin.neutral(pinmodel))

# Building dynamics

darli_model = biped(
    Functional,
    CasadiBackend,
    URDF_PATH,
    torso={"torso": "base_link"},
    foots={"FL_FOOT": "FL_FOOT", "FR_FOOT": "FR_FOOT"},
    root_joint=JointType.FREE_FLYER,
)

darli_model.body("FL_FOOT").add_contact(contact_type="point")
darli_model.body("FL_FOOT").contact.add_cone(mu=1.2)
cwcl = darli_model.body("FL_FOOT").contact.cone.linear()
darli_model.body("FR_FOOT").add_contact(contact_type="point")
darli_model.body("FR_FOOT").contact.add_cone(mu=1.2)
cwcr = darli_model.body("FR_FOOT").contact.cone.linear()

# Creating Integrator

nq = darli_model.nq
nv = darli_model.nv
nu = darli_model.nu

di = DI(
    nq=nq,
    nu=nu,
    free_body=True,
    lagrangian=darli_model.lagrangian,
)

di.add_force(cs.vertcat(np.zeros(7), di.u))
di.add_constraint(di.q[3:7].T @ di.q[3:7] - 1)

# Formulating optimisation problem

nsteps = 100
dt = 0.01
step_length = 0.2
q_init = pin.neutral(pinmodel)

vp = Varitop(integrator=di, nsteps=nsteps, dt=dt, custom_dynamics=True)

# Contact forces acting on left leg
f = vp.problem.variable(3, nsteps)

# Dynamics residual constraint
del_res = vp.integrator.get_residual()
for i in range(1, nsteps):
    qforce = darli_model.contact_qforce(vp.q[:, i], f[:, i], np.zeros(3))

    qft = qforce[:3]
    qrt = qforce[3:6]
    qrest = qforce[6:]
    qrt = 2 * quat_prod(vp.q[3:7, i], cs.vcat([qrt, 0]))
    qforce = cs.vertcat(qft, qrt, qrest, 0)
    vp.subject_to(vp.residual(i) + qforce == 0)

# Friction cone
for i in range(1, nsteps):
    lb = np.full(5, -np.inf)
    ub = np.zeros(5)
    vp.bounded(lb, cwcl[:, :] @ f[:, i], ub)

    lb = 0
    ub = np.inf
    vp.bounded(lb, f[2, i], ub)


def getAngle(P, Q):
    R = P@Q.T
    cos_theta = (cs.trace(R)-1)/2
    return cos_theta


# Left leg constraints

lfoot = darli_model.body("FL_FOOT")
lfoot_init_pos = np.array(lfoot.position(q_init)).copy()
lfoot_init_rot = np.array(lfoot.rotation(q_init)).copy()
lfoot_init_q = vp.q[:, 0]
lfoot_init_dq = (vp.q[:, 1] - vp.q[:, 0]) / vp.dt
lfoot_init_v = vp.integrator.v(lfoot_init_q, lfoot_init_dq)
lfoot_init_vel_tr = lfoot.linear_velocity.world_aligned(lfoot_init_q, lfoot_init_v)
lfoot_init_vel_rt = lfoot.angular_velocity.world_aligned(lfoot_init_q, lfoot_init_v)

# Lfoot velocity is fixed
# for i in range(1, vp.ns):
#     qi = vp.q[:, i - 1]
#     dqi = (vp.q[:, i] - vp.q[:, i - 1]) / vp.dt
#     vi = vp.integrator.v(qi, dqi)
#     lfoot_vel_tr = lfoot.linear_velocity.world_aligned(qi, vi)
#     lfoot_vel_rt = lfoot.angular_velocity.world_aligned(qi, vi)
#     vp.subject_to(lfoot_vel_tr == 0)
#     vp.subject_to(lfoot_vel_rt == 0)

vp.subject_to(vp.q[:, 0] - vp.q[:, 1] == 0)

vp.subject_to(getAngle(lfoot.rotation(vp.q[:, 0]), lfoot_init_rot) - 1 == 0)
vp.subject_to(getAngle(lfoot.rotation(vp.q[:, -1]), lfoot_init_rot) - 1 == 0)

for i in range(0, vp.ns):
    qi = vp.q[:, i]
    # Initial position and rotation is fixed
    vp.subject_to(lfoot.position(qi) - lfoot_init_pos == 0)
    # vp.subject_to(getAngle(lfoot.rotation(qi), lfoot_init_rot) - 1 == 0)

# Right leg constraints
rfoot = darli_model.body("FR_FOOT")
rfoot_init_pos = np.array(rfoot.position(q_init))[:, 0].copy() + np.array([-step_length / 2, 0, 0])
rfoot_end_pos = np.array(rfoot.position(q_init))[:, 0].copy() + np.array([step_length / 2, 0, 0])
rfoot_init_rot = np.array(rfoot.rotation(q_init)).copy()
rfoot_init_q = vp.q[:, 0]
rfoot_init_dq = (vp.q[:, 1] - vp.q[:, 0]) / vp.dt
rfoot_init_v = vp.integrator.v(rfoot_init_q, rfoot_init_dq)
rfoot_end_q = vp.q[:, -2]
rfoot_end_dq = (vp.q[:, -1] - vp.q[:, -2]) / vp.dt
rfoot_end_v = vp.integrator.v(rfoot_end_q, rfoot_end_dq)
rfoot_init_vel_tr = rfoot.linear_velocity.world_aligned(rfoot_init_q, rfoot_init_v)
rfoot_init_vel_rt = rfoot.angular_velocity.world_aligned(rfoot_init_q, rfoot_init_v)
rfoot_end_vel_tr = rfoot.linear_velocity.world_aligned(rfoot_end_q, rfoot_end_v)
rfoot_end_vel_rt = rfoot.angular_velocity.world_aligned(rfoot_end_q, rfoot_end_v)

# Initial position and rotation
vp.subject_to(rfoot.position(vp.q[:, 0]) - rfoot_init_pos == 0)
vp.subject_to(getAngle(rfoot.rotation(vp.q[:, 0]), rfoot_init_rot) - 1 == 0)
# Final position and rotation
vp.subject_to(rfoot.position(vp.q[:, -1]) - rfoot_end_pos == 0)
vp.subject_to(getAngle(rfoot.rotation(vp.q[:, -1]), rfoot_init_rot) - 1 == 0)

# Initial velocity is fixed
vp.subject_to(rfoot_init_vel_tr == 0)
vp.subject_to(rfoot_init_vel_rt == 0)
# Terminal velocity is fixed
vp.subject_to(rfoot_end_vel_tr == 0)
vp.subject_to(rfoot_end_vel_rt == 0)

# No rotation in foot
for i in range(1, vp.ns):
    qi = vp.q[:, i - 1]
    dqi = (vp.q[:, i] - vp.q[:, i - 1]) / vp.dt
    vi = vp.integrator.v(qi, dqi)
    rfoot_vel_tr = rfoot.linear_velocity.world_aligned(qi, vi)
    rfoot_vel_rt = rfoot.angular_velocity.world_aligned(qi, vi)
    rfoot_pos = rfoot.position(qi)
    # no rotational motion
    # vp.subject_to(rfoot_vel_rt == 0)
    # no Y direction motion
    vp.subject_to(rfoot_vel_tr[1] == 0)
    # X velocity is non-negative
    vp.bounded(0, rfoot_vel_tr[0], np.inf)
    # Rfoot height constraint
    vp.bounded(rfoot_init_pos[2], rfoot_pos[2], rfoot_init_pos[2] + 0.2)

# Swing middle height
rfoot_middle_pos = rfoot.position(vp.q[:, nsteps // 2])
vp.bounded(rfoot_init_pos[2] + 0.05, rfoot_middle_pos[2], np.inf)
vp.subject_to(rfoot_middle_pos[0] == 0)

torso = darli_model.body('torso')
vp.subject_to(torso.position(vp.q[:, 0])[0] == 0)
vp.subject_to(torso.position(vp.q[:, -1])[0] - rfoot.position(vp.q[:, -1])[0] == 0)
# Torso
for i in range(0, vp.ns):
    qi = vp.q[:, i]
    torso = darli_model.body('torso')
    torso_pos = torso.position(qi)
    torso_rot = torso.rotation(qi)
    torso_angle = getAngle(torso_rot, np.array(torso.rotation(q_init)).copy())
    vp.subject_to(torso_angle - 1 == 0)
    vp.bounded(-0.1, torso_pos[2], 0.1)

# Initial configuration guess
q_init = pin.neutral(pinmodel)
for i in range(0, nsteps):
    vp.set_initial(vp.q[:, i], q_init)

# Cost function
cost = 0

for i in range(1, nsteps):
    # cost += vp.dt * (vp.u[:, i].T @ vp.u[:, i])
    qi = vp.q[:, i]

    # torso = darli_model.body("torso")
    # torso_pos = torso.position(qi)
    # torso_rot = torso.rotation(qi)
    # torso_angle = getAngle(torso_rot, torso.rotation(q_init))
    # cost += 1e3 * torso_angle**2
    # cost += 1e4 * torso_pos[2]**2
    #
    # lfoot = darli_model.body("FL_FOOT")
    # lfoot_pos = lfoot.position(qi)
    # lfoot_rot = lfoot.rotation(qi)
    # lfoot_angle = getAngle(lfoot_rot, lfoot.rotation(q_init))
    # cost += 1e3 * lfoot_angle**2
    #
    # rfoot = darli_model.body("FR_FOOT")
    # rfoot_pos = rfoot.position(qi)
    # rfoot_rot = rfoot.rotation(qi)
    # rfoot_angle = getAngle(rfoot_rot, rfoot.rotation(q_init))
    # cost += 1e3 * rfoot_angle**2
    # cost += rfoot_pos[1]**2

vp.set_cost(cost)

try:
    sol = vp.solve()
    print('Solver converged')
    qs = sol.value(vp.q)
except Exception:
    print('Solver diverged, using debug values')
    qs = vp.problem.debug.value(vp.q)

rfoot.position(qs[:, 0]), rfoot.position(qs[:, -1])
rfootpos = np.array([rfoot.position(q) for q in qs.T]).squeeze()


def plot_foot_pos(init=rfoot_init_pos, end=rfoot_end_pos):
    fig, ax = plt.subplots()
    ax.plot(rfootpos, label=["x", "y", "z"])
    ax.scatter(np.zeros(3), np.array(init))
    ax.scatter(np.full(3, 100), np.array(end))
    plt.legend()
    plt.show()


def visualize(qs=qs, dt=dt * 10, pause=2):
    sleep(pause)
    for i in range(len(qs.T)):
        vis.display(qs[:, i])
        sleep(dt)
