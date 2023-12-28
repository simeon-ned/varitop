"""Module utility functions"""

import casadi as cs
import numpy as np


def q2rm(q: np.ndarray) -> np.ndarray:
    """Quaternion to rotation matrix

    :param q: quaternion
    :type q: np.ndarray
    :return: rotation matrix
    :rtype: np.ndarray
    """
    q = np.array(q)
    q /= np.linalg.norm(q)
    w, x, y, z = q

    rotation_matrix = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
        ]
    )

    return rotation_matrix


def skew_quaternion(q1: cs.SX) -> cs.SX:
    """Skew symmetric quaternion matrix

    :param q1: quaternion to make matrix from
    :type q1: casadi.SX
    :return: L(q)
    :rtype: casadi.SX"""
    L = cs.vertcat(
        cs.horzcat(q1[0], -q1[1], -q1[2], -q1[3]),
        cs.horzcat(q1[1], q1[0], -q1[3], q1[2]),
        cs.horzcat(q1[2], q1[3], q1[0], -q1[1]),
        cs.horzcat(q1[3], -q1[2], q1[1], q1[0]),
    )

    return L


def qconj(q: cs.SX) -> cs.SX:
    """Quaternion conjugate

    :param q: quaternion to conjugate
    :type q: casadi.SX
    :return: q*
    :rtype: casadi.SX"""
    return cs.vcat([q[0], -q[1], -q[2], -q[3]])


def euler_rule(q1: cs.SX, q2: cs.SX, dt: cs.SX) -> [cs.SX, cs.SX]:
    """Euler midpoint rule for integration

    :param q1: first state
    :param q2: second state
    :param dt: time step
    :type q1: casadi.SX
    :type q2: casadi.SX
    :type dt: casadi.SX
    :return: q, dq
    :rtype: Tuple[casadi.SX, casadi.SX]"""
    return (q1 + q2) / 2, (q2 - q1) / dt
