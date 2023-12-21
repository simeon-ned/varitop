"""Anything unsorted for now goes here."""

import casadi as cs


def skew_quaternion(q1):
    """Skew symmetric quaternion matrix"""
    L = cs.vertcat(
        cs.horzcat(q1[0], -q1[1], -q1[2], -q1[3]),
        cs.horzcat(q1[1], q1[0], -q1[3], q1[2]),
        cs.horzcat(q1[2], q1[3], q1[0], -q1[1]),
        cs.horzcat(q1[3], -q1[2], q1[1], q1[0]),
    )

    return L


def qconj(q):
    """Quaternion conjugate"""
    return cs.vcat([q[0], -q[1], -q[2], -q[3]])


def euler_rule(q1, q2, dt):
    """Euler rule for integration"""
    return (q1 + q2) / 2, (q2 - q1) / dt
