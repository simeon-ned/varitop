"""Anything unsorted for now goes here."""


def euler_quadrature(q1, q2, dt):
    return (q1 + q2) / 2, (q2 - q1) / dt
