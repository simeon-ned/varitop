Variational Integrators
=======================

General Definitions
-------------------

:Variational Integrator:
    VI is a particular example of symplectic integrator.

:Sympeltic Integrator:
    A special case of geometric integrator, designed for solving Hamilton equations. 
    Since time evolution of Hamilton equations is symplectomorphism, integrator
    preserves physical properties of the system (flow of differential equation).
    Consequently, it can preserve system energy, momentum in a long term run.

:Hamilton equations:
    :math:`\dot{p} = \frac{\partial H}{\partial q}` and :math:`\dot{q} = \frac{\partial H}{\partial p}`

:Symplectomorphism:
    Process which conserves 2-form, for instance, :math:`\mathrm{d}q \wedge \mathrm{d}p`

Derivation
----------

Hamilton's Principle
~~~~~~~~~~~~~~~~~~~~

Everything starts with *Hamilton's Principle* also called *Least Action Principle* or *Stationary Action Principle*.

:Least Action Principle:
    .. math::
        S(\mathbf{q}) = \int_{t_0}^{t_f} L(\mathbf{q}, \dot{\mathbf{q}}, t) \mathrm{d}t

:Lagrangian:
    .. math::
        L(\mathbf{q}, \dot{\mathbf{q}}, t) = K(\mathbf{q}, \dot{\mathbf{q}}) - V(\mathbf{q})

    Where :math:`K` is kinetic energy and :math:`V` is potential energy of the system.

Mathematically, we can say that system evolution obeys the following equation:

.. math:: 
    \frac{\delta S}{\delta \mathbf{q}(t)} = 0

Discretization
~~~~~~~~~~~~~~

To avoid variational calculus, we can try to directly discretise the system.

.. math:: 
    S(\mathbf{q}) = \sum_{k=1}^{N} \int_{t_k}^{t_{k+1}}L(\mathbf{q}, \dot{\mathbf{q}}, t) \mathrm{d}t

Using some midpoint rule for discretisation, in particular, Euler rule, we can arrive to:

.. math:: 
    \int_{t_k}^{t_{k+1}}L(\mathbf{q}, \dot{\mathbf{q}}, t) \mathrm{d}t \approx L(\frac{\mathbf{q}_{k + 1} + \mathbf{q}_{k}}{2}, \frac{\mathbf{q}_{k + 1} - \mathbf{q}_k}{\Delta t}, \frac{t_{k + 1} + t_k}{2}) \Delta t

Further on this formulation we will call *Discrete Lagrangian*

:Discrete Lagrangian:
    .. math::
        L_d(\mathbf{q}_k, \mathbf{q}_{k + 1}, \Delta t)

Action functional can be reformulated like following:

.. math:: 
    S(\mathbf{q}_0, \mathbf{q}_1, \ldots, \mathbf{q}_{n - 1}) = \sum_{k=0}^{n - 1}L_d(\mathbf{q}_k, \mathbf{q}_{k + 1}, \Delta t)

Moreover, we can reformulate action variation as well, leaving only terms containing the step variable:

.. math:: 
    \frac{\delta S}{\delta \mathbf{q}_k} = \frac{L_d(\mathbf{q}_k, \mathbf{q}_{k+1}, \Delta t)}{\delta \mathbf{q}_k} + \frac{L_d(\mathbf{q}_{k+1}, \mathbf{q}_{k+2}, \Delta t)}{\delta \mathbf{q}_k}

:Slot Derivative:
    .. math::
        D_1L_d(\mathbf{q}_1, \mathbf{q}_2) = \frac{\delta L_d}{\delta \mathbf{q}_1}
    .. math::
        D_2L_d(\mathbf{q}_1, \mathbf{q}_2) = \frac{\delta L_d}{\delta \mathbf{q}_2}

This way we have a convenient formulation of action variation:

.. math:: 
    \frac{\delta S}{\delta \mathbf{q}_k} = D_2L_d(\mathbf{q}_k, \mathbf{q}_{k + 1}) + D_1L_d(\mathbf{q}_{k + 1}, \mathbf{q}_{k + 2})

Discrete Euler-Lagrange Residual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Final idea is very simple. Given that the variation on each time step is zero, we can rewrite previous equation and call it `residual`:

.. math:: 
    D_2L_d(\mathbf{q}_k, \mathbf{q}_{k + 1}) + D_1L_d(\mathbf{q}_{k + 1}, \mathbf{q}_{k + 2}) = 0

Which can be solved for :math:`\mathbf{q}_{k+2}` given two previous states.

Constrained Dynamics
--------------------

External Forcing
----------------

Discrete Euler-Lagrange in Momentum Form
----------------------------------------