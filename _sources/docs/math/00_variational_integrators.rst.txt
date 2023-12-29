Variational Integrators
=======================

General Definitions
-------------------

Variational Integrator :cite:p:`matthewwestVariationalIntegrators2004`:
    VI is a particular example of symplectic integrator.

Sympeltic Integrator:
    A special case of geometric integrator, designed for solving Hamilton equations. 
    Since time evolution of Hamilton equations is symplectomorphism, integrator
    preserves physical properties of the system (flow of differential equation).
    Consequently, it can preserve system energy, momentum in a long term run.

Hamilton equations:
    :math:`\dot{p} = \frac{\partial H}{\partial q}` and :math:`\dot{q} = \frac{\partial H}{\partial p}`

Symplectomorphism:
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

Straightforward Approach
~~~~~~~~~~~~~~~~~~~~~~~~

We can extend action, incorporating constraints via Lagrange multipliers:

.. math:: 
    S = \int_{t_0}^{t_f}(L + \lambda^T\phi(\mathbf{q}))\mathrm{d}t

    \text{s.t. } \phi(\mathbf{q}) = 0

Using the very same derivations we can arrive to the following:

.. math:: 
    \begin{cases}
    D_2L_d(\mathbf{q}_k, \mathbf{q}_{k + 1}) + D_1L_d(\mathbf{q}_{k + 1}, \mathbf{q}_{k + 2}) + \frac{\delta \phi}{\delta \mathbf{q}_{k+1}}^T(\mathbf{q}_{k+1})\lambda_{k+1} \mathrm{d}t= 0\\
    \phi(\mathbf{q}_{k+2}) = 0
    \end{cases}

Please note, that it is not necessary to have :math:`\mathrm{d}t` term near :math:`\lambda`. 
However, for convenience it is better to apply scaling and have properly defined equation.

Augmented Lagrangian
~~~~~~~~~~~~~~~~~~~~

We can notice, that if we define augmented lagrangian to be:

.. math:: 
    L_{\text{aug}}(\mathbf{q}_k, \mathbf{q}_{k+1}, \Delta t) = L(\frac{\mathbf{q}_{k + 1} + \mathbf{q}_{k}}{2}, \frac{\mathbf{q}_{k + 1} - \mathbf{q}_k}{\Delta t}, \frac{t_{k + 1} + t_k}{2}) \Delta t + \phi(\mathbf{q}_{k})\lambda_{k}\Delta t

Then taking slot derivatives as previously will yield the following result:

.. math:: 
    \begin{align}
    D_2L_{\text{aug}}(\mathbf{q}_{k}, \mathbf{q}_{k+1}) &= D_2L_d(\mathbf{q}_{k}, \mathbf{q}_{k+1})\\
    D_1L_{\text{aug}}(\mathbf{q}_{k + 1}, \mathbf{q}_{k+2}) &= D_1L_d(\mathbf{q}_{k + 1}, \mathbf{q}_{k + 2}) + \nabla\phi^T(\mathbf{q}_{k+1})\lambda_{k+1} \mathrm{d}t= 0
    \end{align}

Consequently, we can conveniently rewrite equations in the following form:

.. math:: 
    \begin{cases}
    D_2L_{\text{aug}}(\textbf{q}_{k}, \textbf{q}_{k+1}) + D_1L_{\text{aug}}(\textbf{q}_{k+1}, \textbf{q}_{k+2}) = 0\\
    \phi(\textbf{q}_{k+2}) = 0
    \end{cases}

External Forcing
----------------

To include external forcing we might want to use slightly different method as a base.
In particular, Lagrange-D'Alembert principle:

.. math:: 
    \delta\int_{t_1}^{t_2}L(\mathbf{q}, \dot{\mathbf{q}}, t)\mathrm{d}t + \int_{t_1}^{t_2}Q\delta\mathbf{q}\mathrm{d}t = 0


Where :math:`Q` represents generalized forces. Further we would need to define discrete forcing.
Consider left (:math:`F_d^-`) and right (:math:`F_d^+`) forces to satisfy the following:

.. math:: 
    F_d^-(\mathbf{q}_k, \mathbf{q}_{k+1})\delta\mathbf{q}_k + F_d^+(\mathbf{q}_{k}, \mathbf{q}_{k+1})\delta{q}_{k+1} \approx \int_{t_k}^{t_{k+1}}F(\mathbf{q}, \dot{\mathbf{q}})\delta\mathbf{q}\mathrm{d}t

With :math:`\mathbf{q}, \dot{\mathbf{q}}` defined using some midpoint rule, for instance, Euler rule.

Consequently, we can derive the following residual:

.. math:: 
    D_2L_d(\mathbf{q}_k, \mathbf{q}_{k + 1}) + D_1L_d(\mathbf{q}_{k + 1}, \mathbf{q}_{k + 2}) + F_d^+(\mathbf{q}_{k}, \mathbf{q}_{k+1}) + F_d^-(\mathbf{q}_{k+1}, \mathbf{q}_{k+2})= 0


Discrete Euler-Lagrange in Momentum Form
----------------------------------------

Finally, let's define residual in momentum form. For this we will need recall 
that :math:`L(\mathbf{q}, \dot{\mathbf{q}}, t)` (Lagrangian) and :math:`H(\mathbf{q}, \mathbf{p}, t)` (Hamiltonian) are Legendre transforms
of each other. With :math:`\mathbf{q}, t` passive variables.

.. math:: 
    \mathbf{p} = \nabla_{\dot{\mathbf{q}}}L(\mathbf{q}, \dot{\mathbf{q}}, t)

    \dot{\mathbf{q}} = \nabla_{\mathbf{p}}H(\mathbf{q}, \mathbf{p}, t)

Where :math:`\mathbf{p}` is generalized momentum and :math:`H` is Hamiltonian. 
Legendre transform follows from their definitions. With some
assumptions we can derive that:

.. math:: 
    \mathbf{p}_{k} = D_2L_d(\mathbf{q}_{k-1}, \mathbf{q}_{k}) = -D_1L_d(\mathbf{q}_{k}, \mathbf{q}_{k+1})

This way we can initialise our system knowing only the single state. Solving residual w.r.t. :math:`\mathbf{q}_{k+1}`
gives the next state and allows to calculate the next momentum.

.. math:: 
    (\mathbf{q}_k, \mathbf{p}_k) \rightarrow (\mathbf{q}_{k+1}, \mathbf{p}_{k+1}) \rightarrow \ldots


References
----------
.. bibliography::