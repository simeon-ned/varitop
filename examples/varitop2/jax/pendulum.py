from varitop.integrator import DelIntegrator
import jax.numpy as jnp
from tap import Tap


class Args(Tap):
    display: bool = False


args = Args().parse_args()

m = 1
g = 9.8


def kinetic_energy(q, dq):
    return 0.5 * dq * m * dq


def potential_energy(q):
    return -m * g * (1 - jnp.cos(q))


def lagrangian(q, dq):
    return kinetic_energy(q, dq) - potential_energy(q)


# Defining integrator
di = DelIntegrator()
di.backend = 'jax'
di.lagrangian = lagrangian


# Initial conditions
ns = 1000
dt = 0.01
q0 = jnp.array(jnp.pi / 2)
q_solution = jnp.zeros((ns, 1))
q_solution = q_solution.at[0].set(q0)
q_solution = q_solution.at[1].set(q0)


for i in range(2, ns):
    try:
        q1 = q_solution[i - 2][0]
        q2 = q_solution[i - 1][0]
        qi = di.step(q1, q2, dt)
        q_solution = q_solution.at[i].set(qi)
    except Exception as e:
        print(f'Solution failed: {e}')
        break


# Plotting animation of pendulum
if args.display:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Set the x and y limits of the plot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Create a line object for the line between the points
    (line1,) = ax.plot([], [], "black", linestyle='-')
    ax.plot([0], [0], 'bo', markersize=7)
    (mass,) = ax.plot([], [], 'ro', markersize=10)

    # Create a scatter plot for the points
    # (points,) = ax.plot([], [], "ro")

    # Animation update function
    def update(frame):
        # Update the line and points data
        alpha = q_solution[frame][0]
        x = jnp.sin(alpha)
        y = jnp.cos(alpha)
        line1.set_data([0, x], [0, y])
        mass.set_data([x], [y])

        return line1, mass

    # Create the animation
    animation = FuncAnimation(fig, update, frames=ns, blit=True)
    plt.show()
