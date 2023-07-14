import numpy as np
import scipy as sc


def HR(x, I, a, b, c, d, r, s):
    """
    Hindmarsh-Rose model.
    """
    x1, x2, x3 = x
    return np.array([x2 - a * x1 ** 3 + b * x1 ** 2 - x3 + I,
                     c - d * x1 ** 2 - x2,
                     r * (s * (x1 - x3) - x2)])

def HR_jac(x, I, a, b, c, d, r, s):
    """
    Jacobian of the Hindmarsh-Rose model.
    """
    x1, x2, x3 = x
    return np.array([[-3 * a * x1 ** 2 + 2 * b * x1, 1, -1],
                     [-2 * d * x1, -1, 0],
                     [r * s - r * x3, -r, -r * s + r * x1]])

def HR_nullclines(I, a, b, c, d, r, s):
    """
    Compute the nullclines of the Hindmarsh-Rose model.
    """
    x1 = np.linspace(-3, 3, 1000)
    x2 = np.linspace(-3, 3, 1000)
    x1_nullcline = (a * x1 ** 3 - b * x1 ** 2 + x1 - I) / c
    x2_nullcline = d * x1 ** 2 + x2
    return x1, x1_nullcline, x2, x2_nullcline

def HR_fixed_points(I, a, b, c, d, r, s):
    """
    Compute the fixed points of the Hindmarsh-Rose model.
    """
    x1, x2, x3 = sy.symbols('x1, x2, x3')
    f1, f2, f3 = HR([x1, x2, x3], I, a, b, c, d, r, s)
    sol = sy.solve([f1, f2, f3], [x1, x2, x3])
    return sol

def HR_eigenvalues(I, a, b, c, d, r, s):
    """
    Compute the eigenvalues of the fixed points of the Hindmarsh-Rose model.
    """
    x1, x2, x3 = sy.symbols('x1, x2, x3')
    f1, f2, f3 = HR([x1, x2, x3], I, a, b, c, d, r, s)
    jac = sy.Matrix([[sy.diff(f1, x1), sy.diff(f1, x2), sy.diff(f1, x3)],
                     [sy.diff(f2, x1), sy.diff(f2, x2), sy.diff(f2, x3)],
                     [sy.diff(f3, x1), sy.diff(f3, x2), sy.diff(f3, x3)]])
    sol = HR_fixed_points(I, a, b, c, d, r, s)
    eig = []
    for i in range(len(sol)):
        eig.append(jac.subs([(x1, sol[i][0]), (x2, sol[i][1]), (x3, sol[i][2])]).eigenvals())
    return eig


# With this function below solve the ode of the HR model for a given set of parameters and initial conditions, over a 3000 ms time period.
# period. The function returns the time series of the three state variables.

def HR_ode(I, a, b, c, d, r, s, x0, dt=0.01, tmax=3000):
    """
    Solve the ode of the Hindmarsh-Rose model for a given set of parameters and initial conditions, over a 3000 ms time period.
    The function returns the time series of the three state variables.
    """
    t = np.arange(0, tmax, dt)
    x = np.zeros((len(t), 3))
    x[0, :] = x0
    for i in range(1, len(t)):
        x[i, :] = x[i - 1, :] + dt * HR(x[i - 1, :], I, a, b, c, d, r, s)
    return t, x

# With the function below plot the above time series of the three state variables.

def HR_plot(I, a, b, c, d, r, s, x0, dt=0.01, tmax=3000):
    """
    Plot the time series of the three state variables of the Hindmarsh-Rose model.
    """
    t, x = HR_ode(I, a, b, c, d, r, s, x0, dt, tmax)
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(t, x[:, 0], color='blue')
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('x1')
    ax[1].plot(t, x[:, 1], color='red')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('x2')
    ax[2].plot(t, x[:, 2], color='green')
    ax[2].set_xlabel('t')
    ax[2].set_ylabel('x3')
    plt.tight_layout()
    plt.show()

# with the below function you can simulate a population of 200 HR neurons, with random initial conditions, and plot the time series of the three state variables of the first 10 neurons.

def HR_population(I, a, b, c, d, r, s, dt=0.01, tmax=3000):
    """
    Simulate a population of 200 HR neurons, with random initial conditions, and plot the time series of the three state variables of the first 10 neurons.
    """
    x0 = np.random.uniform(-3, 3, (200, 3))
    t, x = HR_ode(I, a, b, c, d, r, s, x0, dt, tmax)
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for i in range(10):
        ax[0].plot(t, x[:, i, 0], color='blue')
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('x1')
        ax[1].plot(t, x[:, i, 1], color='red')
        ax[1].set_xlabel('t')
        ax[1].set_ylabel('x2')
        ax[2].plot(t, x[:, i, 2], color='green')
        ax[2].set_xlabel('t')
        ax[2].set_ylabel('x3')
    plt.tight_layout()
    plt.show()

    

