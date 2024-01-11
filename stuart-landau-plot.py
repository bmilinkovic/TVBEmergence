#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Damped Stuart-Landau model equations
def stuart_landau(z, t, alpha, omega):
    dzdt = (1 - 1j*omega)*z - (1 + alpha * np.abs(z)**2) * z
    return dzdt

# Numerical integration using Euler method
def euler_integrate(func, initial_condition, t_values, args=()):
    result = [initial_condition]
    z = initial_condition
    for t in t_values[1:]:
        dt = t - t_values[t_values.tolist().index(t) - 1]
        z = z + dt * func(z, t, *args)
        result.append(z)
    return np.array(result)
# this code still needs to be fixed:
# Parameters
alpha = 0.1
omega = 1.0

# Time values
t_values = np.linspace(0, 40, 1000)  # Increased the range to 40

# Initial condition
initial_condition = 1.0 + 0.5j

# Numerical integration
solution = euler_integrate(stuart_landau, initial_condition, t_values, args=(alpha, omega))

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(np.real(solution), np.imag(solution), t_values, label='Stuart-Landau')

# Set labels
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_zlabel('Time')
ax.set_title('Damped Stuart-Landau Model')

# Remove legend
ax.legend().remove()

# Set grid style
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle='dotted', linewidth=0.5)

# Show the plot
plt.show()



# %%
# Kuramoto model equations
def kuramoto(angles, t, omegas, K, N):
    dthetas_dt = omegas + (K / N) * np.sum(np.sin(angles - angles[:, np.newaxis]), axis=1)
    return dthetas_dt

# Numerical integration using Euler method
def euler_integrate(func, initial_condition, t_values, args=()):
    result = [initial_condition]
    angles = initial_condition
    for t in t_values[1:]:
        dt = t - t_values[t_values.tolist().index(t) - 1]
        angles = angles + dt * func(angles, t, *args)
        result.append(angles)
    return np.array(result)

# Parameters
N = 10  # Number of oscillators
K = 0.5  # Coupling strength



# Initial conditions
initial_angles = np.random.rand(N) * 2 * np.pi
initial_omegas = np.random.randn(N)

# Time values
t_values = np.linspace(0, 20, 1000)

# Numerical integration
solution = euler_integrate(kuramoto, initial_angles, t_values, args=(initial_omegas, K, N))

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
for i in range(N):
    ax.plot(np.cos(solution[:, i]), np.sin(solution[:, i]), t_values, label=f'Oscillator {i+1}')

# Set labels
ax.set_xlabel('Cosine of Phase')
ax.set_ylabel('Sine of Phase')
ax.set_zlabel('Time')
ax.set_title('Kuramoto Model')

# Show the plot
plt.legend()
plt.show()
