import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

# Generate some sample data
x = np.logspace(1, 5, 100)
y = np.logspace(1, 3, 100)

# Create the plot
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y)

# Set the x-axis and y-axis scales to logarithmic
ax.set_xscale('log')
ax.set_yscale('log')

# Define the tick format function
def log_tick_formatter(val, pos=None):
    """
    Formatter function for logarithmic ticks.
    """
    # Find the nearest power of 10 to the tick value
    power = int(np.floor(np.log10(val)))

    # Calculate the logarithmic change
    change = val / 10 ** power

    # Determine if the tick value is a power of 10
    if change == 1:
        # Return the power of 10 as the tick label
        return fr'$10^{{{power}}}$'
    else:
        # Return an empty string for non-logarithmic changes
        return ''

# Set the x-axis tick locator and formatter
ax.xaxis.set_major_locator(LogLocator(subs='all'))
ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

# Set the y-axis tick locator and formatter
ax.yaxis.set_major_locator(LogLocator(subs='all'))
ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

# Display the plot
plt.show()