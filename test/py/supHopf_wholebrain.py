import time as tm
import numpy as np
import tvb.simulator.models.oscillator

from tvb.simulator.lab import *
from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive

import matplotlib as mpl
import matplotlib.pyplot as plt


#%%

# Create the God Damn mobel
hopf = tvb.simulator.models.oscillator.SupHopf()

# initialise the God Damn stoachistic integrator scheme.

heun = integrators.HeunDeterministic(dt=0.1)

ppi_fig = PhasePlaneInteractive(model=hopf, integrator=heun)
ppi_fig.show()