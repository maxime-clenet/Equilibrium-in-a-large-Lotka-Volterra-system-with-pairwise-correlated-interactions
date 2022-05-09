# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet

Dynamics of the Lotka-Volterra model

"""

# Importation of the packages and main functions:

import numpy as np
import matplotlib.pyplot as plt
from functions import elliptic_normal_matrix, dynamics_LV

# %%

# Definition of the parameters of the LV-model:
N_SIZE = 10  # dimension of the population
RHO = 0  # correlation
ALPHA = 1.4  # interaction strength

A = elliptic_normal_matrix(N_SIZE, RHO) / \
    (np.sqrt(N_SIZE)*ALPHA)  # Matrix of interactions

# %%

# Parameter of the dynamics:
NBR_IT = 1000  # Number of iteractions
TAU = 0.012  # Time step
x_init = np.random.random(N_SIZE)*2  # Initial condition

# Compute the dynamics:
sol_dyn = dynamics_LV(A, x_init, nbr_it=NBR_IT, tau=TAU)  # y-axis
x_axis = np.linspace(0, NBR_IT*TAU, NBR_IT)  # x-axis

# Display the dynamics:
fig = plt.figure(1, figsize=(10, 6))
for i in range(sol_dyn.shape[0]):
    lab = '$N_{'+str(i+1)+'}$'
    plt.plot(x_axis, sol_dyn[i, :], color='k', label=lab)
plt.xlabel("Time (t)", fontsize=15)
plt.ylabel("Abundances ($x_i$)", fontsize=15)
plt.show()
