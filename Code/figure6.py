# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet

Proportion of surviving species in function of the interaction drift
for a sample of different correlation values.
"""

# Importation of the packages and the main functions:

import numpy as np
import matplotlib.pyplot as plt
from functions import res_function

# %%

MU = 0  # Interaction drift

rho_index = np.array([-0.5, 0, 0.5])  # Sample of the correlation parameter

# Sample of the interaction strength parameter:
PREC_ALPHA = 50
alpha_sample = np.linspace(1.4, 3, PREC_ALPHA)
y = np.eye(3, alpha_sample.size)

i = 0

for rho in rho_index:
    for j, alpha in enumerate(alpha_sample):
        y[i, j] = res_function(MU, alpha, rho)[3]
    i += 1


# Part dedicated to the display of the figure:

fig = plt.figure(1, figsize=(10, 6))
plt.plot(alpha_sample, y[0, :], linestyle='dashed',
         color='k', label=r'$\rho = -0.5$')
plt.plot(alpha_sample, y[1, :], linestyle='solid',
         color='k', label=r'$\rho = 0$')
plt.plot(alpha_sample, y[2, :], linestyle='dotted',
         color='k', label=r'$\rho = 0.5$')
plt.ylabel(r"Proportion of surviving species ($\phi$)", fontsize=15)
plt.xlabel(r"Interaction strength ($\alpha$)", fontsize=15)
plt.legend(loc='lower right')
plt.show()
