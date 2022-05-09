# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet

Parallel between the theoretical and empirical solutions in the case of 
vanishing species associated to the resolution of system of equations.
"""

# Importation of the packages and the main functions:

import numpy as np
import matplotlib.pyplot as plt
from functions import res_function, empirical_prop

# %%

# Parallel between the theoretical and empirical solutions of the
# fixed point with vanishing species.


MU = 0.2  # interaction drift
RHO = 0.5  # interaction correlation
PREC_ALPHA = 20  # Size sample of the interaction strength

# Compute the theoretical solutions:
x = np.linspace(2, 3, PREC_ALPHA)
theoretical_sol = np.zeros((4, x.size))

for i in range(PREC_ALPHA):
    theoretical_sol[:, i] = res_function(MU, x[i], RHO)

# Compute the empirical solutions:
N_SIZE = 100  # dimension of the matrix for the theoretical solution.
emp_sol = np.zeros((PREC_ALPHA, 3))
for i in range(PREC_ALPHA):
    alpha = x[i]
    print("Iteration:", i+1, '/', PREC_ALPHA)

    emp_sol[i, :] = empirical_prop(
        n=N_SIZE, alpha=alpha, mu=MU, rho=RHO, mc_prec=200)


# %%
# Part dedicated to the display of the solution:

# Proportion of the surviving species in function of alpha:
fig = plt.figure(1, figsize=(10, 6))
plt.plot(x, theoretical_sol[3], 'k', label=r'$p$')
plt.plot(x, emp_sol[:, 0], 'k*', label=r'$\widehat{p}$')
plt.xlabel(r"Interaction strength ($\alpha$)", fontsize=15)
plt.ylabel(r"Proportion of the surviving species ($\phi$)", fontsize=15)
plt.ylim(0.9, 1)
plt.show()
plt.close()


# Root mean square in function of alpha:
fig = plt.figure(1, figsize=(10, 6))
plt.plot(x, theoretical_sol[2], 'k',
         label=r'$\left \langle \mathbf{x}^2 \right \rangle$')
plt.plot(x, emp_sol[:, 2], 'k*',
         label=r'$\widehat{$\left \langle \mathbf{x}^2 \right \rangle$')
plt.xlabel(r"Interaction strength ($\alpha$)", fontsize=15)
plt.ylabel(
    r"Root mean square $\sqrt{\left \langle x^2 \right \rangle}$", fontsize=15)
plt.show()
plt.close()


# Mean in function of alpha:
fig = plt.figure(1, figsize=(10, 6))
plt.plot(x, theoretical_sol[1], 'k')
plt.plot(x, emp_sol[:, 1], 'k*')
plt.xlabel(r"Interaction strength ($\alpha$)", fontsize=15)
plt.ylabel(r"Mean $\left \langle x \right \rangle$", fontsize=15)
plt.show()
plt.close()
