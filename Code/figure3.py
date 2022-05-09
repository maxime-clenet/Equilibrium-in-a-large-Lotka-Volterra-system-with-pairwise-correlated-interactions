# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet

Heatmap of the admissible zone to have a existence and uniqueness
of a solution with vanishing species.
"""

# Importation of the packages and functions:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as m


def limit_mu(rho, alpha):
    """
    An upper bound of mu for the admissible zone.

    Parameters
    ----------
    rho : float [-1,1]
        Correlation parameter.
    alpha : float [np.sqrt(2),+infty]
        Interaction strength parameter.

    Returns
    -------
    float
        Limit of mu in function of rho and alpha.

    """
    return 1/2+np.sqrt(1-(2*(1+rho))/alpha**2)/2


# %%
# The phase diagram can be computed following the definition of the set of admissible parameter A
PREC = 1000

# Initialisation of the rho bound:
limit_rho = np.linspace(-1, 1, PREC)


# Initialisation of the alpha bound:
limit_alpha = np.array([])
for rho in limit_rho:
    limit_alpha = np.append(limit_alpha, np.sqrt(2*(1+rho)))


X, Y = np.meshgrid(limit_rho, limit_alpha, indexing='ij')
Z = np.zeros((PREC, PREC))

# Choice of the color:
cdict = {
    'red':  ((0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
    'green':  ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
    'blue':  ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
}
cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

# Grid to get the limit of mu:
X, Y = np.meshgrid(limit_rho, limit_alpha)
for k in range(PREC):
    for j in range(PREC):
        Z[j, k] = limit_mu(X[j, k], Y[j, k])


# Part dedicated to the plot:

fig = plt.figure(1, figsize=(10, 6))
plt.pcolor(X, Y, Z, cmap=cm, vmin=0.5, vmax=1)
plt.colorbar()
plt.xlabel(r"Correlation ($\rho$)")
plt.ylabel(r"Interaction strength ($\alpha$)")
plt.show()
