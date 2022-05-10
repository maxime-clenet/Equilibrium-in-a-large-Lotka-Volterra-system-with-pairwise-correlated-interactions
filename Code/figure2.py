# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet

This files contains the transition to feasibility representation.

"""

# Importation of the packages and main functions:
import numpy as np
import matplotlib.pyplot as plt
from functions import elliptic_normal_matrix_opti


# %%

# Set the parameters:
MC_PREC = 1000  # number of MC experiments
SIZE_SAMPLE = 40  # definition of the size of the sample
# We introduce the interval of Kappa we want to study.
ind = np.linspace(0.75, 2, SIZE_SAMPLE)

# Introduction of the rho sample studied:
rho_index = np.array([-0.5, 0, 0.5])
NBR_RHO_IT = 3

# Dimension of the matrix:
n = 1000

# Storage matrix of the number of feasible solution:
res_hat = np.eye(NBR_RHO_IT, SIZE_SAMPLE)

j = 0  # initial counter of rho_size
compt = 0  # initial counter of the progress of simulation

# Part of the code dedicated to compute the value
# associated to the transition towards feasibility.

for rho in rho_index:

    res = np.zeros(SIZE_SAMPLE)
    compt_res = 0

    for kappa in ind:
        compt += 1
        print('Progress:', compt, '/', 3*SIZE_SAMPLE, end='\n')
        count_sol = 0

        for i in range(MC_PREC):

            A = elliptic_normal_matrix_opti(
                n, rho)/np.sqrt(n)  # Definition of the matrix
            sol = np.dot(np.linalg.inv(
                np.eye(n, n)-(1/(kappa*np.sqrt(np.log(n))))*A), np.ones(n))  # Compute the solution

            if sol[sol < 0].shape[0] == 0:
                count_sol = count_sol+1  # If positive solution.

        res[compt_res] = count_sol/MC_PREC  # Proportion of positive solution
        compt_res = compt_res+1

    res_hat[j, :] = res
    j = j+1


# Part dedicated to the display of the figure:

fig = plt.figure(1, figsize=(10, 6))
plt.plot(ind, res_hat[0, :], linestyle='dashed',
         color='k', label=r'$\rho = -0.5$')
plt.plot(ind, res_hat[1, :], linestyle='solid',
         color='k', label=r'$\rho = 0$')
plt.plot(ind, res_hat[2, :], linestyle='dotted',
         color='k', label=r'$\rho = 0.5$')
plt.axvline(np.sqrt(2), color='black',
            linestyle='dashdot', label='Threshold')
axes = plt.gca()
axes.xaxis.set_ticks([1, np.sqrt(2), 2])
axes.xaxis.set_ticklabels(
    ('1', r'$\sqrt{2}$', '2'), color='black', fontsize=10)
plt.xlabel(r"$\kappa$", fontsize=15)
plt.ylabel("P(Feasibility)", fontsize=15)
plt.legend(loc='upper left')
plt.show()
