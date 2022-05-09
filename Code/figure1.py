# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet

This file is used to plot the spectrum of the
elliptic model with constant correlation and with a
profile of correlation and keeping a trace for its generation.

"""
# Importation of the main packages and functions:

import numpy as np
import matplotlib.pyplot as plt
from functions import elliptic_normal_matrix_opti, correlated_normal_matrix

# %%
# Part dedicated to the plot of the elliptic matrix
# with a constant correlation parameter.


def plot_elliptic_spectrum(n_size=100, rho=0):
    """
    Representation of the spectrum of a elliptic
    matrix with a given correlation parameter rho.

    Parameters
    ----------
    n : int, optional
        Size of the matrix. The default is 100.
    rho : float in [-1,1], optional
        Correlation of the matrix. The default is 0.

    Returns
    -------
    fig : matplotlib.fig
        Figure of the spectrum.

    """
    A = elliptic_normal_matrix_opti(n_size, rho)  # Definition of the matrix

    # Renormalisation + eigenvalues
    eig_A = np.linalg.eigvals(A/np.sqrt(n_size))

    # Parameters used to display the theoretical frontier:
    u = 0  # x-position of the center
    v = 0  # y-position of the center
    a = 1+rho  # radius on the x-axis
    b = 1-rho  # radius on the y-axis
    t = np.linspace(0, 2*np.pi, 100)

    fig = plt.figure(1, figsize=(6, 6))

    plt.xlabel(r"Real axis ", fontsize=15)
    plt.ylabel("Imaginary axis ", fontsize=15)

    plt.plot(u+a*np.cos(t), v+b*np.sin(t), color='k', linewidth=3)
    plt.plot(eig_A.real, eig_A.imag, '.', color='k')
    plt.grid(color='lightgray', linestyle='--')
    plt.axis("equal")
    plt.xlim(-1.6, 1.6)
    plt.ylim(-1.6, 1.6)
    plt.show()
    return fig


# An example:
plot_elliptic_spectrum(n_size=1000, rho=0.5)

# %%

# Part dedicated to a profile of correlation.


def plot_full_correlated_spectrum(n_size=100, rho_matrix=np.random.random((100, 100))):
    """
    Representation of the spectrum of a elliptic
    matrix with a profile of correlation rho.


    Parameters
    ----------
    n : int, optional
        Dimension of the matrix. The default is 100.
    rho_matrix : numpy.ndarray, optional
        Profile of correlation of size (n,n). The default is np.random.random((100, 100)).

    Returns
    -------
    fig : matplotlib.fig
        Figure of the spectrum.
    norme_C : float
        Spectral norm of the spectrum.

    """

    # Definition of the matrix
    A = correlated_normal_matrix(n_size, rho_matrix)

    A_norm = A/np.sqrt(n_size)  # Renormalisation of the matrix
    eig_A = np.linalg.eigvals(A_norm)  # eigenvalues
    eig_A_norm = np.linalg.eig(np.transpose(A_norm)@A_norm)

    norme = np.sqrt(max(eig_A_norm[0]))

    fig = plt.figure(1, figsize=(10, 6))
    plt.plot(eig_A.real, eig_A.imag, '.', color='b')
    plt.grid(color='lightgray', linestyle='--')
    plt.axis("equal")
    plt.xlabel(r"Real axis", fontsize=15)
    plt.ylabel("Imaginary axis", fontsize=15)
    plt.show()
    return fig, norme


plot_full_correlated_spectrum(
    n_size=1000, rho_matrix=np.random.random((1000, 1000)))
