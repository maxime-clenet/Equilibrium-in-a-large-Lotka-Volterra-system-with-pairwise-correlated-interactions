# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet
"""

# Importation of the packages:

import numpy as np
from lemkelcp import lemkelcp
from scipy import optimize
import scipy.stats as stats

# %%

# Functions dedicated to the generation of the elliptic model:


def elliptic_normal_matrix(n=10, rho=0):
    """
    Create a elliptic random matrix of size (n,n) with correlation term rho.

    Parameters
    ----------
    n : int, optional
        Dimension of the matrix. The default is 10.
    rho : float [-1,1], optional
        Correlation term. The default is 0.

    Returns
    -------
    A : numpy.ndarray(n,n)
        Matrix of interactions.

    """
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1):
            if (j == i):
                A[i, j] = np.random.randn()
            else:
                x = np.random.multivariate_normal(mean, cov, 1)
                A[i, j] = x[0, 0]
                A[j, i] = x[0, 1]
    return A


def elliptic_normal_matrix_opti(n=100, rho=0):
    """
    Same function as elliptic_normal_matrix but with optimal generation.

    Create a elliptic random matrix of size (n,n) with correlation term rho.

    Parameters
    ----------
    n : int, optional
        Dimension of the matrix. The default is 10.
    rho : float [-1,1], optional
        Correlation term. The default is 0.

    Returns
    -------
    A+B.T : numpy.ndarray(n,n)
        Matrix of interactions.

    """
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    A = np.diag(np.random.randn(n))
    B = np.zeros((n, n))
    x = np.random.multivariate_normal(mean, cov, int(n*(n-1)/2))
    A[np.triu_indices(n, 1)] = x[:, 0]
    B[np.triu_indices(n, 1)] = x[:, 1]
    return A+B.T


def correlated_normal_matrix(n=10, rho_matrix=np.zeros((10, 10))):
    """
    Create a elliptic random matrix of size (n,n) with a correlation 
    profile of size (n,n)

    Parameters
    ----------
    n : int, optional
        Dimension of the matrix. The default is 10.
    rho : numpy.ndarray(n,n) with values in [-1,1], optional
        Correlation profile. The default is np.zeros((10,10)).

    Returns
    -------
    A : numpy.ndarray(n,n)
        Matrix of interactions.

    """

    mean = [0, 0]
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1):
            if (j == i):
                A[i, j] = np.random.randn()
            else:
                cov = [[1, rho_matrix[i, j]], [rho_matrix[i, j], 1]]
                x = np.random.multivariate_normal(mean, cov, 1)
                A[i, j] = x[0, 0]
                A[j, i] = x[0, 1]
    return A


# %%

# Function associated with the Lotka-Volterra dynamics

def f_LV(x, A):
    """
    Function used in the RK scheme to approximate the dynamics of the LV EDO.

    Parameters
    ----------
    x : numpy.ndarray(n),
        x_k in the iterative scheme.
    A : numpy.ndarray(n,n),
        Non-normalized matrix of interactions.

    Returns
    -------
    x : numpy.ndarray(n).

    """
    N = len(A)
    x = np.dot(np.diag(x), (np.ones(N)-x-np.dot(A, x)))
    return(x)


def dynamics_LV(A, x_init, nbr_it, tau):
    """
    Runge-Kutta Scheme

    Parameters
    ----------
    A : numpy.ndarray,
        Normalized matrix of interactions.
    x_init : numpy.ndarray,
        Initial condition.
    nbr_it : int,
        Number of iterations.
    tau : float,
        Time step.

    Returns
    -------
    sol_dyn : numpy.ndarray,
        Line i corresponds to the values of the dynamics of species i.

    """
    x = x_init

    compt = 0

    # Matrix of the solution:
    sol_dyn = np.eye(len(A), nbr_it)

    # RK scheme:
    while (compt < nbr_it):

        f1 = f_LV(x, A)
        f2 = f_LV(x+tau*0.5*f1, A)
        f3 = f_LV(x+tau*0.5*f2, A)
        f4 = f_LV(x+tau*f3, A)

        x = x+tau*(f1+2*f2+2*f3+f4)/6

        for i in range(len(A)):
            sol_dyn[i, compt] = x[i]
        compt = compt+1

    print("Convergence dynamique: \n", np.around(
        sol_dyn[:, nbr_it-1], decimals=2))

    return sol_dyn


# %%

# Functions associated the the theoretical resolution of the solution
# of the fixed point problem with vanishing species.

def e_cond(delta):

    p_1 = np.exp(-delta**2/2)
    p_2 = 1-stats.norm.cdf(-delta)

    return (1/np.sqrt(2*np.pi))*p_1/p_2


def e2_cond(delta):
    p_1 = np.exp(-delta**2/2)
    p_2 = 1-stats.norm.cdf(-delta)

    return (1/np.sqrt(2*np.pi))*-delta*p_1/p_2+1


def sys_1(v, m, sigma, phi, mu, alpha, rho):

    delta = (1+m*mu)*alpha/sigma

    return 1-stats.norm.cdf(-delta)-phi


def sys_2(v, m, sigma, phi, mu, alpha, rho):

    delta = (1+m*mu)*alpha/sigma

    coef1 = phi/(1-rho*v/alpha)

    return coef1*(1+m*mu+sigma/alpha*e_cond(delta))-m


def sys_3(v, m, sigma, phi, mu, alpha, rho):

    delta = (1+m*mu)*alpha/sigma

    coef1 = phi/(1-rho*v/alpha)**2

    coef2 = (1+m*mu)**2

    coef3 = 2*(1+m*mu)*sigma/alpha

    coef4 = sigma**2/alpha**2

    return coef1*(coef2+coef3*e_cond(delta)+coef4*e2_cond(delta))-sigma**2


def sys_4(v, m, sigma, phi, mu, alpha, rho):

    coef1 = 1/(alpha-rho*v)

    return phi*coef1-v


def Gamma(x, mu, alpha, rho):
    """


    Parameters
    ----------
    x : x[O] correspond à sigma
        x[1] correspond à pi
    alpha : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    return (sys_1(x[0], x[1], x[2], x[3], mu, alpha, rho), sys_2(x[0], x[1], x[2], x[3], mu, alpha, rho), sys_3(x[0], x[1], x[2], x[3], mu, alpha, rho), sys_4(x[0], x[1], x[2], x[3], mu, alpha, rho))


def res_function(mu, alpha, rho):
    """


    Parameters
    ----------
    mu : TYPE
        DESCRIPTION.
    Sig : TYPE
        DESCRIPTION.
    Beta : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    (v, m, sigma, phi) = optimize.root(
        Gamma, [1, 2.1, 2.1, 0.8], args=(mu, alpha, rho,)).x

    return(v, m, sigma, phi)


# %%

# Functions associated the the empirical resolution of the solution
# of the fixed point problem with vanishing species.


def zero_LCP(N, A):
    """
    This function resolve the LCP problem of our model.
    If a solution exist, this function return the properties 
    of the solution i.e:
    - proportion of persistent species,
    - variance of the persistent species,
    - mean of the persistent species. 

    Parameters
    ----------
    N : int
        Correspond to the size of the population/matrix A.
    A : Matrice de taille (N,N)
        Correspond à la matrice ces intéractions.

    Returns
    -------
    Si la solution existe, cette fonction renvoie 
    le nombre d'espèces persistantes. (En pourcentage)
    - I/N
    L'espérance des abondances x_i !=0 du LCP
    - m_i_chap

    """

    q = np.ones(N)
    M = -np.eye(N)+A
    sol = lemkelcp.lemkelcp(-M, -q, maxIter=10000)

    res_LCP = sol[0]

    res_LCP_pos = res_LCP[res_LCP != 0]
    I = len(res_LCP_pos)

    m = sum(res_LCP_pos)/N
    sigma = np.sqrt(sum(res_LCP**2)/N)
    return (I/N, m, sigma)


def empirical_prop(n, alpha, mu, rho, mc_prec=100):

    S_p = np.zeros(mc_prec)
    S_sigma = np.zeros(mc_prec)
    S_m = np.zeros(mc_prec)

    for i in range(mc_prec):
        A = elliptic_normal_matrix_opti(n, rho)/(np.sqrt(n)*alpha)+mu/n
        (S_p[i], S_m[i], S_sigma[i]) = zero_LCP(len(A), A)

    return np.mean(S_p), np.mean(S_m), np.mean(S_sigma)
