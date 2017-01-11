"""Andres Fernandez Rodriguez, 5692442
   Machine Learning I, Winter2016/17

   This is a Python2 script, Part of the exercise sheet 6
"""

import numpy as np
from  scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator



def f(x):
    """The original deterministic function (trigonometric), as given
    """
    return np.sin(2*np.pi*x)-np.cos(np.pi*x)


def generate_dataset(xmin=0, xmax=1, beta=1, size=1000):
    """generates the dataset following the descriptions of the exercise sheet:
          -> x drawn form a uniform distr.
          -> f is a trigonometric function
          -> noise is a gaussian, added to the result of f
    """
    x_points = sorted(np.random.uniform(xmin,xmax, size))
    noise = lambda: np.random.normal(0, beta**(-0.5))
    return (np.matrix(x_points), np.matrix([f(x)+noise() for x in x_points]))




def design_mat(x, rows=9, s=1):
    """crafts the design matrix, with one bias component and
       'rows'-many RBF components, with means equally distributed
       between 0 and 1, and variance s (1 by default)
       IMPORTANT: design_mat(np.array([x])) IS THE SAME AS THE
       VECTOR OF BASIS FUNCTIONS FOR x€R
    """
    RBF = lambda x,mu,s: np.exp(-1*(np.power(x-mu, 2))/(2*s*s))
    mu_vec = np.linspace(0, 1, rows-1)
    desig = np.vstack([RBF(x, mu, s) for mu in mu_vec])
    desig = np.vstack([np.matrix([1 for _ in range(x.size)]), desig])
    return desig.transpose()


def s_n_inverse(alpha, beta, phi):
    """the estimated covariance matrix of the posterior, from the
       given formula
    """
    symm = phi.transpose().dot(phi)
    return alpha*np.eye(symm.shape[0]) + beta*symm

def post_var(t, sn_m, beta):
    """the implemented formula for the variance
    """
    varphi = design_mat(np.array([t]))
    return 1.0/beta + varphi.dot(sn_m).dot(varphi.transpose()).A1[0]

def post_mean(t, mn_vec):
    """ the implemented formula for the mean
    """
    varphi = design_mat(np.array([t]))
    return varphi.dot(mn_vec).A1[0]



## GLOBAL PARAMETERS
X_MIN = 0
X_MAX = 1
BETA = 100
ALPHA = 1

def main_routine(size):
    """generate the dataset and plot it
    """
    x_ds,y_ds = generate_dataset(X_MIN, X_MAX, BETA, size)
    phi = design_mat(x_ds)
    sn = np.linalg.inv(s_n_inverse(ALPHA, BETA, phi)) # estim. covar mat
    mn = BETA*(sn.dot(phi.transpose()).dot(y_ds.transpose())) # estim. mean


    x_orig = np.arange(0,1,0.01)
    y_orig = [f(i) for i in x_orig]

    y_pred = [post_mean(i,mn)+f(i) for i in x_orig]
    var_pred =[post_var(i, sn, BETA)**1.3*15 for i in x_orig] # the **1.3*15 is simply a plot normalization

    # plot results!
    plt.plot(x_orig, y_orig, "g", label="original")
    plt.errorbar(x_orig, y_pred, yerr=var_pred, label = "predicted")
    plt.plot(x_ds.tolist()[0], y_ds.tolist()[0], "ro", label="samples")

    plt.title('BAYESIAN LEARNING FOR '+str(size)+" SAMPLES")
    legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
    plt.show()


# run it several times, for several datasizes
for n in[2,4,10, 100]:
    main_routine(n)
