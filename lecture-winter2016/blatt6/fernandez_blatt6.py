"""Andres Fernandez Rodriguez, 5692442
   Machine Learning I, Winter2016/17

   This is a Python2 script, Part of the exercise sheet 6
"""

import numpy as np
from  scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm



### GLOBAL PARAMETERS (from the description in the exercise sheet)
W0 = -0.3
W1 = 0.5
W2 = 0.4
X_MIN = -2
X_MAX = 2
Y_MIN = -2
Y_MAX = 2
NOISE_DEV = 0.2
BETA = 25 # because 1/(beta) = (0.2)**2
ALPHA = 0.5 # because 1/alpha = 2



def poly2(x, w0=W0, w1=W1,w2=W2):
    """calculates the value of a polynomial of 2nd degree using horner's schema
    """
    return w0+x*(w1+x*w2)

def generate_dataset(size=1000):
    """generates the dataset following the descriptions of the exercise sheet
    """
    x_points = sorted(np.random.uniform(X_MIN, X_MAX,size))
    return (np.matrix(x_points), np.matrix([poly2(x)+np.random.normal(0, NOISE_DEV)
                                          for x in x_points]))


def design_mat(xarr, rows=3):
    """crafts the design matrix, 2nd-order polynomial by default
    """
    desig = np.vstack([np.power(xarr, i) for i in range(rows)])
    return desig.transpose()

def s_n_inverse(alpha, beta, phi):
    """the estimated covariance matrix of the posterior, from the formulas
       derived in exercises 1 and 2
    """
    symm = phi.transpose().dot(phi)
    return alpha*np.eye(symm.shape[0]) + beta*symm


def m_n(alpha, beta, design_mat, t):
    """the estimated mean of the posterior, from the formulas derived in
       exercises 1 and 2
    """
    sn = np.linalg.inv(s_n_inverse(ALPHA, BETA, design_mat))
    return beta*(sn.dot(design_mat.transpose()).dot(t.transpose()))

def posterior_prob(w, dataset):
    """returns the posterior probability for a given dataset and weight.
       Formula derived in exercises 1 and 2
    """
    X,t = dataset
    phi = design_mat(X)
    sn = np.linalg.inv(s_n_inverse(ALPHA, BETA, phi))
    mn = m_n(ALPHA, BETA, phi, t)
    return multivariate_normal.pdf(w, mn.A1, sn)



### TESTING VARIABLES:
def main_routine(samplesize):
    """performs and plots the the estimation for the given samplesize
    """
    GRID_SIZE = 0.01
    DATASET = generate_dataset(samplesize)
    X,Y= DATASET

    # optimized version of the functions above
    phi = design_mat(X)
    sn = np.linalg.inv(s_n_inverse(ALPHA, BETA, phi)) # estim. covar
    mn = BETA*(sn.dot(phi.transpose()).dot(Y.transpose())) # estim. mean
    posterior_fast = lambda(w): multivariate_normal.pdf(w, mn.A1, sn)

    TRUE_FN = [poly2(x) for x in X.A1]
    APPROX_FN = [poly2(x, mn[0].A1, mn[1].A1, mn[2].A1) for x in X.A1]

    ### PLOT RESULTS
    fig, ax = plt.subplots()
    ax.plot(X.A1, TRUE_FN, "b", alpha=0.5,label="original")
    ax.plot(X.A1, Y.A1, "b", alpha=0.2, label="samples")
    ax.plot(X.A1, APPROX_FN, "R", alpha=0.5, label="approximation")

    plt.title('BAYESIAN LEARNING WITH '+ str(samplesize)+
              " SAMPLES: X-DOMAIN")
    legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
    plt.subplots_adjust(left=0.15)
    plt.show()

    x = np.arange(X_MIN, X_MAX, GRID_SIZE)
    y = np.arange(Y_MIN, Y_MAX, GRID_SIZE)
    # IMPORTANT: since the plot is 2D+color, the thir dimension is
    # taken out: for the w0 axis, the optimal constant W0 is set, so
    # that the plot tends to show the intersection with maximal value
    z = np.array([[posterior_fast([W0,i,j]) for i in x] for j in y])
    z = z.reshape(x.size, y.size)
    plt.imshow(z+10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
            cmap=cm.hot, norm=LogNorm())
    plt.title('BAYESIAN LEARNING WITH '+ str(samplesize)+
              " SAMPLES: FUNCTION SPACE")
    plt.colorbar()
    plt.show()


# test with few samples to see how the posterior evolves from the prior
for i in [2,3,4,100]: main_routine(i)
