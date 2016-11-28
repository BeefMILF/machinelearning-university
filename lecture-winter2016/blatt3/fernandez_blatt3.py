"""Andres Fernandez Rodriguez, 5692442
   Machine Learning I, Winter2016/17

   This is a Python2 script, Part of the exercise sheet 3
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import ticker
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



def multivariate_normal(mu_vector, sigma_matrix, num_samples):
    """This function is basically a wrapper on numpy's
       multivariate normal function.
    """
    return  zip(* np.random.multivariate_normal(
        mu_vector, sigma_matrix, size=num_samples))

def plot_histogram_3d(x, y, num_bins, z_label="z", normalize=False):
    """This function, highly inspired in the snipped provided in
       http://matplotlib.org/examples/mplot3d/hist3d_demo.html
       Takes two numpy arrays of real numbers and plots a 3d
       histogram of them. Parameters:
       - x: the array of x-points
       - y: the array of y-points
       - num_bins: bin resolution PER DIMENSION
       - z_label: the name of the z-axis
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins=num_bins,
                                          range=[[min(x), max(x)],
                                                 [min(y), max(y)]])
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    dx = 0.3*np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()
    if normalize:
        dz = dz/len(x)
        z_label += " (normalized)"
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(z_label)
    saturation = 0.01
    colors = np.array([[0, 0 ,x+saturation] for x in dz])
    colors = colors/(max(dz)+saturation)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')
    plt.show()


def plot_histogram_2d(x, y, num_bins, z_label="z", normalize=False):
    """This function, highly inspired in the snipped provided in
       http://stackoverflow.com/a/26353952
       Takes two numpy arrays of real numbers and plots a 2d
       histogram of them. Parameters:
       - x: the array of x-points
       - y: the array of y-points
       - num_bins: bin resolution PER DIMENSION
       - z_label: the name of the z-axis
    """
    counts,xedges,yedges,Image = plt.hist2d(x, y, bins=num_bins, norm=LogNorm(),
                                            normed=True)
    cb = plt.colorbar(format= "%.3f")
    tick_locator = ticker.MaxNLocator(nbins=10)
    cb.locator = tick_locator
    cb.update_ticks()
    plt.show()

def print_mean_and_var(x, y, num_samples):
    """
    """
    mean_x = sum(x)/len(x)
    mean_y = sum(y)/len(y)
    var_x = sum([(i-mean_x)**2 for i in x])/len(x)
    var_y = sum([(i-mean_y)**2 for i in y])/len(y)
    covar_xy = np.cov(x,y)[0][1]
    print "estimated mean of x for "+str(num_samples)+" samples:", mean_x
    print "estimated mean of y for "+str(num_samples)+" samples:", mean_y
    print "estimated variance of x for "+str(num_samples)+" samples:", var_x
    print "estimated variance of y for"+str(num_samples)+" samples:", var_y
    print "estimated cov(x, y) for "+str(num_samples)+" samples:", covar_xy
    print "\n"


def norm_likelihood(mu, var, xlist):
    """
    """
    factor = (2*np.pi*var)**(-1*len(xlist)/2)
    sqerror = sum([(i-mu)**2 for i in xlist])
    return factor * np.exp((-1/(2*var))*sqerror)


def plot_surface(x, y, z_fn):
    """highly inspired on
       http://matplotlib.org/examples/mplot3d/surface3d_demo.html
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(x, y)
    z = z_fn(x,y)
    ax.set_xlabel('mean')
    ax.set_ylabel('variance')
    ax.set_zlabel('\nlikelihood of 1000 standard multivar. samples',linespacing=3)
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


# generate 1000 multivariate normal samples, plot them and estimate mean,var
MU_VECTOR = [2,3]
SIGMA_MATRIX = [[4,-0.5], [-0.5,2]]
NUM_SAMPLES = 1000
NUM_BINS = 20
norm_x, norm_y = multivariate_normal(MU_VECTOR, SIGMA_MATRIX, NUM_SAMPLES)
plot_histogram_2d(norm_x, norm_y, NUM_BINS, "density of multivariate normal", True)
plot_histogram_3d(norm_x, norm_y, NUM_BINS, "density of multivariate normal", True)
print_mean_and_var(norm_x, norm_y, NUM_SAMPLES)


# estimate mean, var for different sample sizes
NUM_SAMPLES = 2
norm_x, norm_y = multivariate_normal(MU_VECTOR, SIGMA_MATRIX, NUM_SAMPLES)
print_mean_and_var(norm_x, norm_y, NUM_SAMPLES)
NUM_SAMPLES = 20
norm_x, norm_y = multivariate_normal(MU_VECTOR, SIGMA_MATRIX, NUM_SAMPLES)
print_mean_and_var(norm_x, norm_y, NUM_SAMPLES)
NUM_SAMPLES = 200
norm_x, norm_y = multivariate_normal(MU_VECTOR, SIGMA_MATRIX, NUM_SAMPLES)
print_mean_and_var(norm_x, norm_y, NUM_SAMPLES)



# plot the likelihood of standard-normal samples as a function of mean and var
RANGE_MU = np.arange(-1, 1, 0.1)
RANGE_SIGMA = np.arange(0.1, 2, 0.1)
norm_x, norm_y = multivariate_normal([0,0], [[1,0],[0,1]], 100)
plot_surface(RANGE_MU, RANGE_SIGMA, (lambda a,b: norm_likelihood(a,b,norm_x)+
                                     norm_likelihood(a,b,norm_y)))
