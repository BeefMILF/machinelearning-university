import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import ticker



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


MU_VECTOR = [2,3]
SIGMA_MATRIX = [[4,0], [0,2]]
NUM_SAMPLES = 1000
NUM_BINS = 20
norm_x, norm_y = multivariate_normal(MU_VECTOR, SIGMA_MATRIX, NUM_SAMPLES)
plot_histogram_2d(norm_x, norm_y, NUM_BINS, "density of multivariate normal", True)
plot_histogram_3d(norm_x, norm_y, NUM_BINS, "density of multivariate normal", True)
print_mean_and_var(norm_x, norm_y, NUM_SAMPLES)


NUM_SAMPLES = 2
norm_x, norm_y = multivariate_normal(MU_VECTOR, SIGMA_MATRIX, NUM_SAMPLES)
print_mean_and_var(norm_x, norm_y, NUM_SAMPLES)
NUM_SAMPLES = 20
norm_x, norm_y = multivariate_normal(MU_VECTOR, SIGMA_MATRIX, NUM_SAMPLES)
print_mean_and_var(norm_x, norm_y, NUM_SAMPLES)
NUM_SAMPLES = 200
norm_x, norm_y = multivariate_normal(MU_VECTOR, SIGMA_MATRIX, NUM_SAMPLES)
print_mean_and_var(norm_x, norm_y, NUM_SAMPLES)
