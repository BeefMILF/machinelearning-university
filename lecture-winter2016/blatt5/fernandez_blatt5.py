"""Andres Fernandez Rodriguez, 5692442
   Machine Learning I, Winter2016/17

   This is a Python2 script, Part of the exercise sheet 5
"""

import numpy as np
import matplotlib.pyplot as plt



# generate the datasets
def make_datapoint(x):
    """first element is the sample, second the 'ideal' label
    """
    sinx = np.sin(2*np.pi*x)
    return (sinx+0.2*np.random.normal(0,1), sinx)

N = 1000
NUM_DATASETS = 10
X_MIN = -1
X_MAX = 1
X_POINTS = np.linspace(X_MIN,X_MAX,N)
DATASETS = [np.matrix([make_datapoint(p) for p in X_POINTS])
            for _ in range(NUM_DATASETS)]


# define the parametric model
pol = np.polynomial.polynomial.polyval
NUM_PARAMETERS = 8
weights = [0 for _ in range(NUM_PARAMETERS)]
h = lambda datapoint: pol(datapoint[0], weights)
E = lambda datapoint: 0.5*(h(datapoint)-datapoint[1])**2


# define calculations
def adapt_dataset(dataset, num_pars):
    """adapts the dataset creating bias terms and poly. features
    """
    x = dataset[:,0]
    y = dataset[:,1]
    return (np.hstack([np.power(x, i) for i in range(num_pars)]), y)


def get_ml_weights(dataset):
    """performs the normal equation on the dataset
    """
    x = dataset[0]
    y = dataset[1]
    x_trans = x.transpose()
    return np.linalg.inv(x_trans.dot(x)).dot(x_trans).dot(y)


# perform calculations
DATASETS = [adapt_dataset(DATASETS[i], NUM_PARAMETERS) for i in range(NUM_DATASETS)]
WEIGHTS = [get_ml_weights(d) for d in DATASETS]

TOTAL_AVERAGE = sum(DATASETS[n][0]/NUM_DATASETS for n in range(NUM_DATASETS))

# plot results!



fig, ax = plt.subplots()

ax.plot(X_POINTS, DATASETS[0][1], "r", label="original")
ax.plot(X_POINTS, DATASETS[0][0], "b", alpha=0.1)
ax.plot(X_POINTS, TOTAL_AVERAGE, "g--", alpha=0.2, )

plt.title('VARIANCE VS. BIAS TRADEOFF')
legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
plt.subplots_adjust(left=0.15)
plt.show()
