"""Andres Fernandez Rodriguez, 5692442
   Machine Learning I, Winter2016/17

   This is a Python2 script, Part of the exercise sheet 4
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

MU_GEN = 2
VAR_GEN = 4
MU_PRIOR = 0
VAR_PRIOR = 8
DOMAIN = np.arange(-5, 5, 0.1)

def perform_estimation(mu_gen=MU_GEN, var_gen=VAR_GEN, mu_prior=MU_PRIOR,
                       var_prior=VAR_PRIOR, n=100):
    dataset = np.random.normal(mu_gen, var_gen**0.5, n)
    mean = float(sum(dataset))/n if n != 0 else 0
    mu_n = float((n*var_prior*mean)+(var_gen*mu_prior))/(n*var_prior+var_gen)
    var_n = float(var_gen*var_prior)/(n*var_prior+var_gen)
    return (dataset, mu_n, var_n)

def prior_estimation(domain, mu=MU_PRIOR, variance=VAR_PRIOR):
    fn = lambda(x): scipy.stats.norm(mu, variance**0.5).pdf(x)
    return [fn(elt) for elt in domain]

def posterior_estimation(domain, mu_n, var_n):
    fn = lambda(x): float(1)/(var_n*(2*np.pi)**0.5) * np.exp(float((x-mu_n)**2)/
                                                             (-2*var_n))
    return [fn(elt) for elt in domain]



data1 = perform_estimation(n=0)
data2 = perform_estimation(n=1)
data3 = perform_estimation(n=3)
data4 = perform_estimation(n=10)
data5 = perform_estimation(n=30)
data6 = perform_estimation(n=100)
#data7 = perform_estimation(n=300)
#data8 = perform_estimation(n=1000)




plt.plot(DOMAIN, prior_estimation(DOMAIN), label="prior")
plt.plot(DOMAIN, posterior_estimation(DOMAIN, data1[1], data1[2]),label="n=0")
plt.plot(DOMAIN, posterior_estimation(DOMAIN, data2[1], data2[2]),label="n=1")
plt.plot(DOMAIN, posterior_estimation(DOMAIN, data3[1], data3[2]),label="n=3")
plt.plot(DOMAIN, posterior_estimation(DOMAIN, data4[1], data4[2]),label="n=10")
plt.plot(DOMAIN, posterior_estimation(DOMAIN, data5[1], data5[2]),label="n=30")
plt.plot(DOMAIN, posterior_estimation(DOMAIN, data6[1], data6[2]),label="n=100")
#plt.plot(DOMAIN, posterior_estimation(DOMAIN, data7[1], data7[2]),label="n=300")
#plt.plot(DOMAIN, posterior_estimation(DOMAIN, data8[1], data8[2]),label="n=0")


plt.title('BAYESIAN INFERENCE')
legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
plt.subplots_adjust(left=0.15)
plt.show()
