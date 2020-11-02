import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import firwin

'''
Description:
Use correlated random noise to simulate a potential ship velocity plot

Date: 


Author: Hunter Akins
'''

coeffs = firwin(11, .04, fs=.1, window='hann')
coeffs = coeffs[5:-1]

coeffs = np.array([1, .8])
xk_log = []
xk = np.zeros((1+coeffs.size, 1))
xk[0,0] = 2.4
noise_var = .02

xk[1:,0] = noise_var*np.random.randn(coeffs.size)
xk_log.append(xk)

""" Create noise matix G"""
G = np.zeros((coeffs.size,1))
G[0,0] = 1
G[1,0] = 1
print(G)

""" Create update matrix F """
F = np.zeros((xk.size, xk.size))

""" Use an expected variance of 2.4**2"""
noise_var = np.sum(np.square(coeffs))*noise_var
v_var = np.square(2.4)
xdot_weight = np.sqrt((v_var - noise_var) / v_var)


F[0,0] = 1
F[0,1:] = coeffs[:]
for i in range(2, F.shape[0]):
    F[i, i-1] = 1
print(F)

""" Simulate 35 minutes, sampled every 10 seconds
"""
for i in range(6*35):
    wk = noise_var*np.random.randn(1)
    xk = F@xk + G@wk
    xk_log.append(xk)
    

xdots = [x[0,0] for x in xk_log]
plt.plot(xdots)
plt.show()
