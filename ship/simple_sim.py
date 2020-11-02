import numpy as np
from matplotlib import pyplot as plt

'''
Description:
Simplest possible Kalman filter model simulation

Date: 

Author: Hunter Akins
'''


wgn_var = .014

x0 = 2.4
x0_var = np.square(2.4)
alpha = np.sqrt((x0_var - wgn_var)/x0_var)
xlog = []
""" 35 minutes, every 10 seconds update """
for i in range(35*6):
    xlog.append(x0)
    x0 = x0 + np.sqrt(wgn_var)*np.random.randn(1)
xlog.append(x0)
    
plt.plot(xlog)
plt.show()
