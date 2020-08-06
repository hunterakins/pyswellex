import numpy as np
from matplotlib import pyplot as plt
from swellex.audio.garbage import gen_range_dep_movie
from swellex.audio.autoregressive import check_fest, comp_fest
import sys

'''
Description:
Throwaway script to plot some stuff

Date:  7/17

Author: Hunter Akins
'''

freq = int(sys.argv[1])
check_fest(freq,1500, 750)
plt.show()
comp_fest([127, 385], 1500, 750)
plt.show()
