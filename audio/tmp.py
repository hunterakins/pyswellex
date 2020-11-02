import numpy as np
from matplotlib import pyplot as plt
from swellex.audio.garbage import gen_range_dep_movie
from swellex.audio.ts_comp import get_fc, get_bw
from swellex.audio.autoregressive import check_fest, comp_fest, comp_two_fest, comp_filter
import sys

'''
Description:
Throwaway script to plot some stuff

Date:  7/17

Author: Hunter Akins
'''
#comp_filter()

freq = int(sys.argv[1])
freq1 = int(sys.argv[2])
freq2 = int(sys.argv[3])
fig, axes, t = check_fest(freq,1500, 750)
B = get_bw(freq)
B = B/2
df = B/2
axes[0].plot(t, [get_fc(freq)]*len(t))
axes[0].plot(t, [get_fc(freq)+df]*len(t))
axes[0].plot(t, [get_fc(freq)-df]*len(t))
plt.show()

plt.show()
comp_two_fest([freq1, freq2], 1500, 750)
plt.show()
