# HPA axis ODEs

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

import plot_fft_amp
import datetime
from matplotlib.dates import DateFormatter, date2num

# define plor setting
#fonts
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)
lw = 4.0

time = [15,15.25,15.5,15.75,16, 16.25,16.5,17,18]

empiric_depressed = plt.plot(time, [10,17.5,26,27,25,22,19,15,11], label='MDD', linewidth=lw, color='k')

empiric_control = plt.plot(time,[2,10,13,15,15,14,14,10,6], label='control', linewidth=lw,color='0.6')

    # sensitivity checks
    # print('depressed state is caused for u input = ',max([u_val(t) for t in time]))
    # print('depressed state is treated with drugs = ',max([D_val(t) for t in time]))
    #print('min h = ',min((1/h0)*solh), ' max h = ',max((1/h0)*solh), 'h variation = ', max(solh)/min(solh))
    #print('min A = ',min((1/A0)*solA), ' max A = ',max((1/A0)*solA),'A variation = ', max(solA)/min(solA))

#set axis values
ax1 = plt.subplot()
ax1.set_xticks([15,16,17,18])
ax1.set_yticks([5*i for i in range(8)])
ax1.set_xticklabels(["15:00", "16:00", "17:00", "18:00"])
#rotation=45
ax1.set_ylim([0, 32])
ax1.set_xlim([15,18.5])

plt.legend()
plt.title("DEX/CRH test - empirical")
plt.xlabel('hours')
plt.ylabel('ACTH (pg/ml)')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('EMPIRIC_DEX_CRH.svg')

plt.show()

    
