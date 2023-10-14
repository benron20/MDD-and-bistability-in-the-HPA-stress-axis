# HPA axis ODEs

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

import plot_fft_amp
# define plot setting
#fonts
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)
lw = 4.0

# Define time t and input u,D
final_day = 1
time = [i*20 for i in range(13)]

#x2_depressed = plt.plot(time,[20,57,56,55,40,25,22,21,20,20,19,19,19], label='MDD', linewidth=lw, color = 'k')
x2_depressed = plt.errorbar(time,[20,57,56,55,40,25,22,21,20,20,19,19,19],yerr=[1,15,8,4,3,1,1,1,1,1,1,1,1], label='MDD', linewidth=lw, color = 'k')

#x2_control = plt.plot(time,[23,110,115,114,80,30,28,27,26,25,24,23,23], label='control', linewidth=lw,color = '0.6')
x2_control = plt.errorbar(time,[23,110,115,114,80,30,28,27,26,25,24,23,23],yerr=[1,15,14,14,12,2,2,2,5,1,1,1,1], label='control', linewidth=lw,color = '0.6')

#set axis values
ax1 = plt.subplot()
ax1.set_xticks([0,100,200])
ax1.set_yticks([0,50,100])
ax1.set_ylim([0, None])
ax1.set_xlim([0, 225])

#ax1.set_xticklabels(["one", "two", "three", "four"], rotation=45)

plt.legend()
plt.title("CRH test - empirical")
plt.xlabel('minutes')
plt.ylabel('ACTH (pg/ml)')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('empiric_blunted_CRH.svg')

plt.show()

    
