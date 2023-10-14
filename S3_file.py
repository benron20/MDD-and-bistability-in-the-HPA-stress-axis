# HPA axis ODEs

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

import plot_fft_amp
import datetime
from matplotlib.dates import DateFormatter, date2num


# Define parameters, taken from lecture note and "A new model for the HPA axis explains dysregulation of stress hormones on the timescale of weeks"
b_A = 0.1
a_A = 0.1
b_P = 0.05
a_P = 0.05
# a_h, b_h ,a_s, b_s requires additional research. 
a_h = 0.05
b_h = 0.05
a_s = 0.05
b_s = 0.05

# time unit convergance
a1 = 0.17*60*24
a1_ext = 0.016*60*24
a2 = 0.035*60*24
a3 = 0.0086*60*24*0.2
a3_ext = 0.0086*60*24*0.02
# I assume A, P in timescale of Omer Karin CRH paper (previously assumed 14 days), h in 3 weeks
a5 = 0.099
a6 = 0.049
a7 = 1/21

# kGR=4 according to Omer, maybe 5 according to article Ive read.
kGR = 4
# defines a,b and T treshold for GR step function
a=1
b=3
T=1.5

#u_val = lambda t,sd: max((3+np.random.normal(scale=sd) if t_start_pressure<t<t_stop_pressure else 1+np.random.normal(scale=sd)),0.01)
u_val = lambda t: (2 if t_start_pressure<t<t_stop_pressure else 1)
D_val = lambda t: (2 if t_start_treat<t<t_stop_treat else 1)
#x1_dose for 60 minutes. Karin took only 30 minutes, but this fits better:
x1_dose = lambda t: (0.4 if t_start_CRH<t<t_start_CRH+60/(24*60) else 0)
# assume DEX dose for a 5 hours - taken in night and stays untill morning-noon.
x3_dose = lambda t: (100 if t_start_DEX<t<t_start_DEX+30/(24*60) else 0)

# defines MR and GR receptors functions, same as Omer's.  
MR = lambda a: a
GR = lambda a: np.power(np.divide(a,kGR),3)+1

# a step function for the hippocampus
step = lambda x,T: a+b*(x>=T)


# Define ODEs
def dy_dt(y, t):
       
    global u_val
    global D_val
    global x1_dose
    global x3_dose

    
    A = max(float(y[0]),0)
    P = max(float(y[1]),0)
    h = max(float(y[2]),0)
    x1_internal = max(float(y[3]),0)
    x2 = max(float(y[4]),0)
    x3_internal = max(float(y[5]),0)
    x1_external = max(float(y[6]),0)
    x3_external = max(float(y[7]),0)

    x1_tot = x1_internal + x1_external
    x3_tot = x3_internal + x3_external
    
    u_t = u_val(t)
    D_t = D_val(t)
   
    dA_dt = a5*A*(x2 - 1)
    dP_dt = a6*P*(x1_tot - 1)
    dh_dt = a7*(1*D_t/step(x3_tot,T) - h)
    #dh_dt = 0

    dx1_internal_dt = a1*(u_t/(h*MR(x3_tot)*GR(x3_tot)) - x1_internal)
    dx1_external_dt = a1_ext*(x1_dose(t) - x1_external)
    #CRH and x3 influence on other components is combined with internal and external levels - x1_tot, x3_tot
    dx2_dt = a2*(x1_tot*P/GR(x3_tot) - x2)

    dx3_internal_dt = a3*(x2*A - x3_internal)
    dx3_external_dt = a3_ext*(x3_dose(t) - x3_external)
    

    return [dA_dt, dP_dt, dh_dt, dx1_internal_dt, dx2_dt, dx3_internal_dt, dx1_external_dt, dx3_external_dt];


# defines t, u, D for steady state solution
t_start_pressure = 1000
t_stop_pressure = 1500
t_start_treat = 3000
t_stop_treat = 4500
# time where CRH test is made, external x1 is added to the system
t_start_CRH = 10000
t_start_DEX = 10000

noise = False

# finds steady state to Define initial values of healthy state
steady_state = odeint(dy_dt, [1, 1, 1, 1, 1, 1, 1, 1], np.linspace(0,600,180000), hmax = 0.005)

A0 = steady_state[-1,0]
P0 = steady_state[-1,1]
h0 = steady_state[-1,2]
x1int_0 = steady_state[-1,3]
x2_0 = steady_state[-1,4]
x3int_0 = steady_state[-1,5]
x1ext_0 = steady_state[-1,6]
x3ext_0 = steady_state[-1,7]

#######################################################

# defines t, u, D for depressed solution
t_start_pressure = 20
t_stop_pressure = 200
# time where CRH test is made, external x1 is added to the system
t_start_CRH = 10000
noise = False

# finds steady state to Define initial values of depressed solution
steady_state = odeint(dy_dt, [1, 1, 1, 1, 1, 1, 1, 1], np.linspace(0,600,180000), hmax = 0.005)

A1 = steady_state[-1,0]
P1 = steady_state[-1,1]
h1 = steady_state[-1,2]
x1int_1 = steady_state[-1,3]
x2_1 = steady_state[-1,4]
x3int_1 = steady_state[-1,5]
x1ext_1 = steady_state[-1,6]
x3ext_1 = steady_state[-1,7]

#######################################################

# define plor setting
#fonts
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)
lw = 4.0

#calibration to empiric results function
calibrate = lambda x: 80*x - 62
#calibrate = lambda x: x

#Runs a CRH test, on depressed state

# Define time t and input u,D
final_day = 2
time = np.linspace(0,final_day,final_day*300)

u = [u_val(t) for t in range(len(time))]

t_start_pressure = 20
t_stop_pressure = 200
t_start_treat = 5000
t_stop_treat = 6000

# time where DEX is taken, external x3 is added to the system (10 minutes)
t_start_DEX = 10/(24*60)

# time where CRH test is made, external x1 is added to the system (15 hours)
t_start_CRH = 15/24

sol = odeint(dy_dt, [A1, P1, h1, x1int_1, x2_1, x3int_1,x1ext_1,x3ext_1], time, hmax = 0.0005)

solA = sol[:,0]
solP = sol[:,1]
solh = sol[:,2]
solx1_internal = sol[:,3]
solx2 = sol[:,4]
solx3_internal = sol[:,5]
solx1_external = sol[:,6]
solx3_external = sol[:,7]
solx1_tot = solx1_internal + solx1_external
solx3_tot = solx3_internal + solx3_external


ut = [u_val(t) for t in time]

#u_input = plt.plot(time, ut, 'y',  label='u_input')
#Adrenal = plt.plot(time, solA, 'r--',  label='Adrenal')
#Pituitary = plt.plot(time, solP, 'g',  label='Pituitary' )
#D_input = plt.plot(time, [D_val(t) for t in time],  label='treatment')
#hippocampus = plt.plot(time, solh, 'b',  label='hippocampus')

#x1_internal = plt.plot(time, solx1_internal, label='internal CRH')
#x1_external = plt.plot(time, solx1_external, label='external CRH')
#x1 = plt.plot(24*60*time, solx1_tot, label='CRH')

#x2 = plt.plot(time, solx2-(solx2[300*t_start_CRH]-1), label='ACTH')

#fig, ax = plt.subplots()
#hours = [datetime.timedelta(hours=24*i) for i in time]
#myFmt = DateFormatter("%H:%M")
#ax.xaxis.set_major_formatter(myFmt)
x2_depressed = plt.plot(time*24, calibrate(solx2), label='MDD', linewidth=lw, color='k')

    # sensitivity checks
    # print('depressed state is caused for u input = ',max([u_val(t) for t in time]))
    # print('depressed state is treated with drugs = ',max([D_val(t) for t in time]))
    #print('min h = ',min((1/h0)*solh), ' max h = ',max((1/h0)*solh), 'h variation = ', max(solh)/min(solh))

#Runs a CRH test, starting with a healthy steady state

# Define time t and input u,D
final_day = 2
time = np.linspace(0,final_day,final_day*300)

u = [u_val(t) for t in range(len(time))]

t_start_pressure = 20
t_stop_pressure = 200
t_start_treat = 5000
t_stop_treat = 6000

# time where DEX is taken, external x3 is added to the system (10 minutes)
t_start_DEX = 10/(24*60)

# time where CRH test is made, external x1 is added to the system (15 hours)
t_start_CRH = 15/24

sol = odeint(dy_dt, [A0, P0, h0, x1int_0, x2_0, x3int_0,x1ext_0,x3ext_0], time, hmax = 0.0005)

solA = sol[:,0]
solP = sol[:,1]
solh = sol[:,2]
solx1_internal = sol[:,3]
solx2 = sol[:,4]
solx3_internal = sol[:,5]
solx1_external = sol[:,6]
solx3_external = sol[:,7]
solx1_tot = solx1_internal + solx1_external
solx3_tot = solx3_internal + solx3_external

ut = [u_val(t) for t in time]

#u_input = plt.plot(time, ut, 'y',  label='u_input')
#Adrenal = plt.plot(time, solA, 'r--',  label='Adrenal')
#Pituitary = plt.plot(time, solP, 'g',  label='Pituitary' )
#D_input = plt.plot(time, [D_val(t) for t in time],  label='treatment')
#hippocampus = plt.plot(time, solh, 'b',  label='hippocampus')

#x1_internal = plt.plot(time, solx1_internal, label='internal CRH')
#x1_external = plt.plot(time, solx1_external, label='external CRH')
#x1 = plt.plot(24*60*time, solx1_tot, label='CRH', linewidth=lw)

#x2 = plt.plot(time, solx2-(solx2[300*t_start_CRH]-1), label='ACTH')

x2 = plt.plot(24*time, calibrate(solx2), label='control', linewidth=lw,color='0.6')

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
plt.title("DEX/CRH test - model prediction")
plt.xlabel('hours')
plt.ylabel('ACTH (pg/ml)')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('DEX_CRH.svg')

plt.show()

    
