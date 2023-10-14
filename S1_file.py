# HPA axis ODEs

import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

'''
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
a3 = 0.0086*60*24

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


# defines MR and GR receptors functions, same as Omer's.  
MR = lambda a: a
GR = lambda a: np.power(np.divide(a,kGR),3)+1

# a step function for the hippocampus
step = lambda x,T: a+b*(x>=T)


# Define ODEs
def dy_dt(y, t):

    global u_val
    
    A = max(float(y[0]),0)
    P = max(float(y[1]),0)
    h = max(float(y[2]),0)
    x1 = max(float(y[3]),0)
    x2 = max(float(y[4]),0)
    x3 = max(float(y[5]),0)

    u_t = u_val
   
    dA_dt = a5*A*(x2 - 1)
    dP_dt = a6*P*(x1 - 1)
    dh_dt = a7*(1/step(x3,T) - h)
    #dh_dt = 0


    dx1_dt = a1*(u_t/(h*MR(x3)*GR(x3)) - x1)
    #CRH influence on other components is combined with internal and external levels - x1_tot
    dx2_dt = a2*(x1*P/GR(x3) - x2)
    dx3_dt = a3*(x2*A - x3)
    

    return [dA_dt, dP_dt, dh_dt, dx1_dt, dx2_dt, dx3_dt];


# finds steady state to Define initial values
u_val = 1
steady_state = odeint(dy_dt, [1, 1, 1, 1, 1, 1], np.linspace(0,400,180000), hmax = 0.005)

A0 = steady_state[-1,0]
P0 = steady_state[-1,1]
h0 = steady_state[-1,2]
x1_0 = steady_state[-1,3]
x2_0 = steady_state[-1,4]
x3_0 = steady_state[-1,5]
#######################################################
u_val = 5

# finds steady state to Define initial values of depressed solution
steady_state = odeint(dy_dt, [1, 1, 1, 1, 1, 1], np.linspace(0,400,180000), hmax = 0.005)

A1 = steady_state[-1,0]
P1 = steady_state[-1,1]
h1 = steady_state[-1,2]
x1_1 = steady_state[-1,3]
x2_1 = steady_state[-1,4]
x3_1 = steady_state[-1,5]

########################################################
u_inputs = np.linspace(0,3.5,70)
A_healthy = []
A_depressed = []

#Runs a test, starting with a steady state
for uu in u_inputs:

    # calculate final A, starting from healthy state
    final_day = 400
    time = np.linspace(0,final_day,final_day*300)

    u_val = uu
    sol = odeint(dy_dt, [A0, P0, h0, x1_0, x2_0, x3_0], time, hmax = 0.0005)

    solA = sol[:,0]/A0
    A_final = solA[-1]
    A_healthy.append(A_final)
    ######

    # calculate final A, starting from depressed state
    final_day = 400
    time = np.linspace(0,final_day,final_day*300)

    u = [uu for t in range(len(time))]

    t_start_pressure = 2000
    t_stop_pressure = 2000
    t_start_treat = 5000
    t_stop_treat = 6000

    sol = odeint(dy_dt, [A1, P1, h1, x1_1, x2_1, x3_1], time, hmax = 0.0005)

    #divided by A0 on purpose- we want to plot results together and compare between depressed A and healty A
    solA = sol[:,0]/A0
    A_final = solA[-1]
    A_depressed.append(A_final)    
#results from run 18/12/22:
'''
u_inputs = [0, 0.05072464, 0.10144928, 0.15217391, 0.20289855,
       0.25362319, 0.30434783, 0.35507246, 0.4057971 , 0.45652174,
       0.50724638, 0.55797101, 0.60869565, 0.65942029, 0.71014493,
       0.76086957, 0.8115942 , 0.86231884, 0.91304348, 0.96376812,
       1.01449275, 1.06521739, 1.11594203, 1.16666667, 1.2173913 ,
       1.26811594, 1.31884058, 1.36956522, 1.42028986, 1.47101449,
       1.52173913, 1.57246377, 1.62318841, 1.67391304, 1.72463768,
       1.77536232, 1.82608696, 1.87681159, 1.92753623, 1.97826087,
       2.02898551, 2.07971014, 2.13043478, 2.18115942, 2.23188406,
       2.2826087 , 2.33333333, 2.38405797, 2.43478261, 2.48550725,
       2.53623188, 2.58695652, 2.63768116, 2.6884058 , 2.73913043,
       2.78985507, 2.84057971, 2.89130435, 2.94202899, 2.99275362,
       3.04347826, 3.0942029 , 3.14492754, 3.19565217, 3.24637681,
       3.29710145, 3.34782609, 3.39855072, 3.44927536, 3.5]
A_healthy = [6.3528300669485094e-18, 0.05148269203708552, 0.10296385323594984, 0.15443975451128605, 0.20590411118212248, 0.25734813683332136, 0.3087605728915482, 0.360127733212513, 0.41143356884992666, 0.46265975660962805, 0.5137858142172406, 0.5647892441286403, 0.6156457070845963, 0.6663292254498192, 0.7168124152105686, 0.7670667442834276, 0.8170628135716446, 0.8667706560666922, 0.9161600483053258, 0.965200827725413, 1.0138632089762032, 1.062118092070675, 1.1099373554354526, 1.157294127412718, 1.2041630305602358, 1.2505203941346374, 1.2963444313475887, 1.3416153792858763, 1.3863156006968576, 1.4304296480857495, 1.4739442916925958, 3.6706747458495808, 3.7219136084554343, 3.7716025051437025, 3.819837854857834, 3.8667073620661907, 3.912291117084145, 3.956662446564081, 3.9998886395879283, 4.042031578860637, 4.083148292989404, 4.123291441555222, 4.162509742464071, 4.200848349436018, 4.238349186308398, 4.2750512438067885, 4.310990843621025, 4.346201873918984, 4.380716019037574, 4.4145628383080915, 4.44777013043882, 4.480363968078863, 4.512368828169775, 4.543807735720826, 4.574702375940025, 4.605073195822873, 4.634939496368512, 4.664319516425186, 4.693230509078973, 4.721688811335684, 4.74970990778715, 4.777308488892909, 4.804498504371529, 4.831293212214661, 4.857705223728225, 4.883746544981355, 4.909428615002468, 4.93476234101406, 4.959758130991084, 4.984425923758203]
A_depressed = [3.5576635507711254e-17, 0.051482794656172295, 0.10296404026467496, 0.15444000649317308, 0.2059043840242072, 0.2573483189819467, 0.3087603467211962, 0.36012587409167796, 1.55913319370501, 1.7219756995127253, 1.8747096115560289, 2.0175949047377437, 2.1511509446292716, 2.2760391157805033, 2.3929757832955056, 2.5026752413010644, 2.6058163481515417, 2.703025641682774, 2.7948709457810965, 2.881861162931882, 2.9644494128743544, 3.043037754895693, 3.11798245569022, 3.1895992251053658, 3.258168122896749, 3.3239380048163554, 3.387130469815393, 3.4479433203049306, 3.5065535720971774, 3.5631200607445184, 3.6177856930390155, 3.670679390291014, 3.721917765893434, 3.771606574857556, 3.819841968148512, 3.8667115801001017, 3.9122954730964175, 3.956666960118397, 3.999893322651582, 4.042036438801464, 4.083153334210095, 4.1232966664642206, 4.162515152077946, 4.2008539437780685, 4.238354964679933, 4.275057204979213, 4.310996985973137, 4.346208195539954, 4.3807224986238955, 4.414569525782678, 4.447777042435953, 4.48037110109772, 4.512376178573873, 4.543815299845398, 4.574710150137366, 4.605081176483075, 4.634947679928637, 4.664327899382558, 4.693239087993411, 4.721697582831624, 4.7497188685618354, 4.777317635711249, 4.804507834070338, 4.831302721702395, 4.857714909984665, 4.883756405057621, 4.909438646019197, 4.934772540163018, 4.959768495530104, 4.9844364510143215]


#fonts
font = {'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)
lw = 3.0

plt.plot(u_inputs[:50], A_healthy[:50], 'black',  label='Adrenal steady state', linewidth=lw)
plt.plot(u_inputs[:50], A_depressed[:50], 'black',linewidth=lw)

plt.title("hysteresis")
plt.xlabel('u STRESS INPUT')
plt.ylabel('ADRENAL RELATIVE SIZE')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#set axis values
ax1 = plt.subplot()
ax1.set_xticks([0.36,1.52])
ax1.set_xticklabels(["u2", "u1"])
ax1.set_yticks([1.4,3.67])
ax1.set_yticklabels(["A_low", "A_ high"])
#rotation=45
ax1.set_ylim([0, 4.5])
ax1.set_xlim([0,2.5])


plt.savefig('hysteresis.svg')

plt.show()

    
