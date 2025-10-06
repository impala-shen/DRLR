import matplotlib.pyplot as plt
import math
import pyproj
import numpy as np
import control

commd_boom = 0
commd_bucket = 0
global start_dis
Safety_Flag = False
running = True
global vel_c, pos_c
vel_c = 0
pos_c = 0
# param = [510.2548, -562.5049, 2.5368, 0.005, -57.3876, 590.9841, 132.6020*0, 102.8494*0, 3572.24*0, 3263.57*0]
param = [2649.756, -1789.1147223, 1.4878, 0.0034,  -1325.130, 1885.709, 0.3, 0.1, 1576.01608, 1141.886337]
# param = [-39635.34, 47859.80, 0.265, -0.00007,  786600.70, -11618.45, -48.38, 34.43, 3710.02, -63654.69]
l1 = 2.195
l2 = 1.03
g = 9.82
dt = 0.01
ml1 = param[0]
ml2 = param[1]
lc1 = param[2]
lc2 = param[3]
Il1 = param[4]
Il2 = param[5]
fc1 = param[6]
fc2 = param[7]
fv1 = param[8]
fv2 = param[9]

def B_inertia(theta):
    c1 = math.cos(theta[0])
    c2 = math.cos(theta[2])

    B11 = ml1 * pow(lc1, 2) + Il1 + ml2 * (pow(l1, 2) + pow(lc2, 2) + 2 * l1 * lc2 * c2) + Il2
    B12 = ml2 * (pow(lc2, 2) + l1 * lc2 * c2) + Il2
    B22 = ml2 * pow(lc2, 2) + Il2

    B = np.array([[B11, B12],
         [B12, B22]]).reshape((2,2))
    return B

def C_vel(theta):
    s2 = math.sin(theta[2])
    c12 = math.cos(theta[0] + theta[2])
    h = -l1 * lc2 * ml2 * s2
    C = [[h * theta[3], h * (theta[1] + theta[3])],
         [-h * theta[1], 0]]
    # C_tau = np.dot(C, [[theta[1]], [theta[3]]])
    C = np.array(C).reshape((2,2))
    return C

def gq_gravity(theta):
    c1 = math.cos(theta[0])
    c12 = math.cos(theta[0] + theta[2])
    gq = [[ml2*(c12*lc2+l1*c1)*g+lc1*ml1*c1*g], [lc2*ml2*c12*g]]
    gq = np.array(gq).reshape((2,1))
    return gq

def fc_friction(theta):
    fc = [[fc1 * np.sign(theta[1]) + fv1 * theta[1]], [fc2 * np.sign(theta[3]) + fv2 * theta[3]]]
    fc = np.array(fc)
    return fc.reshape((2,1))

def RobotDynamics(B, C, gq, fc, theta, tau):
    dt = 0.01
    l1 = 2.195
    sum = -C - gq - fc + tau
    acc = np.matmul(np.linalg.inv(B), sum)
    # acc = np.linalg.lstsq(B, sum)[0]
    q_dot = np.array([[theta[1]], [theta[3]]])
    q = np.array([[theta[0]], [theta[2]]])
    q_dot_n = q_dot + acc * dt
    q_n = q + q_dot * dt
    return [q_n[0], q_n[1], q_dot_n[0], q_dot_n[1]]

def main_loop(machine: int = 0) -> None:
    simulation_time = 10
    simulation_count = int(simulation_time/dt)
    q = np.zeros((simulation_count, 2))
    q_dot = np.zeros((simulation_count, 2))
    K = [10, 10]
    r = np.zeros((simulation_count, 2))
    r_sum = np.zeros((simulation_count, 2))
    # tau = np.zeros((simulation_count, 2))
    for t in range(0, simulation_count-1):
        theta = [q[t][0], q_dot[t][0], q[t][1], q_dot[t][1]]
        B = B_inertia(theta)
        C = C_vel(theta)
        C_tau = np.matmul(C, [[theta[1]], [theta[3]]])
        gq = gq_gravity(theta)
        fc = fc_friction(theta)
        tau = np.array([[1000 * math.sin(0.5*math.pi*t*dt)], [0 * math.sin(0.5*math.pi*t*dt)]])
        [q[t+1][0], q[t+1][1], q_dot[t+1][0], q_dot[t+1][1]] = RobotDynamics(B, C_tau, gq, fc, theta, tau)
    # plt.plot(q_dot.T[0], 'r-')
    # # # # plt.plot(r.T[1], 'g-')
    # plt.show()
    return 0

main_loop()

