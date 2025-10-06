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
# param = [-85.686747, -190.713263, -5.044443, 0.004235, 4791.194843, 489.593826, 141.243602, 149.947923, 3973.656035, 1362.794578]
# param = [662.2588, -105.3467, 1.3536, 0.0064, -468.9032, 110.3836, 9.2339, -1.1488, 2435.6, -93.95]   ## Observer parameters
param = [517.0199, -649.6875, 2.7759, 0.0023, -174.4662, 686.2141, 166.9010*0, 146.5742*0, 2608.27, 2108.73]   ## Ctr parameters

# param_mob = [-322229.3, 10966.65, 0.0741, -0.00005,  786600.70, -11618.45, -10.3189, -31.916, 240975.54, 119191.3]
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

def Dynamics(theta, u):
    # c1 = math.cos(theta[0])
    # c2 = math.cos(theta[2])
    # B11 = ml1 * pow(lc1, 2) + Il1 + ml2 * (pow(l1, 2) + pow(lc2, 2) + 2 * l1 * lc2 * c2) + Il2
    # B12 = ml2 * (pow(lc2, 2) + l1 * lc2 * c2) + Il2
    # B22 = ml2 * pow(lc2, 2) + Il2
    #
    # B = np.array([[B11, B12],
    #      [B12, B22]]).reshape((2,2))

    B = B_inertia(theta)
    C = C_vel(theta)
    q_dot = np.array([[theta[1]], [theta[3]]]).reshape((2, 1))
    C_tau = np.matmul(C, q_dot).reshape((2,1))
    gq = gq_gravity(theta)
    fc = fc_friction(theta)

    acc = np.matmul(np.linalg.inv(B), (u - C_tau - gq -fc)).reshape((2,1))

    q = np.array([[theta[0]], [theta[2]]]).reshape((2,1))
    q_dot_n = q_dot + acc * 0.1
    q_n = q + q_dot * 0.01
    return [q_n[0], q_n[1], q_dot_n[0], q_dot_n[1]]

def RobotDynamics(theta, theta_d, Kp, Kd):
    c1 = math.cos(theta[0])
    c2 = math.cos(theta[2])
    B11 = ml1 * pow(lc1, 2) + Il1 + ml2 * (pow(l1, 2) + pow(lc2, 2) + 2 * l1 * lc2 * c2) + Il2
    B12 = ml2 * (pow(lc2, 2) + l1 * lc2 * c2) + Il2
    B22 = ml2 * pow(lc2, 2) + Il2

    B = np.array([[B11, B12],
         [B12, B22]]).reshape((2,2))

    y1 = Kp[0]*(theta_d[0]-theta[0]) + Kd[1]*(theta_d[1]-theta[1]) + theta_d[2]
    y2 = Kp[0] * (theta_d[3] - theta[2]) + Kd[1]*(theta_d[4] - theta[3]) + theta_d[5]
    y = np.array([y1, y2]).reshape((2,1))
    acc = np.matmul(np.linalg.inv(B),  np.dot(B, y)).reshape((2,1))
    # acc = np.linalg.lstsq(B, sum)[0]
    q_dot = np.array([[theta[1]], [theta[3]]]).reshape((2,1))
    q = np.array([[theta[0]], [theta[2]]])
    q_dot_n = q_dot + acc * 0.01
    q_n = q + q_dot * 0.01
    return [q_n[0], q_n[1], q_dot_n[0], q_dot_n[1]]

def Jacobian(theta):
    s1 = math.sin(theta[0])
    c1 = math.cos(theta[0])
    c2 = math.cos(theta[2])
    s2 = math.sin(theta[2])

    J = np.array([[-l1*s1-lc2*s2, -lc2*s2],[l1*c1+lc2*c2, lc2*c2],[0, 0]]).reshape((3,2))
    return J


def forceCtr(theta, theta_d, C_, M_, D_, S_, force_, f_d):
    """ Force/position control: Response to force and reference errors
        Since the motion control is position control, the admittance law here will compute the position change.
        Ka: Admittance gain
        force: Current force
        f_d: Constant maximum allowed force
    """
    if force_ > f_d:
        ddq_n = theta_d[5]+(D_*(theta_d[4] - theta[3])-C_*(f_d-force_) + S_*(theta_d[3] - theta[2]))/M_  # This is impedence control
    else:
        ddq_n = theta_d[5]
    acc_d = np.array([[theta_d[2]],[ddq_n]]).reshape((2,1))
    q_dotd = np.array([[theta_d[1]], [theta_d[4]]]).reshape((2,1))
    qd = np.array([[theta_d[0]], [theta_d[3]]])
    q_dot_n = q_dotd + acc_d * 0.01
    q_n = qd + q_dotd * 0.01
    return [q_n[0], q_n[1], q_dot_n[0], q_dot_n[1],acc_d[0],acc_d[1]]

def admittanceCtr(Ka, force_, f_d, q_n_p):
    """ Admittance control: Only response to force
        Since the motion control is position control, the admittance law here will compute the position change.
        Ka: Admittance gain
        force: Current force
        f_d: Constant maximum allowed force
        q_n_p: for integration of new position reference when using velocity/admittance control
    """
    if force_ > f_d:
        dq_n = -Ka*(f_d-force_)   # Velocity
        # q_n = -Ka * (f_d - force_)  # Position
    else:
        dq_n = 0
        # q_n = 0.
    q_dotd = np.array([[0.], [dq_n]]).reshape((2,1))
    q_n = q_n_p + q_dotd * 0.01
    # q_dotd = np.array([[0.], [0.]]).reshape((2,1))
    # q_n = np.array([[0.], [q_n]]).reshape((2,1))
    return [q_n[0], q_n[1], q_dotd[0], q_dotd[1], [0.], [0.]]

def admittanceCtr_two(Mdt, Kd, force_, f_d1, f_d, q_n_p):
    """ Admittance control: Only response to force
    ddq_f = Md-1(tau_e)-Kd*qd_f
        Since the motion control is position control, the admittance law here will compute the position change.
        Ka: Admittance gain
        force: Current force
        f_d: Constant desired force
        q_n_p: for integration of new position reference when using velocity/admittance control
    """
    if force_ > np.abs(f_d1):
        ddq_n = -Mdt*(np.abs(f_d1)-force_)-Kd*q_n_p[3][0]
    else:
        ddq_n = Mdt * (f_d - force_) - Kd * q_n_p[3][0]
        # ddq_n = 0.
    q_ddot = np.array([[0.], [ddq_n]]).reshape((2, 1))
    q_dotd = np.array(q_n_p[2:4]).reshape((2,1))
    dq_n = q_dotd + q_ddot*0.01
    q_n = q_n_p[0:2] + dq_n * 0.01
    # q_dotd = np.array([[0.], [0.]]).reshape((2,1))
    # q_n = np.array([[0.], [q_n]]).reshape((2,1))
    return [q_n[0], q_n[1], dq_n[0], dq_n[1], [0.], [ddq_n]]

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
        # [q[t+1][0], q[t+1][1], q_dot[t+1][0], q_dot[t+1][1]] = RobotDynamics(B, C_tau, gq, fc, theta, tau)
    # plt.plot(q_dot.T[0], 'r-')
    # # # # plt.plot(r.T[1], 'g-')
    # plt.show()
    return 0

main_loop()

