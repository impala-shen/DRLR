#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/chen/aem_project/aem_rl')
import numpy as np
from Controller.controller import Controller
from utils import aem_dynamics
from typing import List

class InvDynamicController(Controller):

    def __init__(self, dt: float, Kp, Kd):
        self.dt = dt
        self.y = np.zeros((2, 1))
        self.tau = np.zeros((2, 1))
        self.qr_p = np.zeros((2, 1))
        self.dqr_p = np.zeros((2, 1))
        self.dqr = np.zeros((2, 1))
        self.ddqr = np.zeros((2, 1))
        self.first_run = True

        # Diagonal gain matrix
        if isinstance(Kp, list):
            self.Kp = np.diagflat(Kp)
            self.Kd = np.diagflat(Kd)
        elif isinstance(Kp, float):
            self.Kp = np.eye(2) * Kp
            self.Kd = np.eye(2) * Kd
        else:
            raise TypeError("S, T are expected to be a list or a float.")

    def reset(self):
        pass

    def update(self, target: List[float], theta: List[float]):
        q = np.array((theta[0], theta[2])).reshape((2, 1))
        dq = np.array((theta[1], theta[3])).reshape((2, 1))
        qr = np.array((target[0], target[3])).reshape((2, 1))
        self.dqr = np.array((target[1], target[4])).reshape((2, 1))
        self.ddqr = np.array((target[2], target[5])).reshape((2, 1))
        # if np.linalg.norm((qr-self.qr_p)) > 1e-4:
        #     self.dqr = (qr - self.qr_p)/self.dt
        # if np.linalg.norm((self.dqr - self.dqr_p)) > 1e-4:
        #     self.ddqr = (self.dqr - self.dqr_p) / self.dt
        # self.dqr_p = self.dqr
        # self.qr_p = qr

        e = q - qr
        dot_e = dq - self.dqr
        B = aem_dynamics.B_inertia(theta)
        C = aem_dynamics.C_vel(theta)
        C_tau = C.T @ dq
        gq = aem_dynamics.gq_gravity(theta).reshape((2, 1))
        fc = aem_dynamics.fc_friction(theta).reshape((2, 1))

        self.y = self.ddqr - self.Kp@e - self.Kd@dot_e
        self.tau = B@self.y + C_tau + gq + fc
        # print('e', e)
        # print('qr', qr)
        # print('tau', self.dqr)

    def get_torque(self):
        return self.tau

def main():
    S = [50, 18]
    T = [120, 80]
    SOSM = InvDynamicController(S, T)
    SOSM.update()

if __name__ == "__main__":
    main()
