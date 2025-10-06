#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/chen/aem_project/aem_rl')
import numpy as np
from Estimator.estimator import Estimator
from utils import aem_dynamics_mob
from typing import List

class SMMomentumObserver(Estimator):

    def __init__(self, dt : float, S, T):
        self.dt = dt
        self.internal_r_sum = np.zeros((2, 1))
        self.p_hat = np.zeros((2, 1))
        self.sigma = np.zeros((2, 1))
        self.first_run = True

        # Diagonal gain matrix
        if isinstance(S, list):
            self.S = np.diagflat(S)
            self.T = np.diagflat(T)
        elif isinstance(S, float):
            self.S = np.eye(2) * S
            self.T = np.eye(2) * T
        else:
            raise TypeError("S, T are expected to be a list or a float.")

    def reset(self):
        pass

    def update(self, theta: List[float], u_nom: List[float]):
        B = aem_dynamics_mob.B_inertia(theta)
        C = aem_dynamics_mob.C_vel(theta)
        dq = np.array((theta[1], theta[3])).reshape((2, 1))
        C_tau = C.T @ dq
        gq = aem_dynamics_mob.gq_gravity(theta).reshape((2, 1))
        fc = aem_dynamics_mob.fc_friction(theta).reshape((2, 1))
        p = B@dq
        e1 = p - self.p_hat
        phat_dot = np.array(u_nom).reshape((2, 1)) + C_tau.reshape((2, 1)) - gq - fc + np.sqrt(np.linalg.norm(e1))*self.T @ np.sign(e1) + self.sigma
        sigma_dot = self.S @ np.sign(e1)
        # print('self.S', self.S)
        # print('sigma_dot ', sigma_dot)
        # print('self.sigma ', self.sigma)
        self.sigma += sigma_dot * self.dt
        self.p_hat += phat_dot * self.dt
        # print('p_hat', self.p_hat)

    def get_estimate(self):
        return self.sigma

def main():
    S = [50, 18]
    T = [120, 80]
    SOSM = SMMomentumObserver(0.01, S, T)
    SOSM.update()

if __name__ == "__main__":
    main()
