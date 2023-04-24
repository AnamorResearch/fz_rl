import numpy as np


class simpleModel():
    def __init__(self, exp_traj, v0=1.0, u0=1.0, P=1.0, a=1.0, d0=1.0, T=0.001, d=0):
        self.d0 = d0
        self.d = d
        self.P = P
        self.a = a
        self.dt = T

        self.v = v0
        self.u = u0
        self.t = 0

        self.U = []

        self.ideal_d = [1.0] * 100 + [1.0 - 0.8 * t / 200 for t in range(200)] + [0.2 + 0.8 * t / 200 for t in range(200)] + [1.0] * 500

        self.sample_d = exp_traj[:, 0]
        self.sample_p = exp_traj[:, 1]
        self.sample_v = exp_traj[:, 2]

    def reset(self):
        self.d = self.d0
        self.t = 0
        self.U = []

        return np.array([self.d0, -0.2])

    def step(self, P, v, u=1.0):
        self.P = P
        self.v = v

        self.u = u

        self.U.append(np.array([self.d, self.u, self.P, self.v]))

        dd = 2 / (self.a * self.P) * (self.v * self.d0 - self.u * self.d) * self.dt
        self.d += dd

        self.t += 1

        is_end = self.t >= 999

        fase = -0.2
        if self.t >= 500:
            fase = 0.2
        elif self.t >= 300:
            fase = 0.1
        elif self.t >= 100:
            fase = -0.1

        next_state = np.array([self.d, fase])
        reward = self.reward(self.d, P, v)

        return np.array([next_state, reward, is_end])

    def reward(self, state, P, v):
        ideal_d = self.ideal_d[self.t]
        exp_p = self.sample_p[self.t]
        exp_v = self.sample_v[self.t]
        return -(ideal_d - state)**2 - (exp_p - P)**2 - (exp_v - v)**2
