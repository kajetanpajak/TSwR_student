import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        self.u_prev = 0
        '''
        q_ddot = f(x, .) + b * u
        '''
        A = np.eye(3, k=1)
        B = np.array([[0], [b], [0]])
        L = np.array([[3*p], [3*p**2], [p**3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        self.b = b
        self.eso.set_B(np.array([[0], [b], [0]]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, b=None):
        q, q_dot = x # q_dot is not observed, q is 
        self.eso.update(q, self.u_prev)
        if b is not None:
            self.set_b(b)
        q_hat, q_dot_hat, f_hat = self.eso.get_state() # q_dot must be estimated

        v = q_d_ddot + self.kp * (q_d - q) + self.kd * (q_d_dot - q_dot_hat)
        u = (v - f_hat) / self.b
        # u = float(np.asarray(u).squeeze())
        self.u_prev = u
        return u
