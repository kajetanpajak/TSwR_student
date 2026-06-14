import numpy as np

from models.manipulator_model import ManiuplatorModel
from observers.eso import ESO
from .controller import Controller


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp, 0.0, 0.01)
        self.Kp = self.gain_matrix(Kp)
        self.Kd = self.gain_matrix(Kd)
        self.u_prev = np.zeros(2)

        self.L = self.observer_gain(p)
        W = np.zeros((2, 6))
        W[:, :2] = np.eye(2)

        q0 = np.asarray(q0)
        A, B = self.system_matrices(q0[:2], q0[2:4])
        self.eso = ESO(A, B, W, self.L, q0, Tp)

    @staticmethod
    def gain_matrix(gain):
        gain = np.asarray(gain, dtype=float)
        if gain.ndim == 0:
            return gain * np.eye(2)
        if gain.ndim == 1:
            return np.diag(gain)
        return gain

    @staticmethod
    def observer_gain(p):
        p = np.asarray(p, dtype=float)
        if p.ndim == 0:
            p = np.array([p, p])

        L = np.zeros((6, 2))
        for i in range(2):
            L[i, i] = 3 * p[i]
            L[i + 2, i] = 3 * p[i] ** 2
            L[i + 4, i] = p[i] ** 3
        return L

    def system_matrices(self, q, q_dot):
        x = np.concatenate([q, q_dot])
        M = self.model.M(x)
        C = self.model.C(x)
        M_inv = np.linalg.inv(M)

        A = np.zeros((6, 6))
        A[:2, 2:4] = np.eye(2)
        A[2:4, 2:4] = -M_inv @ C
        A[2:4, 4:6] = np.eye(2)

        B = np.zeros((6, 2))
        B[2:4, :] = M_inv

        return A, B

    def update_params(self, q, q_dot):
        self.eso.A, self.eso.B = self.system_matrices(q, q_dot)

    def update_eso(self, q, u):
        self.eso.states.append(np.copy(self.eso.state))

        z = np.asarray(self.eso.state).reshape((6, 1))
        q = np.asarray(q).reshape((2, 1))
        u = np.asarray(u).reshape((2, 1))

        z_dot = self.eso.A @ z + self.eso.B @ u + self.eso.L @ (q - self.eso.W @ z)
        self.eso.state = (z + z_dot * self.eso.Tp).reshape(6)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        x = np.asarray(x)
        q = x[:2]
        q_dot_hat = self.eso.get_state()[2:4]

        self.update_params(q, q_dot_hat)
        self.update_eso(q, self.u_prev)

        z_hat = self.eso.get_state()
        q_dot_hat = z_hat[2:4]
        f_hat = z_hat[4:6]

        v = (
            np.asarray(q_d_ddot)
            + self.Kd @ (np.asarray(q_d_dot) - q_dot_hat)
            + self.Kp @ (np.asarray(q_d) - q)
        )

        x_hat = np.concatenate([q, q_dot_hat])
        M = self.model.M(x_hat)
        C = self.model.C(x_hat)
        u = M @ (v - f_hat) + C @ q_dot_hat

        self.u_prev = u
        return u
